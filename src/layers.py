import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import config
from utils import IoU, transform_coord


class MaxPool1s (nn.Module):
  """Max pooling layer with stride 1"""

  def __init__(self, kernel_size):
    super(MaxPool1s, self).__init__()
    self.kernel_size = kernel_size
    self.pad = kernel_size - 1

  def forward(self, x):
    padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
    pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
    return pooled_x


class EmptyLayer(nn.Module):
  """Empty layer for shortcut connection"""

  def __init__(self):
    super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
  """Detection layer
  
  @Args
    anchors: (list) list of anchor box sizes tuple
    num_classes: (int) # classes
    reso: (int) original image resolution
    ignore_thresh: (float)
    TODO: cache ?
  """

  def __init__(self, anchors, num_classes, reso, ignore_thresh):
    super(DetectionLayer, self).__init__()
    self.anchors = anchors
    self.num_classes = num_classes
    self.reso = reso
    self.ignore_thresh = ignore_thresh

  def forward(self, x):
    """
    Transform feature map into 2-D tensor. Transformation includes
    1. Re-organize tensor to make each row correspond to a bbox
    2. Transform center coordinates
      bx = sigmoid(tx) + cx
      by = sigmoid(ty) + cy
    3. Transform width and height
      bw = pw * exp(tw)
      bh = ph * exp(th)
    4. Softmax

    @Args
      x: (Tensor) feature map with size [B, (5+#classes)*3, grid_size, grid_size]
        5 => [4 offsets (xc, yc, w, h), objectness score]
        3 => # anchor boxes pixel-wise

    @Returns
      detections: (Tensor) feature map with size [B, 13*13*3, 5+#classes]
    """
    batch_size, _, grid_size, _ = x.size()
    stride = self.reso // grid_size  # no pooling used, stride is the only downsample
    num_attrs = 5 + self.num_classes  # tx, ty, tw, th, p0
    num_anchors = len(self.anchors)
    anchors = [(a[0]/stride, a[1]/stride) for a in self.anchors]  # anchor for feature map

    # Re-organize
    # [B, (5+num_classes)*3, grid_size, grid_size]
    # => [B, (5+num_classes)*3, grid_size*grid_size]
    # => [B, 3*grid_size*grid_size, (5+num_classes)]
    x = x.view(batch_size, num_attrs*num_anchors, grid_size*grid_size)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, grid_size*grid_size*num_anchors, num_attrs)
    detections = x.new(x.size()).cuda()  # detections.size() = [B, 3*13*13, (5+num_classes)]

    # Transform center coordinates
    # x_y_offset = [[0,0]*3, [0,1]*3, [0,2]*3, ..., [12,12]*3]
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)
    x_offset = torch.FloatTensor(a).view(-1, 1).cuda()
    y_offset = torch.FloatTensor(b).view(-1, 1).cuda()
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    detections[..., 0:2] = torch.sigmoid(x[..., 0:2]) + x_y_offset[..., 0:2]  # bxy = sigmoid(txy) + cxy

    # Log transform
    # anchors: [3,2] => [13*13*3, 2]
    anchors = torch.FloatTensor(anchors).cuda()
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    detections[..., 2:4] = torch.exp(x[..., 2:4]) * anchors  # bwh = pwh * exp(twh)
    detections[..., :4] *= stride  # TODO: ?

    # Softmax
    detections[..., 4:] = torch.sigmoid(x[..., 4:])

    return detections

  def loss(self, y_pred, y_true):
    """Compute loss
    
    @Args
      y_pred: (Tensor) raw offsets predicted feature map with size [bs, ([tx, ty, tw, th, p_obj]+nC)*nA, nG, nG]
      y_true: (Tensor) scaled to (0,1) true offsets annotations with size [bs, nB, [xc, yc, w, h] + label_id]
    
    @Returns
      y_true: (Tensor)
    
    @Variables  TODO: rename variables
      bs: (int) batch size
      nA: (int) number of anchors
      gs: (int) grid size
      nB: (int) number of bboxes
    """
    loss = dict()

    # 1. Prepare
    # 1.1 re-organize y_pred
    # [bs, (5+nC)*nA, gs, gs] => [bs, num_anchors, gs, gs, 5+nC]
    bs, _, gs, _ = y_pred.size()
    nA = len(self.anchors)
    nC = self.num_classes
    y_pred = y_pred.view(bs, nA, 5+nC, gs, gs)
    y_pred = y_pred.permute(0, 1, 3, 4, 2)

    # 1.3 prepare anchor boxes
    stride = self.reso // gs
    anchors = [(a[0]/stride, a[1]/stride) for a in self.anchors]
    anchor_bboxes = torch.zeros(3, 4).cuda()
    anchor_bboxes[:, 2:] = torch.Tensor(anchors)

    # 2. Build gt [tx, ty, tw, th] and masks
    # TODO: f1 score implementation
    total_num = 0
    gt_tx = torch.zeros(bs, nA, gs, gs).cuda()
    gt_ty = torch.zeros(bs, nA, gs, gs).cuda()
    gt_tw = torch.zeros(bs, nA, gs, gs).cuda()
    gt_th = torch.zeros(bs, nA, gs, gs).cuda()
    obj_mask = torch.ByteTensor(bs, nA, gs, gs).fill_(0).cuda()
    cls_mask = torch.ByteTensor(bs, nA, gs, gs, nC).fill_(0).cuda()
    for batch_idx in range(bs):
      for box_idx, y_true_one in enumerate(y_true[batch_idx]):
        total_num += 1
        gt_bbox = y_true_one[:4] * gs  # scale bbox relative to feature map
        gt_cls_label = int(y_true_one[4])

        gt_xc, gt_yc, gt_w, gt_h = gt_bbox[0:4]
        gt_i, gt_j = int(gt_xc), int(gt_yc)
        gt_box_shape = torch.Tensor([0, 0, gt_w, gt_h]).unsqueeze(0).cuda()
        anchor_ious = IoU(gt_box_shape, anchor_bboxes, format='center')
        best_anchor = np.argmax(anchor_ious)
        anchor_w, anchor_h = anchors[best_anchor]

        gt_tw[batch_idx, best_anchor, gt_i, gt_j] = torch.log(gt_w / anchor_w + 1e-16)
        gt_th[batch_idx, best_anchor, gt_i, gt_j] = torch.log(gt_h / anchor_h + 1e-16)
        gt_tx[batch_idx, best_anchor, gt_i, gt_j] = gt_xc - gt_i
        gt_ty[batch_idx, best_anchor, gt_i, gt_j] = gt_yc - gt_j

        obj_mask[batch_idx, best_anchor, gt_i, gt_j] = 1
        cls_mask[batch_idx, best_anchor, gt_i, gt_j, gt_cls_label] = 1

    # 3. activate raw y_pred
    pred_tx = torch.sigmoid(y_pred[..., 0])  # gt tx/ty are not deactivated
    pred_ty = torch.sigmoid(y_pred[..., 1])
    pred_tw = y_pred[..., 2]
    pred_th = y_pred[..., 3]
    pred_conf = y_pred[..., 4]  # no activation because BCELoss has sigmoid layer
    pred_cls = y_pred[..., 5:]

    # 4. Compute loss
    cls_mask = cls_mask[obj_mask]
    nM = obj_mask.sum().float()  # number of anchors (assigned to targets)]
    k = nM / total_num

    MSELoss = nn.MSELoss(size_average=True)
    BCEWithLogitsLoss = nn.BCEWithLogitsLoss(size_average=True)
    CrossEntropyLoss = nn.CrossEntropyLoss()

    if nM > 0:
      loss['x'] = k * MSELoss(pred_tx[obj_mask], gt_tx[obj_mask])
      loss['y'] = k * MSELoss(pred_ty[obj_mask], gt_ty[obj_mask])
      loss['w'] = k * MSELoss(pred_tw[obj_mask], gt_tw[obj_mask])
      loss['h'] = k * MSELoss(pred_th[obj_mask], gt_th[obj_mask])
      loss['conf'] = k * BCEWithLogitsLoss(pred_conf, obj_mask.float())
      loss['cls'] = k * CrossEntropyLoss(pred_cls[obj_mask], torch.argmax(cls_mask, 1))
    else:
      loss['x'] = 0
      loss['y'] = 0
      loss['w'] = 0
      loss['h'] = 0
      loss['conf'] = 0
      loss['cls'] = 0

    cache = dict()
    cache['nM'] = nM
    cache['total_num'] = total_num

    return loss, cache


class NMSLayer(nn.Module):
  """
  NMS layer which performs Non-maximum Suppression
  1. Filter background
  2. Get detection with particular class
  3. Sort by confidence
  4. Suppress non-max detection

  @Args    
    conf_thresh: (float) fore-ground confidence threshold, default 0.5
    nms_thresh: (float) nms threshold, default 0.5
  """

  def __init__(self, conf_thresh=0.5, nms_thresh=0.5):
    super(NMSLayer, self).__init__()
    self.conf_thresh = conf_thresh
    self.nms_thresh = nms_thresh

  def forward(self, x):
    """
    @Args
      x: (Tensor) detection feature map, with size [batch_idx, num_bboxes, [x,y,w,h,p_obj]+num_classes]

    @Returns
      detections: (Tensor) detection result with size [num_bboxes, [image_batch_idx, 4 offsets, p_obj, max_conf, cls_idx]]
    """
    batch_size = x.size(0)
    conf_mask = (x[..., 4] > self.conf_thresh).float().unsqueeze(2)
    x = x * conf_mask
    x[..., :4] = transform_coord(x[..., :4], src='center', dst='corner')
    detections = torch.Tensor()

    for idx in range(batch_size):
      # 1. Filter low confidence prediction
      pred = x[idx]
      max_score, max_idx = torch.max(pred[:, 5:], 1)  # max score in each row
      max_idx = max_idx.float().unsqueeze(1)
      max_score = max_score.float().unsqueeze(1)
      pred = torch.cat((pred[:, :5], max_score, max_idx), 1)
      non_zero_pred = pred[torch.nonzero(pred[:, 4]).squeeze(), :].view(-1, 7)

      if non_zero_pred.size(0) == 0:  # no objects detected
        continue

      classes = torch.unique(non_zero_pred[:, -1])
      for cls in classes:
        # 2. Get prediction with particular class
        cls_mask = non_zero_pred * (non_zero_pred[:, -1] == cls).float().unsqueeze(1)
        cls_idx = torch.nonzero(cls_mask[:, -2]).squeeze()
        cls_pred = non_zero_pred[cls_idx].view(-1, 7)

        # 3. Sort by confidence
        conf_sort_idx = torch.sort(cls_pred[:, 4], descending=True)[1]
        cls_pred = cls_pred[conf_sort_idx]

        # 4. Suppress non-maximum
        # TODO: avoid duplicate computation
        for i in range(cls_pred.size(0)):
          try:
            ious = IoU(cls_pred[i].unsqueeze(0), cls_pred[i+1:])
          except ValueError:
            break
          except IndexError:
            break

          iou_mask = (ious < self.nms_thresh).float().unsqueeze(1)
          cls_pred[i+1:] *= iou_mask
          non_zero_idx = torch.nonzero(cls_pred[:, 4]).squeeze()
          cls_pred = cls_pred[non_zero_idx].view(-1, 7)

        batch_idx = cls_pred.new(cls_pred.size(0), 1).fill_(idx)
        seq = (batch_idx, cls_pred)
        detections = torch.cat(seq, 1) if detections.size(0) == 0 else torch.cat((detections, torch.cat(seq, 1)))

    return detections

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import config
from utils import IoU, transform_coord


class MaxPool1s(nn.Module):
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
        5 => [4 offsets (xc, yc, w, h), objectness score]
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
    detections[..., 4] = torch.sigmoid(detections[..., 4])
    detections[..., 5:] = F.softmax(detections[..., 5:])

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
    gt_tx = torch.zeros(bs, nA, gs, gs, requires_grad=False)
    gt_ty = torch.zeros(bs, nA, gs, gs, requires_grad=False)
    gt_tw = torch.zeros(bs, nA, gs, gs, requires_grad=False)
    gt_th = torch.zeros(bs, nA, gs, gs, requires_grad=False)
    obj_mask = torch.zeros(bs, nA, gs, gs, requires_grad=False)
    non_obj_mask = torch.ones(bs, nA, gs, gs, requires_grad=False)
    cls_mask = torch.zeros(bs, nA, gs, gs, nC, requires_grad=False)
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
        non_obj_mask[batch_idx, anchor_ious > 0.5] = 0  # FIXME: 0.5 as variable
        cls_mask[batch_idx, best_anchor, gt_i, gt_j, gt_cls_label] = 1

    # 3. activate raw y_pred
    pred_tx = torch.sigmoid(y_pred[..., 0])  # gt tx/ty are not deactivated
    pred_ty = torch.sigmoid(y_pred[..., 1])
    pred_tw = y_pred[..., 2]
    pred_th = y_pred[..., 3]
    pred_conf = y_pred[..., 4]
    pred_cls = y_pred[..., 5:]

    # 4. Compute loss
    obj_mask = obj_mask.cuda()
    non_obj_mask = non_obj_mask.cuda()
    cls_mask = cls_mask.cuda()
    gt_tx, gt_ty = gt_tx.cuda(), gt_ty.cuda()
    gt_tw, gt_th = gt_tw.cuda(), gt_th.cuda()

    # average over batch
    MSELoss = nn.MSELoss()
    BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    BCELoss = nn.BCELoss()
    CrossEntropyLoss = nn.CrossEntropyLoss()

    loss['x'] = MSELoss(pred_tx[obj_mask == 1], gt_tx[obj_mask == 1])
    loss['y'] = MSELoss(pred_ty[obj_mask == 1], gt_ty[obj_mask == 1])
    loss['w'] = MSELoss(pred_tw[obj_mask == 1], gt_tw[obj_mask == 1])
    loss['h'] = MSELoss(pred_th[obj_mask == 1], gt_th[obj_mask == 1])
    loss['cls'] = CrossEntropyLoss(pred_cls[obj_mask == 1], torch.argmax(cls_mask[obj_mask == 1], 1))
    loss['conf'] = BCEWithLogitsLoss(pred_conf[obj_mask == 1], obj_mask[obj_mask == 1])
    loss['non_conf'] = BCEWithLogitsLoss(pred_conf[non_obj_mask == 1], non_obj_mask[non_obj_mask == 1])

    cache = dict()

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

  def __init__(self, conf_thresh=0.5, nms_thresh=0.4):
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
    batch_size, num_bboxes, num_attrs = x.size()
    conf_mask = (x[..., 4] > self.conf_thresh).float().unsqueeze(2)
    x = x * conf_mask
    detections = torch.Tensor().cuda()

    for idx in range(batch_size):
      # 1. Filter low confidence prediction
      pred = x[idx]
      pred[:, :4] = transform_coord(pred[:, :4], src='center', dst='corner')
      pred = pred[pred[:, 4].nonzero().squeeze(1)]
      max_score, max_idx = torch.max(pred[:, 5:], 1)  # max score in each row
      max_idx = max_idx.float().unsqueeze(1)
      max_score = max_score.float().unsqueeze(1)
      pred = torch.cat((pred[:, :5], max_score, max_idx), 1)
      pred = pred[pred[:, 5] > 0.3]  # FIXME: 0.3 as variable

      if pred.size(0) == 0:  # no objects detected
        continue

      classes = torch.unique(pred[:, -1])
      for cls in classes:
        cls_pred = pred[pred[:, -1] == cls]
        conf_sort_idx = torch.sort(cls_pred[:, 4], descending=True)[1]
        cls_pred = cls_pred[conf_sort_idx]

        max_preds = []
        while cls_pred.size(0) > 0:
          max_preds.append(cls_pred[0].unsqueeze(0))
          ious = IoU(max_preds[-1], cls_pred)
          cls_pred = cls_pred[ious < self.nms_thresh]

        if len(max_preds) > 0:
          max_preds = torch.cat(max_preds).data
          batch_idx = max_preds.new(max_preds.size(0), 1).fill_(idx)
          seq = (batch_idx, max_preds)
          detections = torch.cat(seq, 1) if detections.size(0) == 0 else torch.cat((detections, torch.cat(seq, 1)))
      
    return detections

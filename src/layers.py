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
  """Detection layer"""

  def __init__(self, anchors, num_classes, reso, ignore_thresh):
    """
    Parameters
      anchors: (list) list of anchor box sizes tuple
      num_classes: (int) # classes
      reso: (int) original image resolution
      ignore_thresh: (float)
    TODO: cache ?
    """
    super(DetectionLayer, self).__init__()
    self.anchors = anchors
    self.num_classes = num_classes
    self.reso = reso
    self.ignore_thresh = ignore_thresh
    self.cache = dict()

  def forward(self, x):
    """Transform feature map into 2-D tensor. Transformation includes
    1. Re-organize tensor to make each row correspond to a bbox
    2. Transform center coordinates
      bx = sigmoid(tx) + cx
      by = sigmoid(ty) + cy
    3. Transform width and height
      bw = pw * exp(tw)
      bh = ph * exp(th)
    4. Softmax

    Parameters
    ----------
    x: (Tensor) feature map with size [B, (5+#classes)*3, grid_size, grid_size]
      5 => [4 offsets (xc, yc, w, h), objectness score]
      3 => # anchor boxes pixel-wise

    Returns
    -------
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
    detections = x.new(x.size())  # detections.size() = [B, 3*13*13, (5+num_classes)]

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
    detections[..., :4] *= stride

    # Softmax
    detections[..., 4:] = torch.sigmoid((x[..., 4:]))
    self.cache['detections'] = detections

    return detections

  def loss(self, y_pred, y_true):
    """Loss function for detection result
    1. Prepare data format
    2. Built gt [tx, ty, tw, th] and masks  TODO: Forward loss computation?
    3. Compute loss

    Parameters
    ----------
    y_pred: (Tensor) raw offsets predicted feature map with size
      [B, ([tx, ty, tw, th, p_obj]+num_classes)*num_anchors, grid_size, grid_size]
    y_true: (Tensor) scaled to (0,1) true offsets annotations with size
      [B, num_bboxes, [xc, yc, w, h] + label_id]
    lambda_coord: (float) coordinates loss weights

    Variables TODO: explain parameters in function
    ---------
    mask: 
    conf_mask:
    cls_mask:
    conf_obj:
    gt_box_shape:
    gt_cls:
    anchor_bboxes:
    gt_bbox: (Tensor) true bbox scaled to grid's size, with size
      [xc, yc, w, h]
    """
    loss = dict()
    correct_num = 0
    total_num = 0

    # 1. Prepare
    # 1.1 re-organize y_pred
    # [B, (5+num_classes) * num_anchors, grid_size, grid_size]
    # => [B, num_anchors, grid_size, grid_size, 5+num_classes]
    batch_size, _, grid_size, _ = y_pred.size()
    num_anchors = len(self.anchors)
    num_attrs = 5 + self.num_classes
    y_pred = y_pred.view(batch_size, num_anchors, num_attrs, grid_size, grid_size)
    y_pred = y_pred.permute(0, 1, 3, 4, 2)
    y_pred_activate = self.cache['detections'].view(batch_size, grid_size, grid_size, num_anchors, num_attrs)
    y_pred_activate = y_pred_activate.permute(0, 3, 1, 2, 4)

    # 1.2 scale bbox relative to feature map
    y_true[..., :4] *= grid_size

    # 1.3 prepare anchor boxes
    stride = self.reso // grid_size
    anchors = [(a[0]/stride, a[1]/stride) for a in self.anchors]
    anchor_bboxes = torch.zeros(3, 4).cuda()
    anchor_bboxes[:, 2:] = torch.Tensor(anchors)

    # TODO: vectorize
    # 2. Build gt [tx, ty, tw, th] and masks
    gt_tx = torch.zeros(batch_size, num_anchors, grid_size, grid_size).cuda()
    gt_ty = torch.zeros(batch_size, num_anchors, grid_size, grid_size).cuda()
    gt_tw = torch.zeros(batch_size, num_anchors, grid_size, grid_size).cuda()
    gt_th = torch.zeros(batch_size, num_anchors, grid_size, grid_size).cuda()
    gt_obj = torch.zeros(batch_size, num_anchors, grid_size, grid_size).cuda()
    gt_cls = torch.zeros(batch_size, num_anchors, grid_size, grid_size, self.num_classes).cuda()
    mask = torch.zeros(batch_size, num_anchors, grid_size, grid_size).cuda()
    conf_mask = torch.ones(batch_size, num_anchors, grid_size, grid_size).cuda()
    cls_mask = torch.ones(batch_size, num_anchors, grid_size, grid_size, self.num_classes).cuda()
    for batch_idx in range(batch_size):
      for box_idx in range(y_true.size(1)):
        y_true_one = y_true[batch_idx, box_idx, ...]
        gt_bbox = y_true_one[:4]
        gt_cls_label = int(y_true_one[4])
        if y_true_one.sum() == 0:  # redundancy label
          break
        else:
          total_num += 1

        gt_xc, gt_yc = int(gt_bbox[0]), int(gt_bbox[1])
        gt_w, gt_h = gt_bbox[2], gt_bbox[3]

        # find resposible anchor box in current cell
        gt_box_shape = torch.Tensor([0, 0, gt_w, gt_h]).unsqueeze(0).cuda()
        anchor_ious = IoU(gt_box_shape, anchor_bboxes, format='center')
        conf_mask[batch_idx, anchor_ious > self.ignore_thresh] = 0
        best_anchor = np.argmax(anchor_ious)
        anchor_w, anchor_h = anchors[best_anchor]
        gt_tw[batch_idx, best_anchor, gt_xc, gt_yc] = math.log(gt_w / anchor_w + 1e-5)
        gt_th[batch_idx, best_anchor, gt_xc, gt_yc] = math.log(gt_h / anchor_h + 1e-5)
        gt_tx[batch_idx, best_anchor, gt_xc, gt_yc] = gt_bbox[0] - gt_xc
        gt_ty[batch_idx, best_anchor, gt_xc, gt_yc] = gt_bbox[1] - gt_yc
        mask[batch_idx, best_anchor, gt_xc, gt_yc] = 1
        cls_mask[batch_idx, best_anchor, gt_xc, gt_yc, :] = 1
        conf_mask[batch_idx, best_anchor, gt_xc, gt_yc] = 1
        gt_obj[batch_idx, best_anchor, gt_xc, gt_yc] = 1
        gt_cls[batch_idx, best_anchor, gt_xc, gt_yc, gt_cls_label] = 1

        gt_bbox = gt_bbox.unsqueeze(0).cuda()
        pred_bbox = y_pred_activate[batch_idx, best_anchor, gt_xc, gt_yc, :4].unsqueeze(0).cuda()
        pred_bbox /= stride
        iou = IoU(pred_bbox, gt_bbox, format='center')
        if iou > 0.5:
          correct_num += 1

    # 3. Compute loss
    # 3.1 coordinates loss
    pred_tx = torch.sigmoid(y_pred[..., 0])  # gt tx/ty are not deactivated
    pred_ty = torch.sigmoid(y_pred[..., 1])
    pred_tw = y_pred[..., 2]
    pred_th = y_pred[..., 3]
    loss['x'] = nn.MSELoss()(pred_tx * mask, gt_tx * mask)
    loss['y'] = nn.MSELoss()(pred_ty * mask, gt_ty * mask)
    loss['w'] = nn.MSELoss()(pred_tw * mask, gt_tw * mask) / 2
    loss['h'] = nn.MSELoss()(pred_th * mask, gt_th * mask) / 2
    # 3.2 confidence loss
    pred_obj = torch.sigmoid(y_pred[..., 4])  # objectness score
    pred_cls = torch.sigmoid(y_pred[..., 5:])  # class score
    loss['conf'] = nn.MSELoss()(pred_obj * conf_mask, gt_obj * conf_mask)
    loss['cls'] = nn.MSELoss()(pred_cls * cls_mask, gt_cls * cls_mask)
    
    return loss, correct_num, total_num


class NMSLayer(nn.Module):
  """NMS layer which performs Non-maximum Suppression
    1. Filter background
    2. Get detection with particular class
    3. Sort by confidence
    4. Suppress non-max detection
  """

  def __init__(self, conf_thresh=0.8, nms_thresh=0.4):
    """
    Parameters
    ----------
    conf_thresh: (float) fore-ground confidence threshold, default 0.5
    nms_thresh: (float) nms threshold, default 0.4
    """
    super(NMSLayer, self).__init__()
    self.conf_thresh = conf_thresh
    self.nms_thresh = nms_thresh

  def forward(self, x):
    """
    Parameters
    ----------
    x: (Tensor) detection feature map, with size
      [batch_idx, num_bboxes, [x,y,w,h,p_obj]+num_classes]

    Returns
    -------
    detections: (Tensor) detection result with size
      [num_bboxes, [image_batch_idx, 4 offsets, p_obj, max_conf, cls_idx]]
    """
    batch_size = x.size(0)
    conf_mask = (x[..., 4] > self.conf_thresh).float().unsqueeze(2)
    x = x * conf_mask
    x[..., :4] = transform_coord(x[..., :4])
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

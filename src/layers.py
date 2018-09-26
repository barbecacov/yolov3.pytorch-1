import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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

  def __init__(self, anchors, num_classes, input_dim):
    """Init the model
    @args
      anchors: (list) list of anchor box sizes tuple
      num_classes: (int) # classes
      input_dim: (int) image size
    """
    super(DetectionLayer, self).__init__()
    self.anchors = anchors
    self.num_classes = num_classes
    self.input_dim = input_dim
    self.cache = dict()  # cache for computing loss

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
    @args
      x: (torch.Tensor) detection result feature map, with size [B, (5+num_classes)*3, 13, 13]
        5 => [4 offsets (xc, yc, w, h), objectness score]
        3 => # anchor boxes pixel-wise
        13 => grid size in last feature map
    @returns
      detections: (torch.Tensor) transformed feature map, with size [B, 13*13*3, 5+num_classes]
    """
    batch_size, _, grid_size, _ = x.size()
    stride = self.input_dim // grid_size  # no pooling used, stride is the only downsample
    num_attrs = 5 + self.num_classes  # tx, ty, tw, th, p0
    num_anchors = len(self.anchors)
    anchors = [(a[0]/stride, a[1]/stride) for a in self.anchors]  # anchor for feature map

    # Re-organize
    # [B, (5+num_classes)*3, 13, 13] => [B, (5+num_classes)*3, 13*13] => [B, 3*13*13, (5+num_classes)]
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

    return detections

  def loss(self, y_pred, y_true):
    """Loss function for detection result
      1. Re-organize y_pred
      2. 
    @args
      y_pred: (torch.Tensor) predicted feature map with size [B, grid_size^2*3, 5+num_classes]
        3 => # anchors
        5 => [tx, ty, tw, th] + objectness
      y_true: (torch.Tensor) annotations with size [B, 15, 5]
        15 => fixed size # bboxes
        5 => [x1, y1, x2, y2] scaled to (0,1) + 1 label
    """
    # 1. Re-organize y_pred
    # [B, grid size^2*3, 5+num_classes] => [B, grid size, grid_size, 3, 5+num_classes]
    batch_size, num_bboxes, _ = y_pred.size()
    num_anchors = len(self.anchors)
    num_attrs = 5 + self.num_classes
    grid_size = int(np.sqrt(num_bboxes / num_anchors))
    y_pred = y_pred.view(batch_size, grid_size, grid_size, num_anchors, num_attrs).contiguous()

    # 2. Object mask
    for i in range(batch_size):
      y_pred_batch = y_pred[i]
      y_true_batch = y_true[i]
      gt_bboxes = y_true_batch[:, :4]
      gt_labels = y_true_batch[:, 4].int()
      gt_bboxes = (gt_bboxes * grid_size).int()  # bbox's values are (0,1)
      pred_bboxes = y_pred_batch[..., 0:4]
      pred_bboxes[..., :4] = transform_coord(pred_bboxes[..., :4])

      from IPython import embed
      embed()


class NMSLayer(nn.Module):
  """
  NMS layer which performs Non-maximum Suppression
    1. Filter background
    2. Get detection with particular class
    3. Sort by confidence
    4. Suppress non-max detection
  """

  def __init__(self, conf_thresh=0.8, nms_thresh=0.4):
    """Init layer
    @args
      conf_thresh: (float) fore-ground confidence threshold, default 0.5
      nms_thresh: (float) nms threshold, default 0.4
    """
    super(NMSLayer, self).__init__()
    self.conf_thresh = conf_thresh
    self.nms_thresh = nms_thresh

  def forward(self, x):
    """Forward pass
    @args
      x: (torch.Tensor) detection feature map, with size [batch_idx, # bboxes, 5+num_classes]
        5 => [x, y, w, h, objectness score]
    @returns
      detections: (torch.Tensor) detection result with with [# bboxes, 8]
        8 => [image batch idx, 4 offsets, objectness, max conf, class idx]
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

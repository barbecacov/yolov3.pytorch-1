import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


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
    """
    @args
      anchors: (list) list of anchor box sizes tuple
      num_classes: (int)
      input_dim: (int)
    """
    super(DetectionLayer, self).__init__()
    self.anchors = anchors
    self.num_classes = num_classes
    self.input_dim = input_dim

  def forward(self, prediction):
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

    @args
      prediction: (torch.Tensor) detection result feature map, with size [B, 85*3, 13, 13]
        85 => [4 offsets, objectness score, 80 class score]
        3 => # anchor boxes pixel-wise
        13 => grid size in last feature map

    @return
      prediction: (torch.Tensor) transformed feature map, with size [B, 13*13*3, 85]
    """
    batch_size, _, grid_size, _ = prediction.size()
    stride = self.input_dim // grid_size  # no pooling used, stride is the only downsample
    num_attrs = 5 + self.num_classes  # tx, ty, tw, th, p0
    num_anchors = len(self.anchors)

    anchors = [(a[0]/stride, a[1]/stride) for a in self.anchors]  # anchor for feature map

    # Re-organize
    # [batch_size, 85*3, 13, 13] => [batch_size, 85*3, 13*13] => [batch_size, 3*13*13, 85]
    prediction = prediction.view(batch_size, num_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, num_attrs)

    # Transform center coordinates
    # x_y_offset = [[0,0]*3, [0,1]*3, [0,2]*3, ..., [12,12]*3]
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)
    x_offset = torch.FloatTensor(a).view(-1, 1).cuda()
    y_offset = torch.FloatTensor(b).view(-1, 1).cuda()
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    prediction[..., 0:2] = F.sigmoid(prediction[..., 0:2]) + x_y_offset[..., 0:2]  # bxy = sigmoid(txy) + cxy

    # Log transform
    # anchors: [3,2] => [13*13*3, 2]
    anchors = torch.FloatTensor(anchors).cuda()
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[..., 2:4] = torch.exp(prediction[..., 2:4])*anchors  # bwh = pwh * exp(twh)

    prediction[..., :4] *= stride

    # Softmax
    prediction[..., 4:] = F.sigmoid((prediction[..., 4:]))

    return prediction

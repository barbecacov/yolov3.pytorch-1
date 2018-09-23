import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from layers import MaxPool1s, EmptyLayer, DetectionLayer
from utils import parse_cfg, IoU, transform_coord


class YOLOv3(nn.Module):
  """YOLO v3 model"""

  def __init__(self, cfgfile, input_dim):
    """
    @args
      cfgfile: (str) path to yolo v3 config file
      input_dim: (int) 
    """
    super(YOLOv3, self).__init__()
    self.blocks = parse_cfg(cfgfile)
    self.input_dim = input_dim
    self.module_list = self.build_model(self.blocks)

  def build_model(self, blocks):
    """Build YOLOv3 model from building blocks
    @args
      blocks: (list) list of building blocks description

    @return
      module_list: (nn.ModuleList) module list of neural network
    """
    module_list = nn.ModuleList()
    in_channels = 3  # start from RGB 3 channels
    out_channels_list = []

    for idx, block in enumerate(blocks):
      module = nn.Sequential()

      # Convolutional layer
      if block['type'] == 'convolutional':
        activation = block['activation']
        try:
          batch_normalize = int(block['batch_normalize'])
          bias = False
        except:
          batch_normalize = 0
          bias = True
        out_channels = int(block['filters'])
        kernel_size = int(block['size'])
        padding = (kernel_size - 1) // 2 if block['pad'] else 0
        stride = int(block['stride'])
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        module.add_module("conv_{0}".format(idx), conv)

        if batch_normalize != 0:
          module.add_module("bn_{0}".format(idx), nn.BatchNorm2d(out_channels))

        if activation == "leaky":  # for yolo, it's either leaky ReLU or linear
          module.add_module("leaky_{0}".format(idx), nn.LeakyReLU(0.1, inplace=True))

      # Max pooling layer
      elif block['type'] == 'maxpool':
        stride = int(block["stride"])
        size = int(block["size"])
        if stride != 1:
          maxpool = nn.MaxPool2d(size, stride)
        else:
          maxpool = MaxPool1s(size)

        module.add_module("maxpool_{}".format(idx), maxpool)

      # Up sample layer
      elif block['type'] == 'upsample':
        stride = int(block["stride"])  # always to be 2 in yolo-v3
        upsample = nn.Upsample(scale_factor=stride, mode="nearest")
        module.add_module("upsample_{}".format(idx), upsample)

      # Shortcut layer
      elif block['type'] == 'shortcut':
        shortcut = EmptyLayer()
        module.add_module("shortcut_{}".format(idx), shortcut)

      # Routing layer
      elif block['type'] == 'route':
        route = EmptyLayer()
        module.add_module('route_{}'.format(idx), route)

        block['layers'] = block['layers'].split(',')
        if len(block['layers']) == 1:
          start = int(block['layers'][0])
          out_channels = out_channels_list[idx+start]
        elif len(block['layers']) == 2:
          start = int(block['layers'][0])
          end = int(block['layers'][1])
          out_channels = out_channels_list[idx+start] + out_channels_list[end]

      # Detection layer
      elif block['type'] == 'yolo':
        mask = block['mask'].split(',')
        mask = [int(x) for x in mask]

        anchors = block['anchors'].split(',')
        anchors = [int(a) for a in anchors]
        anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
        anchors = [anchors[i] for i in mask]

        num_classes = int(block['classes'])

        detection = DetectionLayer(anchors, num_classes, self.input_dim)
        module.add_module('detection_{}'.format(idx), detection)

      module_list.append(module)
      in_channels = out_channels
      out_channels_list.append(out_channels)

    return module_list

  def forward(self, x):
    """
    Forwarad pass of YOLO v3

    @args
      x: (torch.Tensor) input Tensor, with size [batch_size, C, H, W]
    
    @return
      detections: (torch.Tensor) detection in different scales, with size [batch_size, # bboxes, 85]
        # bboxes => 13 * 13 (# grid size in last feature map) * 3 (# anchor boxes) * 3 (# scales)
        85 => [4 offsets, objectness score, 80 class score]
    """
    detections = torch.Tensor()  # detection results
    outputs = dict()   # output cache for route layer

    for i, block in enumerate(self.blocks):
      # Convolutional, upsample, maxpooling layer
      if block['type'] == 'convolutional' or block['type'] == 'upsample' or block['type'] == 'maxpool':
        x = self.module_list[i](x)
        outputs[i] = x

      # Shortcut layer
      elif block['type'] == 'shortcut':
        x = outputs[i-1] + outputs[i+int(block['from'])]
        outputs[i] = x

      # Routing layer, length = 1 or 2
      elif block['type'] == 'route':
        layers = block['layers']
        layers = [int(a) for a in layers]

        if len(layers) == 1:  # [-3] means output previous 3
          x = outputs[i + (layers[0])]

        elif len(layers) == 2:  # [-1, 61] means concatnate previous 1 and No.61
          layers[1] = layers[1] - i
          map1 = outputs[i + layers[0]]
          map2 = outputs[i + layers[1]]
          x = torch.cat((map1, map2), 1)  # [BxCxHxW] cat with depth

        outputs[i] = x

      elif block['type'] == 'yolo':
        x = self.module_list[i](x)
        detections = x if len(detections.size()) == 1 else torch.cat((detections, x), 1)
        outputs[i] = outputs[i-1]  # skip

    return detections

  def load_weights(self, path):
    """
    Load weights from disk. YOLOv3 is fully convolutional, so only conv layers' weights will be loaded
    Weights data are organized as
      1. (optinoal) bn_biases => bn_weights => bn_mean => bn_var
      1. (optional) conv_bias
      2. conv_weights

    @args
      path: (str) path to weights file
    """
    fp = open(path, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)
    weights = np.fromfile(fp, dtype=np.float32)
    fp.close()

    header = torch.from_numpy(header)

    ptr = 0
    for i, module in enumerate(self.module_list):
      block = self.blocks[i]

      if block['type'] == "convolutional":
        batch_normalize = int(block['batch_normalize']) if 'batch_normalize' in block else 0
        conv = module[0]

        if batch_normalize > 0:
          bn = module[1]
          num_bn_biases = bn.bias.numel()

          bn_biases = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
          bn_biases = bn_biases.view_as(bn.bias.data)
          bn.bias.data.copy_(bn_biases)
          ptr += num_bn_biases

          bn_weights = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
          bn_weights = bn_weights.view_as(bn.weight.data)
          bn.weight.data.copy_(bn_weights)
          ptr += num_bn_biases

          bn_running_mean = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
          bn_running_mean = bn_running_mean.view_as(bn.running_mean)
          bn.running_mean.copy_(bn_running_mean)
          ptr += num_bn_biases

          bn_running_var = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
          bn_running_var = bn_running_var.view_as(bn.running_var)
          bn.running_var.copy_(bn_running_var)
          ptr += num_bn_biases

        else:
          num_biases = conv.bias.numel()
          conv_biases = torch.from_numpy(weights[ptr:ptr+num_biases])
          conv_biases = conv_biases.view_as(conv.bias.data)
          conv.bias.data.copy_(conv_biases)
          ptr = ptr + num_biases

        num_weights = conv.weight.numel()
        conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
        conv_weights = conv_weights.view_as(conv.weight.data)
        conv.weight.data.copy_(conv_weights)
        ptr = ptr + num_weights


def nms(prediction, num_classes, conf_thresh=0.5, nms_thresh=0.4):
  """
  Perform Non-maximum Suppression
    1. Filter background
    2. Get prediction with particular class
    3. Sort by confidence
    4. Suppress non-max prediction

  @args
    prediction: (torch.Tensor) prediction feature map, with size [batch_idx, # bboxes, 85]
      85 => [4 offsets, objectness score, 80 class score]
    num_class: (int)
    conf_thresh: (float) fore-ground confidence threshold, default 0.5
    nms_thresh: (float) nms threshold, default 0.4

  @return
    output: (torch.Tensor) detection result with with [# bboxes, 8]
      8 => [image batch idx, 4 offsets, objectness, max conf, class idx]
  """
  batch_size = prediction.size(0)
  conf_mask = (prediction[..., 4] > conf_thresh).float().unsqueeze(2)
  prediction = prediction * conf_mask
  prediction[...,:4] = transform_coord(prediction[...,:4])
  output = torch.Tensor()

  for idx in range(batch_size):
    # 1. Filter low confidence prediction
    pred = prediction[idx]
    max_score, max_idx = torch.max(pred[:, 5:], 1)  # max score in each row
    max_idx = max_idx.float().unsqueeze(1)
    max_score = max_score.float().unsqueeze(1)
    pred = torch.cat((pred[:, :5], max_score, max_idx), 1)
    non_zero_pred = pred[torch.nonzero(pred[:, 4]).squeeze(), :].view(-1, 7)
    
    if non_zero_pred.size(0) == 0: # no objects detected
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

        iou_mask = (ious < nms_thresh).float().unsqueeze(1)
        cls_pred[i+1:] *= iou_mask
        non_zero_idx = torch.nonzero(cls_pred[:, 4]).squeeze()
        cls_pred = cls_pred[non_zero_idx].view(-1, 7)

      batch_idx = cls_pred.new(cls_pred.size(0), 1).fill_(idx)
      seq = (batch_idx, cls_pred)
      output = torch.cat(seq, 1) if output.size(0) == 0 else torch.cat((output, torch.cat(seq, 1)))

  return output



def get_test_input():
  """Generate test image"""
  import cv2
  img = cv2.imread("../assets/test.png")
  img = cv2.resize(img, (416, 416))          # resize to the input dimension
  img_ = img[:, :, ::-1].transpose((2, 0, 1))  # BGR -> RGB | H X W C -> C X H X
  img_ = img_[np.newaxis, :, :, :]/255.0       # Add a channel at 0 (for batch) | Normalise
  img_ = torch.from_numpy(img_).float()     # Convert to float
  img_ = Variable(img_)                     # Convert to Variable
  return img_



if __name__ == '__main__':
  net = YOLOv3('../static/yolov3.cfg', 320)
  net.load_weights('../static/yolov3.weights')
  net = net.cuda()
  input = get_test_input().cuda()
  prediction = net(input)
  detection = nms(prediction, 80)
  # print(detection.size())  # [3,8]
  np.save('../static/detection.npy', detection.data.cpu().numpy())

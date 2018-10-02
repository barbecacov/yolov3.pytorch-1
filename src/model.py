import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from collections import defaultdict

from layers import MaxPool1s, EmptyLayer, DetectionLayer, NMSLayer
from utils import parse_cfg


class YOLOv3(nn.Module):
  """YOLO v3 model

  @Args
    cfgfile: (str) path to yolo v3 config file  
    reso: (int) original image resolution  
  """

  def __init__(self, cfgfile, reso):
    super(YOLOv3, self).__init__()
    self.blocks = parse_cfg(cfgfile)
    self.reso = reso
    self.cache = dict()  # cache for computing loss
    self.module_list = self.build_model(self.blocks)
    self.nms = NMSLayer()

  def build_model(self, blocks):
    """Build YOLOv3 model from building blocks

    @Args
      blocks: (list) list of building blocks description

    @Returns
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
        ignore_thresh = float(block['ignore_thresh'])

        detection = DetectionLayer(anchors, num_classes, self.reso, ignore_thresh)
        module.add_module('detection_{}'.format(idx), detection)

      module_list.append(module)
      in_channels = out_channels
      out_channels_list.append(out_channels)

    return module_list

  def forward(self, x):
    """Forwarad pass of YOLO v3

    @Args
      x: (Tensor) input Tensor, with size[batch_size, C, H, W]

    @Variables
      self.cache: (dict) cache of raw detection result, each with size [batch_size, num_bboxes, [xc, yc, w, h, p_ob]+num_classes]
      TODO: cache ?  

    @Returns
      detections: (Tensor) detection result with size [num_bboxes, [batch idx, x1, y1, x2, y2, p0, conf, label]]
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

        if len(layers) == 1:  # layers = [-3]: output layer -3
          x = outputs[i + (layers[0])]

        elif len(layers) == 2:  # layers = [-1, 61]: cat layer -1 and No.61
          layers[1] = layers[1] - i
          map1 = outputs[i + layers[0]]
          map2 = outputs[i + layers[1]]
          x = torch.cat((map1, map2), 1)  # cat with depth

        outputs[i] = x

      elif block['type'] == 'yolo':
        self.cache[i] = x  # cache for loss
        x = self.module_list[i](x)
        detections = x if len(detections.size()) == 1 else torch.cat((detections, x), 1)
        outputs[i] = outputs[i-1]  # skip

    detections = self.nms(detections)

    return detections

  def loss(self, y_true):
    """Compute loss

    @Args
      y_true: (Tensor) annotations with size [bs, num_bboxes, 5=(xc, yc, w, h, label_id)]

    @Variables
      y_pred: (Tensor) raw detections with size [bs, ([tx,ty,tw,th,p_obj]+num_classes)*3, grid_size, grid_size]
    """
    losses = defaultdict(float)
    for i, y_pred in self.cache.items():
      block = self.blocks[i]
      assert block['type'] == 'yolo'
      loss, cache = self.module_list[i][0].loss(y_pred, y_true.clone())
      for name in loss.keys():
        losses[name] += loss[name]
        losses['total'] += loss[name]
    return losses, cache

  def load_weights(self, path, cut=None):
    """Load darknet weights from disk.
    YOLOv3 is fully convolutional, so only conv layers' weights will be loaded
    Darknet's weights data are organized as
      1. (optinoal) bn_biases => bn_weights => bn_mean => bn_var
      1. (optional) conv_bias
      2. conv_weights

    @Args
      path: (str) path to .weights file
      cut: (optinoal, int) cutting layer
    """
    fp = open(path, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)
    weights = np.fromfile(fp, dtype=np.float32)
    fp.close()

    header = torch.from_numpy(header)

    ptr = 0
    for i, module in enumerate(self.module_list):
      block = self.blocks[i]

      if cut is not None and i == cut:
        print("Stop before", block['type'], "block (No.%d)" % (i+1))
        break

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


if __name__ == '__main__':
  import os
  import config
  from utils import save_checkpoint
  opj = os.path.join

  model = YOLOv3(config.network['coco']['cfg'], 416).cuda()

  # transfer pretrained darknets
  model.load_weights('../checkpoints/darknet/darknet53.conv.74.weights', cut=74)
  save_checkpoint(opj(config.CKPT_ROOT), 0, {
    'epoch': 0,
    'state_dict': model.state_dict(),
    'mAP': 0  # FIXME: ?
  })

  model = YOLOv3(config.network['coco']['cfg'], 416).cuda()

  # transfer pretrained yolo
  model.load_weights('../checkpoints/darknet/yolov3-coco.weights', cut=74)
  save_checkpoint(opj(config.CKPT_ROOT, 'coco'), -1, {
    'epoch': -1,
    'state_dict': model.state_dict(),
    'mAP': 0  # FIXME: ?
  })

"""Archive functions or objects"""

def load_weights(self, path):
  """
  Load weights from disk. YOLOv3 is fully convolutional, so only conv layers' weights will be loaded
  Darknet's weights data are organized as
    1. (optinoal) bn_biases => bn_weights => bn_mean => bn_var
    1. (optional) conv_bias
    2. conv_weights

  @Args
    path: (str) path to .weights file
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

"""Archive functions or objects"""

# model.py
# YOLOv3
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

# layers.py
# DetectionLayer
def loss(self, y_pred, y_true):
  """Loss function for detection result
  1. Prepare data format
  2. Built gt [tx, ty, tw, th] and masks  TODO: Forward loss computation?
  3. Compute loss

  @Args
    y_pred: (Tensor) raw offsets predicted feature map with size
      [B, ([tx, ty, tw, th, p_obj]+num_classes)*num_anchors, grid_size, grid_size]
    y_true: (Tensor) scaled to (0,1) true offsets annotations with size
      [B, num_bboxes, [xc, yc, w, h] + label_id]
    lambda_coord: (float) coordinates loss weights

  @Varibles TODO: explain parameters in function
    mask: 
    conf_mask:
    cls_mask:
    conf_obj:
    gt_box_shape:
    gt_cls:
    anchor_bboxes:
    gt_bbox: (Tensor) true bbox scaled to grid's size, with size [xc, yc, w, h]
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
  pred_cls = torch.sigmoid(y_pred[..., 5:]) # class score
  loss['conf'] = nn.MSELoss()(pred_obj * conf_mask, gt_obj * conf_mask)
  loss['cls'] = nn.MSELoss()(pred_cls * cls_mask, gt_cls * cls_mask)
  
  return loss, correct_num, total_num
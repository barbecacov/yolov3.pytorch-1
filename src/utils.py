import os
import torch
import datetime
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
opj = os.path.join

import config


def parse_cfg(cfgfile):
  """Parse a configuration file
  @args
    cfgfile: (str) path to config file
  @returns
    blocks: (list) list of blocks, with each block describes a block in the NN to be built
  """
  file = open(cfgfile, 'r')
  lines = file.read().split('\n')  # store the lines in a list
  lines = [x for x in lines if len(x) > 0]  # skip empty lines
  lines = [x for x in lines if x[0] != '#']  # skip comment
  lines = [x.rstrip().lstrip() for x in lines]
  file.close()

  block = {}
  blocks = []

  for line in lines:
    if line[0] == "[":  # This marks the start of a new block
      if len(block) != 0:
        blocks.append(block)
        block = {}
      block['type'] = line[1:-1].rstrip()
    else:
      key, value = line.split("=")
      block[key.rstrip()] = value.lstrip()
  blocks.append(block)

  return blocks


def transform_coord(bbox):
  """Transform bbox coordinates
    |---------|           (x1,y1) *---------|
    |         |                   |         |
    |  (x,y)  | h    ===>         |         |
    |         |                   |         |
    |_________|                   |_________* (x2,y2)
        w
  @args
    bbox: (torch.Tensor) bbox with size [batch_size, # bboxes, 4]
      4 = [center x, center y, height, width] 
  @returns
    bbox_transformed: (torch.Tensor) transformed bbox with size [batch_size, # bboxes, 4]
      4 = [top-left x, top-left y, right-bottom x, right-bottom y]
  """
  bbox_transformed = bbox.new(bbox.size())
  bbox_transformed[..., 0] = (bbox[..., 0] - bbox[..., 2]/2)
  bbox_transformed[..., 1] = (bbox[..., 1] - bbox[..., 3]/2)
  bbox_transformed[..., 2] = (bbox[..., 0] + bbox[..., 2]/2)
  bbox_transformed[..., 3] = (bbox[..., 1] + bbox[..., 3]/2)
  return bbox_transformed


def IoU(box1, box2, format='corner'):
  """Compute IoU between box1 and box2
  @args
    box: (torch.Tensor) bboxes with size [# bboxes, 4]
    format: (str) bbox format
      'corner' => [x1, y1, x2, y2]
      'center' => [xc, yc, w, h]
  """
  if format == 'center':
    box1 = transform_coord(box1)
    box2 = transform_coord(box2)

  b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
  b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

  inter_rect_x1 = torch.max(b1_x1, b2_x1)
  inter_rect_y1 = torch.max(b1_y1, b2_y1)
  inter_rect_x2 = torch.min(b1_x2, b2_x2)
  inter_rect_y2 = torch.min(b1_y2, b2_y2)

  inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape).cuda())*torch.max(inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda())
  b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
  b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

  return inter_area / (b1_area + b2_area - inter_area)


def save_detection(img_path, detection, dets_dir, reso):
  """Draw and save detection result
  @args
    img_path: (str) path to image
    detection: (np.array) detection result, with size [#bbox, 8]
      8 = [batch_idx, top-left x, top-left y, bottom-right x, bottom-right y, objectness, conf, class idx]
    dets_dir: (str) detection result save path
    reso: (int) image resolution
  """
  class_names = config.datasets['coco']['class_names']
  img_name = img_path.split('/')[-1]

  img = Image.open(img_path)
  w, h = img.size
  h_ratio = h / reso
  w_ratio = w / reso
  h_ratio, w_ratio
  draw = ImageDraw.Draw(img)

  for i in range(detection.shape[0]):
    bbox = detection[i, 1:5]
    label = class_names[int(detection[i, -1])]
    conf = '%.2f' % detection[i, -2]
    caption = str(label) + ' ' + str(conf)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    draw.rectangle(((x1 * w_ratio, y1 * h_ratio, x2 * w_ratio, y2 * h_ratio)), outline='red')
    draw.text((x1 * w_ratio, y1 * h_ratio), caption, fill='red')

  img.save(opj(dets_dir, img_name))


def get_current_time():
  """Get current datetime

  @returns
    time: (str) time in format "dd-hh-mm"
  """
  time = str(datetime.datetime.now())
  time = time.split('-')[-1].split('.')[0]
  time = time.replace(' ', ':')
  day, hour, minute, _ = time.split(':')
  if day[-1] == '1':
    day += 'st'
  elif day[-1] == '2':
    day += 'nd'
  elif day[-1] == '3':
    day += 'rd'
  else:
    day += 'th'
  time = day + '-' + hour + ':' + minute
  return str(time)

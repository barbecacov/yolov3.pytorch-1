import os
import json
opj = os.path.join

ROOT = '/media/data_1/home/penggao/penggao/detection/yolo3.pytorch'
LOG_ROOT = opj(ROOT, 'log')
CKPT_ROOT = opj(ROOT, 'checkpoints')

def parse_names(path):
  """Parse names .json"""
  with open(path) as json_data:
    d = json.load(json_data)
    return d


def create_category_mapping(d):
  mapping = dict()
  for idx, id in enumerate(d):
    mapping[id] = idx
  return mapping


datasets = {
    'tejani': {
        'train_root': '/media/data_2/COCO_SIXD/tejani',
        'class_names': ['coffee cup', 'shampoo', 'joystick', 'camera', 'juice carton', 'milk']
    },
    'coco': {
        'num_classes': 80,
        'train_root': '/media/data_2/COCO/2017/',
        'category_id_mapping': create_category_mapping([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]),
        'class_names': ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    }
}

network = {
    'tejani': {
        'cfg': opj(ROOT, 'lib/yolov3-tejani.cfg')
    },
    'coco': {
        'cfg': opj(ROOT, 'lib/yolov3-coco.cfg'),
        'weights': opj(ROOT, 'lib/yolov3-coco.weights')
    }
}

test = {
    'images_dir': opj(ROOT, 'assets/test_imgs'),
    'result_dir': opj(ROOT, 'assets/test_dets')
}

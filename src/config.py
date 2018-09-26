import os
import json
opj = os.path.join

ROOT_DIR = '/media/data_1/home/penggao/penggao/detection/yolo3.pytorch'


def parse_names(path):
  """Parse names .json"""
  with open(path) as json_data:
    d = json.load(json_data)
    return d


datasets = {
    'tejani': {
        'train_root': '/media/data_2/COCO_SIXD/tejani',
        'class_names': ['coffee cup', 'shampoo', 'joystick', 'camera', 'juice carton', 'milk']
    },
    'coco': {
        'num_classes': 80,
        'train_root': '/media/data_2/COCO/2017/',
        'class_names': parse_names(opj(ROOT_DIR, 'lib/coco-names.json'))
    }
}

network = {
    'tejani': {
        'cfg': opj(ROOT_DIR, 'lib/yolov3-tejani.cfg')
    },
    'coco': {
        'cfg': opj(ROOT_DIR, 'lib/yolov3-coco.cfg'),
        'weights': opj(ROOT_DIR, 'lib/yolov3-coco.weights')
    }
}

test = {
    'images_dir': opj(ROOT_DIR, 'assets/test_imgs'),
    'result_dir': opj(ROOT_DIR, 'assets/test_dets')
}

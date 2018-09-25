import os
opj = os.path.join

ROOT_DIR = '/media/data_1/home/penggao/penggao/detection/yolo3.pytorch'

datasets = {
    'tejani': {
        'train_root': '/media/data_2/COCO_SIXD/tejani',
        'class_names': []
    }
}

network = {
    'weights': opj(ROOT_DIR, 'lib/yolov3.weights'),
    'cfg': opj(ROOT_DIR, 'lib/yolov3.cfg')
}

test = {
    'images_dir': opj(ROOT_DIR, 'assets/imgs'),
    'result_dir': opj(ROOT_DIR, 'assets/dets')
}

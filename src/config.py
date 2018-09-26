import os
opj = os.path.join

ROOT_DIR = '/media/data_1/home/penggao/penggao/detection/yolo3.pytorch'

datasets = {
    'tejani': {
        'train_root': '/media/data_2/COCO_SIXD/tejani',
        'class_names': ['coffee cup', 'shampoo', 'joystick', 'camera', 'juice carton', 'milk']
    },
    'coco': {
        'num_classes': 80,
        'class_names': ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
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

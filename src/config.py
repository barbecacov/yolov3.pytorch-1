import os
opj = os.path.join

ROOT_DIR = '/media/data_1/home/penggao/penggao/detection/yolo3.pytorch'

datasets = {
    'coco': {
        'num_classes': 80,
        'class_names': ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic' 'light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    }
}

network = {
    'weights': opj(ROOT_DIR, 'static/yolov3.weights'),
    'cfg': opj(ROOT_DIR, 'static/yolov3.cfg')
}

test = {
    'images_dir': opj(ROOT_DIR, 'assets/imgs'),
    'result_dir': opj(ROOT_DIR, 'assets/dets')
}

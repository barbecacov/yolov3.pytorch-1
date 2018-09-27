# yolov3.pytorch

**❗ ATTENTION** 2018/09/25 This repo is under construction, only used for personal use

This repository contains code for a object detector based on [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf), implemented in PyTorch. This repository is based on [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3). I re-implemented it in PyTorch for better readability and re-useablity.

## File Tree 🌲

```bash
.
├── assets/           # folder of assets
│   ├── dets/           # evaluation results folder
│   └── imgs/           # evaluation images folder
├── README.md
├── src/              # folder of source file
│   ├── config.py       # configuration file
│   ├── dataset.py      # dataset
│   ├── layers.py       # supported layers for YOLO v3 model
│   ├── model.py        # YOLO v3 model
│   ├── test.py         # evaluation code
│   ├── train.py        # training code
│   └── utils.py        # utils function
└── lib               # folder of static file, like .cfg and .weights
```

## Requirements 🤔

* Python 3.6
* PyTorch **0.4** (v0.4.1 or v0.3 is not supported)
* Pillow
* Numpy
* CUDA (**CPU is not supported for both training and test!**)

## Train 🏹

To be implemented

## Evaluation 🎯

### How to run this code

1. Download pretrained YOLO v3 weights [here](https://pjreddie.com/media/files/yolov3.weights) and move it to `lib/yolov3.weights`
2. Run `python test.py`
3. You can sepcify detection images directory and detection results directory in `config.py`

### Inputs arguments

* `--reso`, image resolution. Image will be resize to `(reso, reso)` during evaluation. The higher resolution is, the more accurate the detection

### Evaluation example

![](https://raw.githubusercontent.com/ECer23/yolov3.pytorch/master/assets/dets/dog.jpg)

## TODO ✅

### Important

- [x] Evaluation on image
- [ ] Training on user specified datasets
- [ ] Metrics on evaluation result

### Not important

- [ ] CPU support

## Reference 🔍

* [Series: YOLO object detector in PyTorch](https://blog.paperspace.com/tag/series-yolo/) A very nice tutorial of YOLO v3
* [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3) PyTorch implmentation of YOLO v3, with only evaluation part
* [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) PyTorch implmentation of YOLO v3, with both training and evaluation parts

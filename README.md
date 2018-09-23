# yolov3.pytorch

This repository contains code for a object detector based on [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf), implemented in PyTorch. This repository is based on [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3). I re-implemented it in PyTorch for better readability and re-useablity.

## 🤔 Requirements

* Python 3.6
* PyTorch **0.4** (v0.4.1 or v0.3 is not supported)
* PIL 5.2.0
* Numpy 1.15.1
* CUDA (**CPU is not supported for both training and test!**)

## 🏹 Train

To be implemented

## 🎯 Evaluation

1. Download pretrained YOLO v3 weights [here](https://pjreddie.com/media/files/yolov3.weights) and move it to `static/yolov3.weights`
2. Run `python test.py --save`
3. You can sepcify detection images directory and detection results directory in `config.py`

Evaluation example

![](https://raw.githubusercontent.com/ECer23/yolov3.pytorch/master/assets/dets/person.jpg)

## ✅ TODO

- [x] Evaluation on image
- [ ] Training on user specified datasets
- [ ] Metrics on evaluation result

## 🔍 Reference

* [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3) Another PyTorch implmentation of YOLO v3, with only evaluation part
* [Series: YOLO object detector in PyTorch](https://blog.paperspace.com/tag/series-yolo/) A very nice tutorial of YOLO v3

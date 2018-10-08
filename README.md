# yolov3.pytorch

* 2018/10/08 **UPDATE** loss function seems to work, furthur test is needed. Details about training could be seen in issue [#1](https://github.com/ECer23/yolov3.pytorch/issues/1)
* 2018/10/07 **WARNING** :warning: mAP computation seems not very accruate
* 2018/10/07 Roll back and fix evaluation error. Loss function is still not working, while evaluation is working.

This repository is used for object detection. The algorithm is based on [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf), implemented in PyTorch v0.4. **Thanks to  [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3) and [ultralytics/yolov3](https://github.com/ultralytics/yolov3)**, based on their work, I re-implemented YOLO v3 in PyTorch for better readability and re-useablity.

## Environments

* Python 3.6
* PyTorch **0.4.0** (0.4.1 or 0.3 is not supported)
* CUDA (**CPU is not supported**)

## Train

### How to train on COCO

1. Download [COCO detection](http://cocodataset.org/#download) dataset and annotions, or prepare your own dataset follow the instructions in [How to train on custom dataset](https://github.com/ECer23/yolov3.pytorch#train-on-custom-dataset)
2. Download official pre-trained Darknet53 weights on ImageNet [here](https://pjreddie.com/media/files/darknet53.conv.74)
3. Transform the weights to PyTorch readable file `0.ckpt` by running
    ```bash
    $ python model.py
    ```
3. Provide information of dataset in `config.py` like
    ```python
    'coco': {
        'num_classes': 80,
        'train_imgs': '/media/data_2/COCO/2017/train2017',
        'train_anno': '/media/data_2/COCO/2017/annotations/instances_train2017.json',
        'class_names': ['person', ...]
    }
    ````
4. Train the model by running
    ```bash
    $ python train.py
    ```
5. (optional) Visualize the training process by running `tensorboard --logdir logs`

### How to train on custom dataset

1. Implement your own dataset in `dataset.py` by writing an inherited class from `torch.utils.data.dataset.Dataset`. The core function is `__getitem__`, which return data for `DataLoader`. Details can be viewed in `dataset.py`. Here is an example

    ```python
    # user defined dataset example

    class TestDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, imgs_dir, transform):
        # init the datset

    def __getitem__(self, index):
        # return (img_tenosr, labels)

    def __len__(self):
        # return length of dataset
    ```

2. To be continued ...


### Training visualization

![](https://raw.githubusercontent.com/ECer23/yolov3.pytorch/master/assets/tensorboard.png)

## Evaluation

### How to evaluate on COCO

1. Download official pretrained YOLO v3 weights [here](https://pjreddie.com/media/files/yolov3.weights)
2. Transform it by running `python src/model.py` to transform official pre-trained YOLOv3 on COCO `checkpoints/darknet/yolov3-coco.weights` to pytorch readable checkpoint file `checkpoints/coco/-1.ckpt`
3. Evaluate on validation sets you specify in `config.py` and compute the mAP by running
    ```bash
    $ python src/evaluate.py
    ```
4. (optional) Save evaluation results by adding `--save`. Results will be saved in `assets/results`

### How to detect COCO objects

1. Download official pretrained YOLO v3 weights [here](https://pjreddie.com/media/files/yolov3.weights)
2. Transform it by running `python src/model.py` to transform official pre-trained YOLOv3 on COCO `checkpoints/darknet/yolov3-coco.weights` to pytorch readable checkpoint file `checkpoints/coco/-1.ckpt`
3. Specify the images folder in `config.py`
    ```python
    demo = {
      'images_dir': opj(ROOT, 'assets/imgs'),
      'result_dir': opj(ROOT, 'assets/dets')
    }
    ```
4. Detect your own images by running
    ```bash
    $ python demo.py
    ```

### Evaluation results

:warning: mAP computation seems not very accruate

| Dataset name | Implementation | Resolution | Notes | mAP | FPS |
|---|---|---|---|---|---|
| COCO 2017 | mine | 416 | official pretrained YOLO v3 weights | 63.4 | |
| COCO 2017 | official | 608 | paper results | 57.9 | |

### Evaluation demo

![](https://github.com/ECer23/yolov3.pytorch/raw/master/assets/dets/person.jpg)

## TODO

- [ ] Evaluation
  - [x] ~~Draw right bounding box~~
  - [ ] mAP re-implementated
  - [ ] FPS recording
- [ ] Training
  - [x] ~~Loss function implementation~~
  - [x] ~~Visualize training process~~
  - [x] ~~Use pre trained Darknet model to train on custom datasets~~
  - [x] ~~Validation~~
  - [ ] Train COCO from scratch
  - [ ] Train custom datasets from scratch
  - [ ] Learning rate scheduler
  - [ ] Data augumentation
- [ ] General
  - [ ] Generalize annotation format to VOC for every dataset
  - [ ] CPU support
  - [x] ~~Memory use imporvements~~


## Reference

* [Series: YOLO object detector in PyTorch](https://blog.paperspace.com/tag/series-yolo/) A very nice tutorial of YOLO v3
* [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3) PyTorch implmentation of YOLO v3, with only evaluation part
* [ultralytics/yolov3](https://github.com/ultralytics/yolov3) PyTorch implmentation of YOLO v3, with both training and evaluation parts
* [utkuozbulak/pytorch-custom-dataset-examples](https://github.com/utkuozbulak/pytorch-custom-dataset-examples) Example of PyTorch custom dataset

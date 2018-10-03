# yolov3.pytorch

* 2018/10/03 **UPDATE** Loss will converge after 1 epoch but the model couldn't generate any bounding boxes. Details could be seen in issue [#1](https://github.com/ECer23/yolov3.pytorch/issues/1)
* 2018/10/02 Could load pre trained darknet-53 to train from scartch
* 2018/09/30 mAP evaluation implemented

This repository contains code for a object detector based on [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf), implemented in PyTorch. This repository is based on [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3). I re-implemented it in PyTorch for better readability and re-useablity.

## Requirements

* Python 3.6
* PyTorch **0.4** (v0.4.1 or v0.3 is not supported)
* Pillow
* Numpy
* CUDA (**CPU is not supported**)

## Train

### How to train the model

1. Download [COCO detection](http://cocodataset.org/#download) dataset and annotions, or prepare your own dataset follow the instructions in [Train on custom dataset](https://github.com/ECer23/yolov3.pytorch#train-on-custom-dataset)
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
    ```
4. Train the model by running
    ```bash
    $ python train.py
    ```
5. (optional) Visualize the training process by running `tensorboard --logdir ../log`

### Train on custom dataset

In `dataset.py`, you can implement your own dataset by writing an inherited class from `torch.utils.data.dataset.Dataset`. The core function is `__getitem__`, which return data for `DataLoader`. Details can be viewed in `dataset.py`

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

If you want to train on COCO, just use `CocoDataset` class. Please install [COCO Python API](https://github.com/cocodataset/cocoapi), which is requried by PyTorch's COCO dataset loader.

I've implemented `prepare_train_dataset` in `dataset.py` to prepare COCO dataloader. You can re-implement this function to adapt to your custom dataset.

### Training visualization

![](https://raw.githubusercontent.com/ECer23/yolov3.pytorch/master/assets/demo.png)

## Evaluation

### How to evaluate on COCO

1. Download official pretrained YOLO v3 weights [here](https://pjreddie.com/media/files/yolov3.weights)
2. Transform it by running `python src/model.py` to transform official pre-trained YOLOv3 on COCO `checkpoints/darknet/yolov3-coco.weights` to pytorch readable checkpoint file `checkpoints/coco/-1.ckpt`
3. Evaluate on validation sets you specify in `config.py` and compute the mAP by running

    ```bash
    $ python src/evaluate.py
    ```
    
    Validation results will be saved in `assets/results`
4. (optional) You can also detect your own images by running
    
    ```bash
    $ python demo.py
    ```

    Before that you should specify the images folder in `config.py`
    ```python
    demo = {
      'images_dir': opj(ROOT, 'assets/imgs'),
      'result_dir': opj(ROOT, 'assets/dets')
    }
    ```

### Evaluation results

| Dataset name | mAP |
|---|---|
| COCO 2017 (official pre-trained weights) | 63.358% |

### Evaluation demo

![](https://github.com/ECer23/yolov3.pytorch/raw/master/assets/dets/person.jpg)

## TODO

### Important

- [x] ~~Evaluation on image~~
- [ ] Training on user custom datasets
  - [x] ~~Loss function implementation~~
  - [x] ~~Visualize training process~~
  - [x] ~~Use pre trained Darknet model to train on custom datasets~~
  - [x] ~~Validation~~
  - [ ] Kmeans clustering?
  - [ ] Evaluate on test-dev
  - [ ] Train COCO and custom datasets from scratch

### Not important

- [ ] Data augumentation ?
- [ ] CPU support
- [x] ~~Memory use imporvements~~

## Reference

* [Series: YOLO object detector in PyTorch](https://blog.paperspace.com/tag/series-yolo/) A very nice tutorial of YOLO v3
* [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3) PyTorch implmentation of YOLO v3, with only evaluation part
* [ultralytics/yolov3](https://github.com/ultralytics/yolov3) PyTorch implmentation of YOLO v3, with both training and evaluation parts
* [utkuozbulak/pytorch-custom-dataset-examples](https://github.com/utkuozbulak/pytorch-custom-dataset-examples) Example of PyTorch custom dataset

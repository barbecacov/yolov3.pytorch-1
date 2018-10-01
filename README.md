# yolov3.pytorch

* 2018/10/01 **❗ATTENTION** Loss function is not working
* 2018/09/30 **UPDATE** mAP is implemented
* ~~2018/09/28 **UPDATE** Loss function is implmented~~

This repository contains code for a object detector based on [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf), implemented in PyTorch. This repository is based on [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3). I re-implemented it in PyTorch for better readability and re-useablity.

## File Tree

```bash
.
├── assets/           # folder of assets
│   ├── dets/           # evaluation results folder
│   ├── imgs/           # evaluation images folder
│   └── demo.gif        # training demo
├── checkpoints/      # folder of checkpoints
│   └── coco/           # sub folder for each dataset
├── lib/              # folder of static file, like .cfg and .weights
├── log/              # folder of logging files
│   └── dd.hh.mm/       # sub folder for each run, named with time
├── scripts/          # folder of scripts
│   ├── download.sh     # download .weights files
│   └── evaluate.sh     # run evaluation`
└── src/              # folder of source file
    ├── config.py        # configuration file
    ├── dataset.py       # dataset
    ├── layers.py        # supported layers for YOLO v3 model
    ├── model.py         # YOLO v3 model
    ├── test.py          # evaluation code
    ├── train.py         # training code
    └── utils.py         # utils function
```

## Requirements

* Python 3.6
* PyTorch **0.4** (v0.4.1 or v0.3 is not supported)
* Pillow
* Numpy
* CUDA (**CPU is not supported for both training and test!**)

## Train

### How to train the model

1. Download [COCO detection](http://cocodataset.org/#download) dataset and annotions, or prepare your own dataset follow the instructions in [Train on custom dataset](https://github.com/ECer23/yolov3.pytorch#train-on-custom-dataset)
2. Provide information of dataset in `config.py`
3. Run `python train.py`
4. (optional) Visualize the training process by running `tensorboard --logdir ../log`

### Input Arguments

* `--reso`, image resolution. Image will be resize to `(reso, reso)` during training. The higher resolution is, the more accurate the detection
* `--lr`, learning rate
* `--batch`, batch size
* `--dataset`, training dataset name
* `--epoch`, start epoch of training

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

### Tensorboard demo

![](https://raw.githubusercontent.com/ECer23/yolov3.pytorch/master/assets/demo.gif)

## Evaluation

### How to evaluate

1. Download official pretrained YOLO v3 checkpoint [here](https://pjreddie.com/media/files/yolov3.weights)
2. Transform it by running `python checkpoints/coco/transform.py` to transform it to pytorch readable checkpoint file `-1.ckpt`
3. Run `python src/evaluate.py`, it will evaluate the validation sets you specify in `config.py` and compute the mAP. Validation results will be saved in `assets/results`
4. (optional) You can also detect your own images by running `python demo.py`. Just specify the images folder in `config.py`
    ```python
    demo = {
      'images_dir': opj(ROOT, 'assets/imgs'),
      'result_dir': opj(ROOT, 'assets/dets')
    }
    ```

### Input arguments

* `--reso`, image resolution. Image will be resize to `(reso, reso)` during validation. The higher resolution is, the more accurate the detection
* `--batch`, batch size
* `--dataset`, validation dataset name
* `--epoch`, checkpoint epoch, -1 for official pretrained YOLO v3.

### Evaluation results

| Dataset name | mAP |
|---|---|
| COCO 2017 (official pre-trained weights) | 63.358% |

### Evaluation demo

![](https://github.com/ECer23/yolov3.pytorch/raw/master/assets/results/000000397133.jpg)

## TODO

### Important

- [x] ~~Evaluation on image~~
- [ ] Training on user custom datasets
  - [ ] Loss function implementation
  - [x] ~~Visualize training process~~
  - [ ] Use pre trained Darknet model to train on custom datasets
  - [x] ~~Validation~~
  - [ ] Tutorials of training from scratch

### Not important

- [ ] CPU support
- [ ] Memory use imporvements

## Reference

* [Series: YOLO object detector in PyTorch](https://blog.paperspace.com/tag/series-yolo/) A very nice tutorial of YOLO v3
* [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3) PyTorch implmentation of YOLO v3, with only evaluation part
* [ultralytics/yolov3](https://github.com/ultralytics/yolov3) PyTorch implmentation of YOLO v3, with both training and evaluation parts
* [utkuozbulak/pytorch-custom-dataset-examples](https://github.com/utkuozbulak/pytorch-custom-dataset-examples) Example of PyTorch custom dataset

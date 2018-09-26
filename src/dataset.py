import os
import time
import torch
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from xml.etree import ElementTree
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torch.utils.data.dataloader import default_collate
opj = os.path.join

import config


class TestDataset(torch.utils.data.dataset.Dataset):
  """Dataset for evaluataion"""

  def __init__(self, imgs_dir, transform):
    """
    @args
      imgs_dir: (str) test images directory
      transform: (torchvision.transforms)
    """
    self.imgs_dir = imgs_dir
    self.imgs_list = os.listdir(imgs_dir)
    self.transform = transform

  def get_path(self, index):
    """
    Get image path

    @args:
      index: (int)
    """
    img_name = self.imgs_list[index]
    img_path = os.path.join(self.imgs_dir, img_name)
    return img_path

  def __getitem__(self, index):
    """Inherited method"""
    img_name = self.imgs_list[index]
    img_path = os.path.join(self.imgs_dir, img_name)
    img = Image.open(img_path)
    img_tensor = self.transform(img)
    return img_tensor, 0  # TODO: fix label

  def __len__(self):
    return len(self.imgs_list)


class CocoDataset(CocoDetection):
  def __getitem__(self, index):
    coco = self.coco
    img_id = self.ids[index]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    target = coco.loadAnns(ann_ids)
    assert len(target) < 50, "# bboxes exceed 50"
    target_tensor = torch.zeros(50, 5)
    for i in range(len(target)):
      target_tensor[i, :4] = torch.Tensor(target[i]['bbox'])
      target_tensor[i, 4] = float(target[i]['category_id'])
    path = coco.loadImgs(img_id)[0]['file_name']
    img = Image.open(os.path.join(self.root, path)).convert('RGB')
    w, h = img.size
    target_tensor[i, 0] /= w
    target_tensor[i, 2] /= w
    target_tensor[i, 1] /= h
    target_tensor[i, 3] /= h
    if self.transform is not None:
        img = self.transform(img)
    return path, img, target_tensor


class SixdDataset(torch.utils.data.dataset.Dataset):
  """Image dataset for SIXD"""

  def __init__(self, root, listname, transform):
    """Init class
    @args
      root: (str) path to dataset
      listname: (str) image list filename
      transform: (torchvision.transforms)
    @params
      self.img_names: (list) list of image filename
      self.annos: (list) list of image annotations, each with size [#bbox, 5]
    """
    self.transform = transform
    self.root = root
    self.img_dir = opj(root, 'JPEGImages')
    self.anno_dir = opj(root, 'Annotations')
    self.filelist = opj(root, 'ImageSets/Main', listname)
    libname = opj(config.ROOT_DIR, 'lib', root.split('/')[-1] + '.pkl')

    if os.path.exists(libname):
      with open(libname, 'rb') as f:
        obj = pickle.load(f)
      self.img_names = obj['img_names']
      self.annos = obj['annos']
      print("successfully load annotations from disk")
    else:
      start_time = time.time()
      print("retriving image names")

      self.img_names = []
      with open(self.filelist) as f:
        data = f.readlines()
        for line in data:
          self.img_names.append(line.split(' ')[0])

      print("parsing annotations")
      self.annos = []
      for img_name in tqdm(self.img_names, ncols=80):
        img_idx = img_name.split('_')[0]
        anno_path = opj(self.anno_dir, img_idx + '.xml')
        root = ElementTree.parse(anno_path).getroot()
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        objects = root.findall('object')
        img_anno = np.ndarray((len(objects), 5))
        
        for i, o in enumerate(objects):
          bndbox = o.find('bndbox')
          
          for j, child in enumerate(bndbox):         # bbox
            img_anno[i, j] = int(child.text)
          img_anno[i, :2] /= width  # scale to (0,1)
          img_anno[i, 2:] /= height  # scale to (0,1)
          
          # x1, x2, y1, y2  => x1, y1, x2, y2
          temp = img_anno[i, 1]
          img_anno[i, 1] = img_anno[i, 2]
          img_anno[i, 2] = temp

          img_anno[i, 4] = int(o.find('name').text)  # label

        self.annos.append(img_anno)

      with open(opj(config.ROOT_DIR, 'lib', libname), 'wb') as f:
        pickle.dump({
            'img_names': self.img_names,
            'annos': self.annos
        }, f)

      print("done (t = %.2f)" % (time.time() - start_time))
      print("save annotations to disk")

  def __getitem__(self, index):
    """Return dataset item
    @args
      index: (int) item index
    @returns
      img_tensor: (torch.Tensor) Tensor with size [C, H, W]
      img_anno: (torch.Tensor) corresponding annotation with size [15, 5]
      img_name: (str) image name
    """
    img_name = self.img_names[index]
    img = Image.open(opj(self.img_dir, img_name))
    img_tensor = self.transform(img)
    num_bbox = self.annos[index].shape[0]
    assert num_bbox < 15, "# bboxes exceed 15!"
    img_anno = torch.zeros(15, 5)  # fixed size padding
    img_anno[:num_bbox, :] = torch.Tensor(self.annos[index])
    return img_name, img_tensor, img_anno

  def __len__(self):
    return len(self.img_names)


def prepare_test_dataset(path, reso, batch_size=1):
  """Prepare dataset for evaluation
  @args
    path: (str) path to images
    reso: (int) evaluation image resolution
    batch_size: (int) default 1
  @returns
    img_datasets: (torchvision.datasets) test image datasets
    dataloader: (DataLoader)
  """
  transform = transforms.Compose([
      transforms.Resize(size=(reso, reso), interpolation=3),
      transforms.ToTensor()
  ])

  img_datasets = TestDataset(path, transform)
  dataloader = torch.utils.data.DataLoader(img_datasets, batch_size=batch_size, num_workers=4)

  return img_datasets, dataloader


def prepare_train_dataset(name, reso, batch_size=32):
  """
  Prepare dataset for training/validation

  @args
    name: (str) dataset name [tejani, hinter]
    reso: (int) training/validation image resolution
    batch_size: (int) default 1

  @returns
    trainloader, valloader: (Dataloader) dataloader for training and validation
  """
  transform = transforms.Compose([
      transforms.Resize(size=(reso, reso), interpolation=3),
      transforms.ToTensor()
  ])

  train_root = config.datasets[name]['train_root']

  if name == 'coco':
    img_datasets = CocoDataset(
      root=opj(train_root, 'train2017'),
      annFile=opj(train_root, 'annotations/instances_train2017.json'),
      transform=transform
    )
  else:
    img_datasets = SixdDataset(train_root, 'train.txt', transform=transform)

  dataloder = torch.utils.data.DataLoader(img_datasets, batch_size=batch_size, shuffle=True)

  return img_datasets, dataloder

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


class DemoDataset(torch.utils.data.dataset.Dataset):
  """Dataset for evaluataion"""

  def __init__(self, imgs_dir, transform):
    """
    @Args
      imgs_dir: (str) test images directory
      transform: (torchvision.transforms)
    """
    self.imgs_dir = imgs_dir
    self.imgs_list = os.listdir(imgs_dir)
    self.transform = transform

  def __getitem__(self, index):
    """Inherited method"""
    img_name = self.imgs_list[index]
    img_path = os.path.join(self.imgs_dir, img_name)
    img = Image.open(img_path)
    img_tensor = self.transform(img)
    return img_path, img_tensor

  def __len__(self):
    return len(self.imgs_list)


class CocoDataset(CocoDetection):
  def __getitem__(self, index):
    """
    @Returns
      path: (str) image file name
      img: (Tensor) with size [C, H, W]
      TODO: memory improvements ?
      target_tensor: (list of Tensor) each list item with size [xc, yc, w, h, label]
    """
    coco = self.coco
    img_id = self.ids[index]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    path = coco.loadImgs(img_id)[0]['file_name']
    img = Image.open(os.path.join(self.root, path)).convert('RGB')
    w, h = img.size
    target = coco.loadAnns(ann_ids)
    annos = torch.zeros(len(target), 5)
    for i in range(len(target)):
      bbox = torch.Tensor(target[i]['bbox'])  # [x1, y1, w, h]
      label = config.datasets['coco']['category_id_mapping'][int(target[i]['category_id'])]
      annos[i, 0] = (bbox[0] + bbox[2] / 2) / w  # xc
      annos[i, 1] = (bbox[1] + bbox[3] / 2) / h  # yc
      annos[i, 2] = bbox[2] / w  # w
      annos[i, 3] = bbox[3] / h  # h
      annos[i, 4] = label  # 0-80
    if self.transform is not None:
      img = self.transform(img)
    return path, img, annos

  @staticmethod
  def collate_fn(batch):
    """Collate function for Coco DataLoader

    @Returns
      names: (tuple) each is a str of image filename
      images: (Tensor) with size [bs, C, H, W]
      annos: (tuple) each is a Tensor of annotations
    """
    names, images, annos = zip(*batch)
    images = default_collate(images)
    return names, images, annos


class SixdDataset(torch.utils.data.dataset.Dataset):
  """Image dataset for SIXD"""

  def __init__(self, root, listname, transform):
    """Init class
    @Args
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
    libname = opj(config.ROOT, 'lib', root.split('/')[-1] + '.pkl')

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
          img_anno[i, 4] = int(o.find('name').text)  # label

        # x1, x2, y1, y2  => xc, yc, w, h
        xc = (img_anno[:, 0] + img_anno[:, 1]) / 2
        yc = (img_anno[:, 2] + img_anno[:, 3]) / 2
        w = img_anno[:, 1] - img_anno[:, 0]
        h = img_anno[:, 3] - img_anno[:, 2]
        img_anno[:, 0] = xc
        img_anno[:, 1] = yc
        img_anno[:, 2] = w
        img_anno[:, 3] = h

        self.annos.append(img_anno)

      with open(opj(config.ROOT, 'lib', libname), 'wb') as f:
        pickle.dump({
            'img_names': self.img_names,
            'annos': self.annos
        }, f)

      print("done (t = %.2f)" % (time.time() - start_time))
      print("save annotations to disk")

  def __getitem__(self, index):
    """    
    @Args

    index: (int) item index

    @Returns

    img_tensor: (Tensor) Tensor with size [C, H, W]
    img_anno: (Tensor) corresponding annotation with size [15, 5]
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


def prepare_demo_dataset(path, reso, batch_size=1):
  """Prepare dataset for demo

  @Args
    path: (str) path to images
    reso: (int) evaluation image resolution
    batch_size: (int) default 1

  @Returns
    img_datasets: (torchvision.datasets) demo image datasets
    dataloader: (DataLoader)
  """
  transform = transforms.Compose([
      transforms.Resize(size=(reso, reso), interpolation=3),
      transforms.ToTensor()
  ])

  img_datasets = DemoDataset(path, transform)
  dataloader = torch.utils.data.DataLoader(img_datasets, batch_size=batch_size, num_workers=8)

  return img_datasets, dataloader


def prepare_train_dataset(name, reso, batch_size=32):
  """Prepare dataset for training

  @Args  
    name: (str) dataset name [coco]
    reso: (int) training image resolution
    batch_size: (int) default 32

  @Returns
    img_datasets: (CocoDataset) image datasets
    trainloader: (Dataloader) dataloader for training
  """
  transform = transforms.Compose([
      transforms.Resize(size=(reso, reso), interpolation=3),
      transforms.ToTensor()
  ])

  if name == 'coco':
    path = config.datasets[name]
    img_datasets = CocoDataset(root=path['train_imgs'], annFile=path['train_anno'], transform=transform)
    dataloder = torch.utils.data.DataLoader(img_datasets, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=CocoDataset.collate_fn)
  elif name == 'tejani':
    path = config.datasets[name]
    img_datasets = SixdDataset(root=path['train_root'], listname='train.txt', transform=transform)

  return img_datasets, dataloder


def prepare_val_dataset(name, reso, batch_size=32):
  """Prepare dataset for validation

  @Args  
    name: (str) dataset name [tejani, hinter]
    reso: (int) validation image resolution
    batch_size: (int) default 32

  @Returns
    img_datasets: (CocoDataset)
    dataloader: (Dataloader)
  """
  transform = transforms.Compose([
      transforms.Resize(size=(reso, reso), interpolation=3),
      transforms.ToTensor()
  ])

  if name == 'coco':
    path = config.datasets[name]
    img_datasets = CocoDataset(root=path['val_imgs'], annFile=path['val_anno'], transform=transform)

  dataloder = torch.utils.data.DataLoader(img_datasets, batch_size=batch_size, num_workers=4, collate_fn=CocoDataset.collate_fn, shuffle=True)

  return img_datasets, dataloder

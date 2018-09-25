import os
import time
import torch
import pickle
import xml.etree.ElementTree
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
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


class SixdDataset(torch.utils.data.dataset.Dataset):
  """Image dataset for SIXD"""

  def __init__(self, root, listname, transform):
    """Init class
    @args
      root: (str) path to dataset
      listname: (str) image list filename
      transform: (torchvision.transforms)
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
      print("Successfully load annotations from disk")
    else:
      start_time = time.time()

      print("Get image names")
      self.img_names = []
      with open(self.filelist) as f:
        data = f.readlines()
        for line in data:
          self.img_names.append(line.split(' ')[0])

      print("Parsing annotations")
      self.annos = []
      for img_name in tqdm(self.img_names, ncols=80):
        img_idx = img_name.split('_')[0]
        anno_path = opj(self.anno_dir, img_idx + '.xml')
        root = xml.etree.ElementTree.parse(anno_path).getroot()
        objects = root.findall('object')
        img_anno = []
        for o in objects:
          label = {}
          label['name'] = int(o.find('name').text)
          label['bbox'] = []
          bndbox = o.find('bndbox')
          for child in bndbox:
            label['bbox'].append(int(child.text))
          img_anno.append(label)
        self.annos.append(img_anno)

      with open(opj(config.ROOT_DIR, 'lib', libname), 'wb') as f:
        pickle.dump({
            'img_names': self.img_names,
            'annos': self.annos
        }, f)

      print("Done (t = %.2f)" % (time.time() - start_time))
      print("Save annotations to disk")

  def __getitem__(self, index):
    """Return dataset item
    @args
      index: (int) item index
    @returns
      img_tensor: (torch.Tensor) Tensor with size [C, H, W]
      img_anno: (list) corresponding annotation
      img_name: (str) image name
    """
    img_name = self.img_names[index]
    img = Image.open(opj(self.img_dir, img_name))
    img_tensor = self.transform(img)
    img_anno = self.annos[index]
    return img_name, img_tensor, BboxList(img_anno)

  def __len__(self):
    return len(self.img_names)

  @staticmethod
  def collate_fn(batch):
    """Collate function for SixdDataset, used in DataLoader
    @args
      batch: (list) list of item from self.__getitem__()
    @returns
      target_batch: (tuple) (names, tensor, bboxes)
        names: (list) image names
        tensor: (torch.Tensor) with size [batch_size, C, H, W]
        bboxes: (list) length = batch_size, each represents one bbox
    """
    target_batch = default_collate(batch)
    bboxes = []
    for item in batch:
      bbox = item[2]
      bboxes.append(bbox)
    target_batch[2] = bboxes
    return target_batch


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

  img_datasets = SixdDataset(train_root, 'train.txt', transform=transform)
  dataloder = torch.utils.data.DataLoader(img_datasets, batch_size=batch_size, shuffle=True, collate_fn=SixdDataset.collate_fn)

  return img_datasets, dataloder

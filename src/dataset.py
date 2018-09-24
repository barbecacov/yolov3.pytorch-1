import os
import torch
import config
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CocoDetection
opj = os.path.join


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


def prepare_test_dataset(path, reso, batch_size=1):
  """
  Prepare dataset for evaluation

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


def prepare_trainval_dataset(root_dir, reso, batch_size=32):
  """
  Prepare dataset for training/validation

  @args
    root_dir: (str) root directory to training datasets
    reso: (int) training/validation image resolution
    batch_size: (int) default 1

  @returns
    trainloader, valloader: (Dataloader) dataloader for training and validation
  """
  transform = transforms.Compose([
      transforms.Resize(size=(reso, reso), interpolation=3),
      transforms.ToTensor()
  ])

  train_datasets = CocoDetection(
      root=opj(root_dir, 'train2017'),
      annFile=opj(root_dir, 'annotations/instances_train2017.json'),
      transform=transform
  )
  trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, num_workers=4, shuffle=True)

  val_datasets = CocoDetection(
      root=opj(root_dir, 'val2017'),
      annFile=opj(root_dir, 'annotations/instances_val2017.json'),
      transform=transform
  )
  valloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, num_workers=4, shuffle=False)

  return train_datasets, val_datasets, trainloader, valloader

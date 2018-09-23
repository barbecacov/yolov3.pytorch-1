import os
import torch
from PIL import Image
from torchvision import transforms

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

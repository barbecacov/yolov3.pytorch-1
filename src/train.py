import argparse
from tqdm import tqdm

import config
from model import YOLOv3, nms
from dataset import prepare_trainval_dataset


def parse_arg():
  parser = argparse.ArgumentParser(description='YOLO v3 training')
  parser.add_argument('--reso', default=416, type=int, help="Input image resolution")
  parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate")
  parser.add_argument('--batch_size', default=32, type=int)
  parser.add_argument('--start_epoch', default=0, type=int)
  parser.add_argument('-r', action='store_true', help="Resume from checkpoint")
  return parser.parse_args()


def train(epoch, lr):
  """Training wrapper"""
  pass

def val(batch_idx):
  """Validation wrapper"""
  pass


if __name__ == '__main__':
  print("\n==> Parsing arguments ...\n")
  args = parse_arg()
  cfg = config.network['cfg']
  coco_root = config.datasets['coco']['root_dir']
  resolution = args.reso
  learning_rate = args.lr
  batch_size = args.batch_size
  start_epoch = args.start_epoch
  for arg in vars(args):
    print(arg, '=', getattr(args, arg))

  print("\n==> Prepare Data ...\n")
  train_datasets, val_datasets, trainloader, valloader = prepare_trainval_dataset(coco_root, resolution, batch_size)
  print("# Training images:", len(train_datasets))
  print("# Validation images:", len(val_datasets))

  print("\n==> Loading network ...\n")
  yolo = YOLOv3(cfg, resolution).cuda()

  print("\n==> Training ...\n")
  yolo.train()
  for i in range(start_epoch, start_epoch + 100):
    pass

import torch
import argparse
import warnings
import torch.optim as optim
from tqdm import tqdm
warnings.filterwarnings("ignore")

import config
from model import YOLOv3
from dataset import prepare_train_dataset


def parse_arg():
  parser = argparse.ArgumentParser(description='YOLO v3 training')
  parser.add_argument('--reso', default=416, type=int, help="Input image resolution")
  parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate")
  parser.add_argument('--batch_size', default=4, type=int)
  parser.add_argument('--start_epoch', default=0, type=int)
  parser.add_argument('--dataset', default='tejani', choices=['tejani'], type=str, help="Dataset name")
  parser.add_argument('-r', action='store_true', help="Resume from checkpoint")
  return parser.parse_args()


def train(epoch, trainloader, yolo, lr):
  """
  Training wrapper

  @args
    epoch: (int) training epoch
    trainloader: (Dataloader) train data loader
    yolo: (nn.Module) YOLOv3 model
    lr: (float) learning rate
  """
  optimizer = optim.SGD(yolo.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
  for batch_idx, (names, inputs, targets) in enumerate(tqdm(trainloader, ncols=80)):
    inputs, targets = inputs.cuda(), targets.cuda()
    detections = yolo(inputs)
    yolo.loss(targets)


def val(batch_idx):
  """Validation wrapper"""
  pass


if __name__ == '__main__':
  print("\n==> Parsing arguments ...\n")
  args = parse_arg()
  cfg = config.network[args.dataset]['cfg']
  for arg in vars(args):
    print(arg, '=', getattr(args, arg))

  print("\n==> Prepare Data ...\n")
  img_datasets, dataloader = prepare_train_dataset(args.dataset, args.reso, args.batch_size)
  print("# Training images:", len(img_datasets))

  print("\n==> Loading network ...\n")
  yolo = YOLOv3(cfg, args.reso).cuda()

  print("\n==> Training ...\n")
  yolo.train()
  train(0, dataloader, yolo, args.lr)

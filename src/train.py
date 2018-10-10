import os
import time
import torch
import argparse
import warnings
import numpy as np
from PIL import Image
import torch.optim as optim
from termcolor import colored
from pyemojify import emojify
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from torchvision import transforms, utils
import torch.optim.lr_scheduler as lr_scheduler
opj = os.path.join
warnings.filterwarnings("ignore")

import config
from model import YOLOv3
from dataset import prepare_train_dataset, prepare_val_dataset
from utils import get_current_time, draw_detection, save_checkpoint, load_checkpoint, mAP, log


def parse_arg():
  parser = argparse.ArgumentParser(description='YOLO v3 training')
  parser.add_argument('--reso', default=416, type=int, help="Input image resolution")
  parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate")
  parser.add_argument('--batch', default=16, type=int, help="Batch size")
  parser.add_argument('--dataset', default='voc', choices=['voc', 'coco'], type=str, help="Dataset name")
  parser.add_argument('--checkpoint', default='0.0', type=str, help="Checkpoint name in format: `epoch.iteration`")
  parser.add_argument('--gpu', default='0', type=str, help="GPU id")
  return parser.parse_args()


args = parse_arg()
cfg = config.network[args.dataset]['cfg']
log_dir = opj(config.LOG_ROOT, get_current_time())
writer = SummaryWriter(log_dir=log_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def train(epoch, trainloader, yolo, optimizer):
  """Training wrapper

  @Args
    epoch: (int) training epoch
    trainloader: (Dataloader) train data loader 
    yolo: (nn.Module) YOLOv3 model
    optimizer: (optim) optimizer
  """
  yolo.train()
  tbar = tqdm(trainloader, ncols=80)
  tbar.set_description('training')
  for batch_idx, (paths, inputs, targets) in enumerate(tbar):
    global_step = batch_idx + epoch * len(trainloader)
    
    # learning rate warm up
    if (epoch == 0) & (batch_idx <= 1000):
      lr = optimizer.param_groups[0]['lr'] * (batch_idx / 1000) ** 4
      for g in optimizer.param_groups:
        g['lr'] = lr

    optimizer.zero_grad()
    inputs = inputs.cuda()
    yolo(inputs, targets)
    log(writer, 'train_loss', yolo.loss, global_step)
    yolo.loss['total'].backward()
    optimizer.step()

    # save something every 1000 iterations
    if (global_step + 1) % 1000 == 0:
      save_checkpoint(opj(config.CKPT_ROOT, args.dataset), epoch, batch_idx + 1, {
          'epoch': epoch,
          'iteration': batch_idx + 1,
          'state_dict': yolo.state_dict()
      })


if __name__ == '__main__':
  # 1. Parsing arguments
  print(colored("\n==>", 'blue'), emojify("Parsing arguments :zap:\n"))
  assert args.reso % 32 == 0, emojify("Resolution must be interger times of 32 :shit:")
  for arg in vars(args):
    print(arg, ':', getattr(args, arg))
  print("log_dir :", log_dir)

  # 2. Loading network
  # TODO: resume tensorboard
  print(colored("\n==>", 'blue'), emojify("Loading network :hourglass:\n"))
  yolo = YOLOv3(cfg, args.reso).cuda()
  start_epoch, start_iteration = args.checkpoint.split('.')
  start_epoch, start_iteration, state_dict = load_checkpoint(
      opj(config.CKPT_ROOT, args.dataset),
      int(start_epoch),
      int(start_iteration)
  )
  yolo.load_state_dict(state_dict)
  print("Model starts training from epoch %d iteration %d" % (start_epoch, start_iteration))

  # 3. Preparing data
  print(colored("\n==>", 'blue'), emojify("Preparing data :coffee:\n"))
  train_img_datasets, train_dataloader = prepare_train_dataset(args.dataset, args.reso, args.batch)
  print("Number of training images:", len(train_img_datasets))

  # 4. Training
  print(colored("\n==>", 'blue'), emojify("Training :snowflake:\n"))
  optimizer = optim.SGD(filter(lambda p: p.requires_grad, yolo.parameters()),
                        lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
  for epoch in range(start_epoch, start_epoch+20):
    print("[EPOCH] %d, learning rate = %.5f" % (epoch, optimizer.param_groups[0]['lr']))
    train(epoch, train_dataloader, yolo, optimizer)

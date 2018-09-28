import os
import torch
import argparse
import warnings
import torch.optim as optim
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from torchvision import transforms, utils
opj = os.path.join
warnings.filterwarnings("ignore")

import config
from model import YOLOv3
from dataset import prepare_train_dataset
from utils import get_current_time, draw_detection


def parse_arg():
  parser = argparse.ArgumentParser(description='YOLO v3 training')
  parser.add_argument('--reso', default=416, type=int, help="Input image resolution")
  parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate")
  parser.add_argument('--batch_size', default=16, type=int)
  parser.add_argument('--start_epoch', default=0, type=int)
  parser.add_argument('--dataset', default='coco', choices=['tejani', 'coco'], type=str, help="Dataset name")
  parser.add_argument('-r', action='store_true', help="Resume from checkpoint")
  return parser.parse_args()


args = parse_arg()
cfg = config.network[args.dataset]['cfg']
weights = config.network[args.dataset]['weights']
log_dir = opj(config.LOG_ROOT, get_current_time())
writer = SummaryWriter(log_dir=log_dir)


def train(epoch, trainloader, yolo, lr):
  """Training wrapper

  Parameters
    epoch: (int) training epoch
    trainloader: (Dataloader) train data loader 
    yolo: (nn.Module) YOLOv3 model
    lr: (float) learning rate
  """
  optimizer = optim.SGD(yolo.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
  tbar = tqdm(trainloader, ncols=100)
  for batch_idx, (names, inputs, targets) in enumerate(tbar):
    optimizer.zero_grad()

    global_step = batch_idx + epoch*len(trainloader)
    inputs, targets = inputs.cuda(), targets.cuda()
    detections = yolo(inputs)
    
    loss, correct_num, total_num = yolo.loss(targets)

    writer.add_scalar('loss/total', loss['total'], global_step)
    writer.add_scalar('loss/coord', loss['coord'], global_step)
    writer.add_scalar('loss/conf', loss['conf'], global_step)
    writer.add_scalar('loss/cls', loss['cls'], global_step)

    loss['total'].backward()
    optimizer.step()

    if batch_idx % 100 == 0:
      if detections.size(0) == 0:
        continue
      # TODO: train_root format
      if args.dataset == 'coco':
        img_path = opj(config.datasets[args.dataset]['train_root'], 'train2017', names[0])
      else:
        img_path = opj(config.datasets[args.dataset]['train_root'], 'JPEGImages', names[0])
      detection = detections[detections[:, 0] == 0]
      img = draw_detection(img_path, detection, yolo.get_reso())
      img_tensor = utils.make_grid(transforms.ToTensor()(img))
      writer.add_image('image', img_tensor, global_step)


def val(batch_idx):
  """Validation wrapper"""
  pass


if __name__ == '__main__':
  print("\n==> Parsing arguments ...\n")
  for arg in vars(args):
    print(arg, ':', getattr(args, arg))
  print("log_dir :", log_dir)

  print("\n==> Prepare Data ...\n")
  img_datasets, dataloader = prepare_train_dataset(args.dataset, args.reso, args.batch_size)
  print("# Training images:", len(img_datasets))

  print("\n==> Loading network ...\n")
  yolo = YOLOv3(cfg, args.reso).cuda()
  # yolo.load_weights(weights)

  print("\n==> Training ...\n")
  yolo.train()
  for epoch in range(args.start_epoch, args.start_epoch+100):
    train(epoch, dataloader, yolo, args.lr)
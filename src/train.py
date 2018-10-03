import os
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
opj = os.path.join
warnings.filterwarnings("ignore")

import config
from model import YOLOv3
from dataset import prepare_train_dataset
from utils import get_current_time, draw_detection, save_checkpoint, load_checkpoint, mAP


def parse_arg():
  parser = argparse.ArgumentParser(description='YOLO v3 training')
  parser.add_argument('--reso', default=416, type=int, help="Input image resolution")
  parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate")
  parser.add_argument('--batch', default=16, type=int, help="Batch size")
  parser.add_argument('--dataset', default='coco', choices=['tejani', 'coco'], type=str, help="Dataset name")
  parser.add_argument('--checkpoint', default='0.0', type=str, help="Checkpoint name in format: `epoch.iteration`")
  return parser.parse_args()


args = parse_arg()
cfg = config.network[args.dataset]['cfg']
log_dir = opj(config.LOG_ROOT, get_current_time())
writer = SummaryWriter(log_dir=log_dir)


def train(epoch, trainloader, yolo, lr):
  """Training wrapper

  @Args
    epoch: (int) training epoch
    trainloader: (Dataloader) train data loader 
    yolo: (nn.Module) YOLOv3 model
    lr: (float) learning rate
  """
  train_mAP = None
  optimizer = optim.SGD(yolo.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
  tbar = tqdm(trainloader, ncols=80)
  for batch_idx, (names, inputs, targets) in enumerate(tbar):
    optimizer.zero_grad()

    global_step = batch_idx + epoch*len(trainloader)
    inputs = inputs.cuda()
    detections = yolo(inputs)

    loss, cache = yolo.loss(targets)
    with torch.no_grad():
      mAPs = mAP(detections, targets, yolo.reso)
    mAP_batch = np.mean(mAPs)
    train_mAP = (mAP_batch + train_mAP * batch_idx) / (1 + batch_idx) if train_mAP is not None else mAP_batch
    tbar.set_description('%.3f%%' % (train_mAP * 100))

    writer.add_scalar('loss/x', loss['x'], global_step)
    writer.add_scalar('loss/y', loss['y'], global_step)
    writer.add_scalar('loss/w', loss['w'], global_step)
    writer.add_scalar('loss/h', loss['h'], global_step)
    writer.add_scalar('loss/conf', loss['conf'], global_step)
    writer.add_scalar('loss/cls', loss['cls'], global_step)
    writer.add_scalar('loss/total', loss['total'], global_step)
    writer.add_scalar('metric/mAP', train_mAP, global_step)

    loss['total'].backward()
    optimizer.step()

    # save something every 1000 iterations
    if (batch_idx + 1) % 1000 == 0:
      save_checkpoint(opj(config.CKPT_ROOT, args.dataset), epoch, batch_idx + 1, {
        'epoch': epoch,
        'iteration': batch_idx + 1,
        'state_dict': yolo.state_dict(),
        'mAP': train_mAP,
      })

      img_idx = 0  # idx in a batch

      if args.dataset == 'coco':
        img_path = opj(config.datasets[args.dataset]['train_imgs'], names[img_idx])

      try:
        detection = detections[detections[:, 0] == img_idx]
        if detection.size(0) == 0:
          raise Exception
      except Exception:
        print(emojify("\nDetection disappeared :scream:"))
        pred_img = Image.open(img_path)
      else:
        pred_img = draw_detection(img_path, detection.detach().cpu(), yolo.reso, type='pred')

      gt_img = draw_detection(img_path, targets[img_idx].detach().cpu(), yolo.reso, type='gt')
      gt_img_tensor = utils.make_grid(transforms.ToTensor()(gt_img))
      pred_img_tensor = utils.make_grid(transforms.ToTensor()(pred_img))
      writer.add_image('image/pred', pred_img_tensor, global_step)
      writer.add_image('image/gt', gt_img_tensor, global_step)


if __name__ == '__main__':
  # 1. Parsing arguments
  print(colored("\n==>", 'blue'), emojify("Parsing arguments :hammer:\n"))
  assert args.reso % 32 == 0, emojify("Resolution must be interger times of 32 :shit:")
  for arg in vars(args):
    print(arg, ':', getattr(args, arg))
  print("log_dir :", log_dir)

  # 2. Loading network
  print(colored("\n==>", 'blue'), emojify("Loading network :hourglass:\n"))
  yolo = YOLOv3(cfg, args.reso).cuda()
  start_epoch, start_iteration = args.checkpoint.split('.')
  start_epoch, start_iteration, best_mAP, state_dict = load_checkpoint(
    opj(config.CKPT_ROOT, args.dataset),
    int(start_epoch),
    int(start_iteration)
  )
  yolo.load_state_dict(state_dict)
  print("Model starts training from epoch %d iteration %d, with mAP %.2f%%" % (start_epoch, start_iteration, best_mAP * 100))

  # 3. Preparing data
  print(colored("\n==>", 'blue'), emojify("Preparing data :coffee:\n"))
  img_datasets, dataloader = prepare_train_dataset(args.dataset, args.reso, args.batch)
  print("# Training images:", len(img_datasets))

  # 4. Training
  print(colored("\n==>", 'blue'), emojify("Training :seedling:\n"))
  yolo.train()
  for epoch in range(start_epoch, start_epoch+20):
    print("[EPOCH] %d" % epoch)
    train(epoch, dataloader, yolo, args.lr)

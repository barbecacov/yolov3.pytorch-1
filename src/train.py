import os
import torch
import argparse
import warnings
import torch.optim as optim
from pyemojify import emojify
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from torchvision import transforms, utils
opj = os.path.join
warnings.filterwarnings("ignore")

import config
from model import YOLOv3
from dataset import prepare_train_dataset
from utils import get_current_time, draw_detection, save_checkpoint, load_checkpoint


def parse_arg():
  parser = argparse.ArgumentParser(description='YOLO v3 training')
  parser.add_argument('--reso', default=416, type=int, help="Input image resolution")
  parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate")
  parser.add_argument('--batch', default=20, type=int, help="Batch size")
  parser.add_argument('--dataset', default='coco', choices=['tejani', 'coco'], type=str, help="Dataset name")
  parser.add_argument('--epoch', default=0, type=int, help="Start epoch of training")
  parser.add_argument('--save', action='store_true', help="Save image during training")
  return parser.parse_args()


args = parse_arg()
cfg = config.network[args.dataset]['cfg']
log_dir = opj(config.LOG_ROOT, get_current_time())
writer = SummaryWriter(log_dir=log_dir)


def train(epoch, trainloader, yolo, lr, save_img=True):
  """Training wrapper

  @Args
    epoch: (int) training epoch
    trainloader: (Dataloader) train data loader 
    yolo: (nn.Module) YOLOv3 model
    lr: (float) learning rate
  """
  optimizer = optim.SGD(yolo.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
  tbar = tqdm(trainloader, ncols=80)
  for batch_idx, (names, inputs, targets) in enumerate(tbar):
    optimizer.zero_grad()

    global_step = batch_idx + epoch*len(trainloader)
    inputs, targets = inputs.cuda(), targets.cuda()
    detections = yolo(inputs)

    loss, cache = yolo.loss(targets)

    writer.add_scalar('loss/x', loss['x'], global_step)
    writer.add_scalar('loss/y', loss['y'], global_step)
    writer.add_scalar('loss/w', loss['w'], global_step)
    writer.add_scalar('loss/h', loss['h'], global_step)
    writer.add_scalar('loss/conf', loss['conf'], global_step)
    writer.add_scalar('loss/cls', loss['cls'], global_step)
    writer.add_scalar('loss/total', loss['total'], global_step)

    loss['total'].backward()
    # tbar.set_description('%d/%d' % (cache['nM'], cache['total_num']))
    optimizer.step()

    if save_img == True:
      if detections.size(0) < inputs.size(0) / 2:
        print(emojify("\nDetection disappeared :scream:\n"))
        continue

      if batch_idx % 1 == 0:
        # TODO: train_root format
        if args.dataset == 'coco':
          img_path = opj(config.datasets[args.dataset]['train_root'], 'train2017', names[0])
        else:
          img_path = opj(config.datasets[args.dataset]['train_root'], 'JPEGImages', names[0])

        detection = detections[detections[:, 0] == 0]
        img = draw_detection(img_path, detection, yolo.reso)
        img_tensor = utils.make_grid(transforms.ToTensor()(img))
        writer.add_image('image', img_tensor, global_step)


if __name__ == '__main__':
  print(emojify("\n==> Parsing arguments :hammer:\n"))
  assert args.reso % 32 == 0, emojify("Resolution must be interger times of 32 :shit:")
  for arg in vars(args):
    print(arg, ':', getattr(args, arg))
  print("log_dir :", log_dir)

  print(emojify("\n==> Preparing Data :coffee:\n"))
  img_datasets, dataloader = prepare_train_dataset(args.dataset, args.reso, args.batch)
  print("# Training images:", len(img_datasets))

  print(emojify("\n==> Loading network :hourglass:\n"))
  yolo = YOLOv3(cfg, args.reso).cuda()
  start_epoch = args.epoch
  best_mAP = 0
  if start_epoch != 0:
    start_epoch, best_mAP, state_dict = load_checkpoint(opj(config.CKPT_ROOT, args.dataset), start_epoch)
    yolo.load_state_dict(state_dict)
  print("Model starts training from epoch %d" % start_epoch)

  print(emojify("\n==> Training :seedling:\n"))
  yolo.train()
  for epoch in range(start_epoch, start_epoch+20):
    train(epoch, dataloader, yolo, args.lr, args.save)
    if epoch % 5 == 4:  # save every 4 epochs
      save_dict = {
        'mAP': 0,
        'epoch': epoch,
        'state_dict': yolo.state_dict()
      }
      save_checkpoint(opj(config.CKPT_ROOT, args.dataset), epoch, save_dict)

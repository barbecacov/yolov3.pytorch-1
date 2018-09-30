import os
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from PIL import Image
from pyemojify import emojify
opj = os.path.join
warnings.filterwarnings("ignore")

import config
from model import YOLOv3
from dataset import prepare_val_dataset
from utils import draw_detection, load_checkpoint, mAP


def parse_arg():
  parser = argparse.ArgumentParser(description='YOLO v3 training')
  parser.add_argument('--reso', default=416, type=int, help="Input image resolution")
  parser.add_argument('--batch', default=32, type=int, help="Batch size")
  parser.add_argument('--dataset', default='coco', choices=['tejani', 'coco'], type=str, help="Dataset name")
  parser.add_argument('--epoch', default=-1, type=int, help="Start epoch of validation")
  parser.add_argument('--save', action='store_true', help="Save image during validation")
  return parser.parse_args()


args = parse_arg()
cfg = config.network[args.dataset]['cfg']


def val(valloader, yolo, save_img=True):
  """Validation wrapper

  @Args
    valloader: (Dataloader) validation data loader 
    yolo: (nn.Module) YOLOv3 model
    save_img: (bool) whether to save images during validation
  """
  mAPs = []
  tbar = tqdm(valloader, ncols=80)
  for batch_idx, (names, inputs, targets) in enumerate(tbar):
    inputs, targets = inputs.cuda(), targets.cuda()
    detections = yolo(inputs)
    mAP_batch = mAP(detections, targets, args.reso)
    mAPs += mAP_batch
    tbar.set_description("mAP=%.2f%%" % (np.mean(mAPs) * 100))

    if save_img == True:
      img_path = opj(config.datasets[args.dataset]['train_root'], 'train2017', names[0])
      img_name = img_path.split('/')[-1]

      try:
        detection = detections[detections[:, 0] == 0]
      except Exception:
        print(emojify("\nDetection disappeared :scream:\n"))
        img = Image.open(img_path)
      else:
        img = draw_detection(img_path, detection, yolo.reso)

      img.save(config.evaluate['result_dir'], img_name)

  return mAPs


if __name__ == '__main__':
  # 1. Parsing arguments
  print(emojify("\n==> Parsing arguments :hammer:\n"))
  assert args.reso % 32 == 0, emojify("Resolution must be interger times of 32 :shit:")
  for arg in vars(args):
    print(arg, ':', getattr(args, arg))

  # 2. Prepare data
  print(emojify("\n==> Preparing data ... :coffee:\n"))
  img_datasets, dataloader = prepare_val_dataset(args.dataset, args.reso, args.batch)
  print("Number of evaluate images:", len(img_datasets))

  # 3. Loading network
  print(emojify("\n==> Loading network ... :hourglass:\n"))
  yolo = YOLOv3(cfg, args.reso).cuda()
  if args.epoch != 0:
    start_epoch, _, state_dict = load_checkpoint(opj(config.CKPT_ROOT, args.dataset), args.epoch)
    yolo.load_state_dict(state_dict)
  print("Checkpoint epoch:", start_epoch)

  print(emojify("\n==> Evaluating ...\n"))
  yolo.eval()
  with torch.no_grad():
    mAPs = val(dataloader, yolo, args.save)
  print(emojify("Done! mAP: %.3f :+1:\n" % (np.mean(mAPs) * 100)))

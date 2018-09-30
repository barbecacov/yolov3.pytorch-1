import os
import argparse
import warnings
from tqdm import tqdm
from PIL import Image
from pyemojify import emojify
opj = os.path.join
warnings.filterwarnings("ignore")

import config
from model import YOLOv3
from dataset import prepare_demo_dataset
from utils import draw_detection, load_checkpoint


def parse_arg():
  parser = argparse.ArgumentParser(description='YOLO v3 training')
  parser.add_argument('--reso', default=416, type=int, help="Input image resolution of the network")
  parser.add_argument('--epoch', default=-1, type=int, help="Epoch of checkpoint. -1 for official pre trained checkpoint")
  parser.add_argument('--dataset', default='coco', choices=['coco'], type=str, help="Trained dataset name")
  return parser.parse_args()


args = parse_arg()
cfg = config.network['coco']['cfg']              # model cfg file path


if __name__ == '__main__':
  print(emojify("\n==> Parsing arguments ... :hammer:\n"))
  assert args.reso % 32 == 0, "Resolution must be interger times of 32"
  for arg in vars(args):
    print(arg, ':', getattr(args, arg))

  print(emojify("\n==> Prepare data ... :coffee:\n"))
  img_datasets, dataloader = prepare_demo_dataset(config.demo['images_dir'], args.reso)
  print("Number of demo images:", len(img_datasets))

  print(emojify("\n==> Loading network ... :hourglass:\n"))
  yolo = YOLOv3(cfg, args.reso).cuda()
  start_epoch, _, state_dict = load_checkpoint(opj(config.CKPT_ROOT, args.dataset), args.epoch)
  yolo.load_state_dict(state_dict)
  print("Checkpoint epoch:", start_epoch)

  print(emojify("\n==> Evaluation ...\n"))
  yolo.eval()
  for batch_idx, (inputs, _) in enumerate(tqdm(dataloader, ncols=80)):
    inputs = inputs.cuda()
    detections = yolo(inputs)

    img_path = img_datasets.get_path(batch_idx)
    img_name = img_path.split('/')[-1]
    img = draw_detection(img_path, detections.data.cpu().numpy(), args.reso)
    img.save(opj(config.demo['result_dir'], img_name))
  print(emojify("Done! :+1:\n"))

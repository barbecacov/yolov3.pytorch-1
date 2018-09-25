import os
import argparse
import warnings
from tqdm import tqdm
from PIL import Image
opj = os.path.join
warnings.filterwarnings("ignore")

import config
from model import YOLOv3
from utils import save_detection
from dataset import prepare_test_dataset


def parse_arg():
  parser = argparse.ArgumentParser(description='YOLO v3 training')
  parser.add_argument('--reso', default=640, type=int, help="Input image resolution of the network")
  return parser.parse_args()


if __name__ == '__main__':
  print("\n==> Parsing arguments ...\n")
  args = parse_arg()
  num_classes = config.datasets['coco']['num_classes']
  cfg = config.network['cfg']          # model cfg file path
  weights = config.network['weights']  # pretrained weights path
  images_dir = config.test['images_dir']   # eval images directory
  result_dir = config.test['result_dir']     # detection result directory
  assert args.reso % 32 == 0, "Resolution must be interger times of 32"

  print("\n==> Loading network ...\n")
  yolo = YOLOv3(cfg, args.reso)
  yolo.load_weights(weights)
  yolo = yolo.cuda()

  print("\n==> Loading data ...\n")
  img_datasets, dataloader = prepare_test_dataset(images_dir, args.reso)
  print("# Test images:", len(img_datasets))

  print("\n==> Evaluation ...\n")
  yolo.eval()
  for batch_idx, (inputs, _) in enumerate(tqdm(dataloader, ncols=80)):
    inputs = inputs.cuda()
    detections = yolo(inputs)

    img_path = img_datasets.get_path(batch_idx)
    save_detection(img_path, detections.data.cpu().numpy(), result_dir, args.reso)

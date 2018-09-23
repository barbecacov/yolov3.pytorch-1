import os
import argparse
import warnings
from PIL import Image
opj = os.path.join
warnings.filterwarnings("ignore")

import config
from model import YOLOv3, nms
from utils import prepare_eval_dataset, save_detection



def parse_arg():
  parser = argparse.ArgumentParser(description='YOLO v3 training')
  parser.add_argument('--reso', default=416, type=int, help="Input image resolution of the network")
  return parser.parse_args()


if __name__ == '__main__':
  print("\n==> Parsing arguments ...\n")
  args = parse_arg()
  num_classes = config.datasets['coco']['num_classes']
  cfg = config.network['cfg']          # model cfg file path
  weights = config.network['weights']  # pretrained weights path
  images = config.test['images_dir']   # eval images directory
  dets = config.test['result_dir']     # detection result directory
  assert args.reso % 32 == 0, "Resolution must be interger times of 32"

  print("\n==> Loading network ...\n")
  yolo = YOLOv3(cfg, args.reso)  # TODO: input dim
  yolo.load_weights(weights)
  yolo = yolo.cuda()

  print("\n==> Loading data ...\n")
  img_datasets, dataloader = prepare_eval_dataset(images, args.reso)
  print("Test images number:", len(img_datasets))

  print("\n==> Evaluation ...\n")
  yolo.eval()
  # tbar = tqdm(dataloader, ascii=True)
  for batch_idx, (inputs, _) in enumerate(dataloader):
    inputs = inputs.cuda()
    prediction = yolo(inputs)
    detection = nms(prediction, num_classes)
  
    img_path = img_datasets.get_path(batch_idx)
    save_detection(img_path, detection.data.cpu().numpy(), dets, args.reso)


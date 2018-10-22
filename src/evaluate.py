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
from dataset import prepare_val_dataset, prepare_train_dataset
from utils import draw_detection, load_checkpoint, mAP


def parse_arg():
    parser = argparse.ArgumentParser(description='YOLO v3 training')
    parser.add_argument('--reso', default=416, type=int, help="Input image resolution")
    parser.add_argument('--batch', default=12, type=int, help="Batch size")
    parser.add_argument('--dataset', default='voc', choices=['tejani', 'coco'], type=str, help="Dataset name")
    parser.add_argument('--checkpoint', default='-1.-1', type=str, help="Checkpoint name in format: `epoch.iteration`")
    parser.add_argument('--save', action='store_true', help="Save image during validation")
    parser.add_argument('--gpu', default='0', help="GPU ids")
    return parser.parse_args()


args = parse_arg()
cfg = config.network[args.dataset]['cfg']
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def val(valloader, yolo, save_img=True):
    """Validation wrapper

    Args
      valloader: (Dataloader) validation data loader 
      yolo: (nn.Module) YOLOv3 model
      save_img: (bool) whether to save images during validation
    """
    mAPs = []
    tbar = tqdm(valloader, ncols=80, ascii=True)
    for batch_idx, (names, inputs, targets) in enumerate(tbar):
        inputs = inputs.cuda()
        detections = yolo(inputs)
        mAP_batch = mAP(detections, targets, args.reso)
        mAPs += mAP_batch
        tbar.set_description("mAP=%.2f" % (np.mean(mAPs) * 100))

        if save_img == True and batch_idx % 4 == 0:
            img_path = opj(config.datasets[args.dataset]['val_imgs'], names[0])
            img_name = img_path.split('/')[-1]

            try:
                detection = detections[detections[:, 0] == 0]
            except Exception:
                img = Image.open(img_path)
            else:
                img = draw_detection(img_path, detection, yolo.reso, type='pred')

            img.save(opj(config.evaluate['result_dir'], img_name))

    return mAPs


if __name__ == '__main__':
    # 1. Parsing arguments
    print(emojify("\n==> Parsing arguments :zap:\n"))
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
    start_epoch, start_iteration = args.checkpoint.split('.')
    start_epoch, start_iteration, state_dict = load_checkpoint(
        opj(config.CKPT_ROOT, args.dataset),
        int(start_epoch),
        int(start_iteration)
    )
    yolo.load_state_dict(state_dict)
    print("Model starts training from epoch %d iteration %d" % (start_epoch, start_iteration))

    print(emojify("\n==> Evaluating ...\n"))
    if args.save == True:
        os.system('rm ' + opj(config.evaluate['result_dir'], '*.jpg'))
    yolo.eval()
    with torch.no_grad():
        mAPs = val(dataloader, yolo, args.save)
    print(emojify("Done! mAP: %.3f :+1:\n" % (np.mean(mAPs) * 100)))

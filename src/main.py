import os
import time
import random
import argparse

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from config import cfg
from models.posenet import PoseNet
from datasets.hrp import HRPDataset
from loss import HeatmapLoss
from engine import train, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--checkpoint', type=str)
    return parser


def main(args):
    cfg.merge_from_file(args.config)
    cfg.freeze()
    print(cfg)

    if cfg.SEED != -1:
        seed = cfg.SEED
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    writer = SummaryWriter(args.output_dir)

    device = torch.device(cfg.DEVICE)

    model = PoseNet((cfg.DATASET.NUM_LIMBS, cfg.DATASET.NUM_KEYPOINTS + 1), cfg.MODEL.FINAL_KERNEL, cfg.MODEL.BACKBONE, cfg.MODEL.PRETRAINED)
    if args.eval:
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    model = model.to(device)

    transforms_list_train = [
        # torchvision.transforms.ToPILImage(),
        # torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
        torchvision.transforms.ToTensor()
    ]
    transforms_list_val = [torchvision.transforms.ToTensor()]
    if None not in (cfg.DATASET.MEAN, cfg.DATASET.STD):
        transforms_list_train.append(torchvision.transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD))
        transforms_list_val.append(torchvision.transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD))
    transforms_train = torchvision.transforms.Compose(transforms_list_train)
    transforms_val = torchvision.transforms.Compose(transforms_list_val)

    dataset_train = HRPDataset(cfg, args.dataset_path, f'{cfg.DATASET.TRAIN}.json', True, False, transforms_train)
    dataset_val = HRPDataset(cfg, args.dataset_path, f'{cfg.DATASET.TEST}.json', False, cfg.TEST.KEEP_RATIO, transforms_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    criterion = HeatmapLoss(use_weight=cfg.LOSS.USE_WEIGHT).to(device)

    if args.eval:
        evaluate(model, criterion, data_loader_val, device, cfg)
        return

    if cfg.TRAIN.OPTIMIZER == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WD, amsgrad=False)
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, amsgrad=False)
    elif cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WD, nesterov=False)

    lr_scheduler = None
    if cfg.TRAIN.LR_FACTOR > 0:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.TRAIN.LR_STEPS, cfg.TRAIN.LR_FACTOR)

    min_loss = float('inf')

    for epoch in range(cfg.NUM_EPOCHS):
        start_time = time.time() 
        loss_train = train(model, criterion, data_loader_train, optimizer, device, epoch, cfg)
        elapsed_time = int(time.time() - start_time) + 1
        print('Train in {}m, {}s'.format(elapsed_time // 60, elapsed_time % 60))
        writer.add_scalar('loss/train', loss_train, epoch + 1)

        start_time = time.time()
        loss_val, ap_val = evaluate(model, criterion, data_loader_val, device, cfg)
        elapsed_time = int(time.time() - start_time) + 1
        print('Test in {}m, {}s: loss: {:.5f}'.format(elapsed_time // 60, elapsed_time % 60, loss_val))
        writer.add_scalar('loss/val', loss_val, epoch + 1)
        writer.add_scalar('ap/val', ap_val, epoch + 1)

        # save checkpoint
        if loss_val < min_loss:
            min_loss = loss_val
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoint.pth'))

        if lr_scheduler is not None:
            lr_scheduler.step()

    writer.close()


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    main(args)

import torch
import numpy as np

from datasets.pose_eval import PoseEval
from utils.ops import group_dets, aggregate_multi_scale


def train(model, criterion, data_loader, optimizer, device, epoch, cfg):
    steps = len(data_loader)
    losses = []
 
    model.train()

    for i, (inputs, targets, targets_weight, _) in enumerate(data_loader):
        inputs = inputs.to(device)
        targets = list(map(lambda x: (x[0].to(device, non_blocking=True), x[1].to(device, non_blocking=True) if isinstance(x[1], torch.Tensor) else x[1]), targets))
        targets_weight = list(map(lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x, targets_weight))

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets, targets_weight)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % cfg.PRINT_FREQ == 0 or (i + 1) == steps:
            avg_loss = np.mean(losses)
            print('Epoch: {:<3d} [{:<2d}/{:<2d}] loss: {:.5f} ({:.5f})'.format(epoch + 1, i + 1, steps, losses[-1], avg_loss))

    return avg_loss


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, cfg):
    max_num_dets, num_scales, project, nms_kernel, det_thr = cfg.DATASET.MAX_NUM_DETECTIONS, cfg.MODEL.NUM_SCALES, cfg.TEST.PROJECT, cfg.TEST.NMS_KERNEL, cfg.TEST.DETECTION_THRESHOLD
    num_keypoints = len(data_loader.dataset.flip_order)
    stride = np.ones_like(data_loader.dataset.stride) if project else data_loader.dataset.stride
    test_limb_cfg = dict(cfg.TEST.LIMB)
    pose_evaluator = PoseEval(data_loader.dataset.coco)
    losses = []

    model.eval()

    for inputs, targets, targets_weight, metas in data_loader:
        inputs = inputs.to(device)
        targets = list(map(lambda x: (x[0].to(device, non_blocking=True), x[1].to(device, non_blocking=True) if isinstance(x[1], torch.Tensor) else x[1]), targets))
        targets_weight = list(map(lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x, targets_weight))

        outputs = model(inputs)
        loss = criterion(outputs, targets, targets_weight)

        losses.append(loss.item())

        output_size = (inputs.size(-2), inputs.size(-1)) if project else None
        num_hms_scales = (len(outputs) - 2) // 2
        hms = aggregate_multi_scale(outputs[2:2 + num_hms_scales], num_scales, num_keypoints, output_size)
        limbs = aggregate_multi_scale(outputs[2 + num_hms_scales:], num_scales, output_size=(hms.size(-2), hms.size(-1)))
        grouped, dets = group_dets(hms, limbs, data_loader.dataset.limbs, test_limb_cfg, nms_kernel, max_num_dets, det_thr)
        grouped[..., :2] *= stride
        pose_evaluator.collect(grouped, metas)

    avg_loss = np.mean(losses)
    ap = pose_evaluator.evaluate()

    return avg_loss, ap

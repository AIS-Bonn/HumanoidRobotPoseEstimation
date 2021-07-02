import torch
from torch import nn


class MSELoss(nn.Module):

    def __init__(self, reduction='mean', use_weight=False):
        super(MSELoss, self).__init__()
        self.reduction = reduction
        self.use_weight = use_weight

    def forward(self, inputs, targets, targets_weight=None):
        loss = torch.pow(inputs - targets, 2)

        if self.use_weight:
            loss = loss * targets_weight.unsqueeze(1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()

        return loss


class HeatmapLoss(nn.Module):

    def __init__(self, use_weight=False):
        super(HeatmapLoss, self).__init__()
        self.keypoint_loss = MSELoss(reduction='none', use_weight=use_weight)
        self.limb_loss = MSELoss(reduction='none', use_weight=use_weight)

    def forward(self, inputs, targets, targets_weight=None):
        num_scales = len(targets)
        limbs_start_idx = len(inputs) - num_scales
        keypoints_start_idx = limbs_start_idx - (len(inputs) - 2) // 2
        keypoints, limbs = [], []

        for i in range(num_scales):  # loop through scales
            keypoints_inputs = inputs[keypoints_start_idx + i]
            limbs_inputs = inputs[limbs_start_idx + i]
            for j in range(keypoints_inputs.size(1)):  # loop through stacks
                keypoints.append(self.keypoint_loss(keypoints_inputs[:, j], targets[i][0], targets_weight[i]).mean(dim=(1, 2, 3)))

            if limbs_inputs is not None:
                for j in range(limbs_inputs.size(1)):  # loop through stacks
                    limbs.append(self.limb_loss(limbs_inputs[:, j], targets[i][1], targets_weight[i]).mean(dim=(1, 2, 3)))

        return torch.stack(keypoints, dim=1).mean(dim=0).sum() + torch.stack(limbs, dim=1).mean(dim=0).sum()

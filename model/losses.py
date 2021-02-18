from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from model.metrics import SMOOTH

EPS = SMOOTH


class LSoftLoss(nn.Module):
    def __init__(self, beta=0.5, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # https://arxiv.org/pdf/1901.01189.pdf
        with torch.no_grad():
            pred = torch.sigmoid(input)
            target_update = self.beta * target + (1 - self.beta) * pred

        loss = F.binary_cross_entropy_with_logits(input, target_update, reduction=self.reduction)
        return loss


class NPLoss(nn.Module):
    def __init__(self, gamma, confident_loss_func: Optional[nn.Module], noisy_loss_func: Optional[nn.Module],
                 scale_filtered_loss=False):
        super().__init__()
        self.gamma = gamma
        self.combined_noisy_loss = CombinedNoisyLoss(confident_loss_func, noisy_loss_func, scale_filtered_loss)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            pred = torch.sigmoid(input)  # type:torch.Tensor
            conf_negative = pred < self.gamma  # type:torch.Tensor
            conf_positive = pred > (1 - self.gamma)  # type:torch.Tensor
            conf_mask = torch.logical_or(conf_negative, conf_negative)

            conf_mask = conf_mask.float()
            conf_targets = conf_positive.float()

        loss = self.combined_noisy_loss(input, conf_targets, conf_mask)
        return loss


class CombinedNoisyLoss(nn.Module):
    def __init__(self, confident_loss_func: Optional[nn.Module], noisy_loss_func: Optional[nn.Module],
                 scale_filtered_loss=False, combined_noisy_loss_reduction='mean'):
        super().__init__()
        self.confident_loss_func = confident_loss_func
        self.noisy_loss_func = noisy_loss_func
        self.scale_filtered_loss = scale_filtered_loss
        self.combined_noisy_loss_reduction = combined_noisy_loss_reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor, confident_mask: torch.Tensor):
        batch_size = target.shape[0]
        input = input.flatten()
        target = target.flatten()
        confident_mask = confident_mask.flatten()

        confident_indexes = confident_mask.nonzero(as_tuple=False).squeeze(1)
        noisy_indexes = (confident_mask == 0).nonzero(as_tuple=False).squeeze(1)

        if self.combined_noisy_loss_reduction == 'none':
            noisy_mask = 1 - confident_mask
            if self.confident_loss_func and confident_indexes.shape[0] > 0:
                confident_loss = self.confident_loss_func(input, target)
                confident_loss = confident_loss * confident_mask
            else:
                confident_loss = torch.zeros((), device=input.device)

            if self.noisy_loss_func and noisy_indexes.shape[0] > 0:
                noisy_loss = self.noisy_loss_func(input, target)
                noisy_loss = noisy_loss * noisy_mask
            else:
                noisy_loss = torch.zeros((), device=input.device)

            loss = confident_loss + noisy_loss
            return loss

        confident_loss, noisy_loss = torch.zeros((), device=input.device), torch.zeros((), device=input.device)

        if self.confident_loss_func and confident_indexes.shape[0] > 0:
            confident_loss = self.confident_loss_func(input[confident_indexes], target[confident_indexes])
            confident_loss = torch.mean(confident_loss)
            if self.scale_filtered_loss:
                confident_loss = confident_loss * (confident_indexes.shape[0] / batch_size)

        if self.noisy_loss_func and noisy_indexes.shape[0] > 0:
            noisy_loss = self.noisy_loss_func(input[noisy_indexes], target[noisy_indexes])
            noisy_loss = torch.mean(noisy_loss)
            if self.scale_filtered_loss:
                noisy_loss = noisy_loss * (noisy_indexes.shape[0] / batch_size)

        if self.noisy_loss_func is None:
            return confident_loss
        elif self.confident_loss_func is None:
            return noisy_loss

        return 0.5 * confident_loss + 0.5 * noisy_loss


class CombinedTPLSoftLoss(CombinedNoisyLoss):
    def __init__(self, beta, scale_filtered_loss, combined_noisy_loss_reduction):
        super().__init__(nn.BCEWithLogitsLoss(reduction='none'), LSoftLoss(beta, reduction='none'),
                         scale_filtered_loss=scale_filtered_loss,
                         combined_noisy_loss_reduction=combined_noisy_loss_reduction)


class BCELossWithSmoothing(nn.BCEWithLogitsLoss):
    def __init__(self, epsilon, reduction='mean'):
        self.epsilon = epsilon
        super().__init__(reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        with torch.no_grad():
            target = (1 - self.epsilon) * target + self.epsilon / target.shape[-1]

        return super(BCELossWithSmoothing, self).forward(input, target)


class BinaryRecallLoss(torch.nn.Module):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = torch.sigmoid(input)

        tp = torch.sum(torch.mul(input, target), dim=1)
        fn = torch.sum(target, dim=1) - tp

        score = (tp + SMOOTH) / (tp + fn + SMOOTH)
        score = 1 - torch.mean(score)

        return score


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_func = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce_func(input, target)
        probas = torch.sigmoid(input)

        loss = torch.where(target >= 0.5, self.alpha * (1. - probas) ** self.gamma * bce_loss,
                           probas ** self.gamma * bce_loss)

        loss = loss.sum(dim=-1)
        loss = loss.mean()

        return loss

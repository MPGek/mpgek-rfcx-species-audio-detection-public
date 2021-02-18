from typing import Optional, List

import numpy as np
import torch

from model.losses import CombinedNoisyLoss


def prepare_fp_targets(data_class, device):
    targets = data_class.reshape((-1, 2, data_class.shape[-1] // 2))
    ones = torch.ones(targets.shape)
    present_targets = torch.where(targets > 0, ones, targets)
    present_targets, _ = present_targets.max(dim=1)
    targets = (targets[:, 0, :] + 1 - targets[:, 1, :]) / 2
    targets = targets * present_targets

    targets = targets.to(device)
    present_targets = present_targets.to(device)

    return present_targets, targets


def calculate_single_loss(criterion: torch.nn.Module, outputs, target, confident_mask: Optional[torch.Tensor]):
    if isinstance(criterion, CombinedNoisyLoss):
        if confident_mask is None:
            confident_mask = torch.where(target >= 0.5, 1, 0)
        return criterion(outputs, target, confident_mask)

    return criterion(outputs, target)


def calculate_losses(outputs: torch.Tensor, data_class: torch.Tensor, device, criterion_list, use_fp: bool,
                     aux_weights: Optional[List]):
    if not use_fp:
        present_mask = None
        targets = data_class.to(device)  # type:torch.Tensor
    else:
        present_mask, targets = prepare_fp_targets(data_class, device)

    if aux_weights is None:
        loss = calculate_single_loss(criterion_list[0], outputs, targets, present_mask)  # Loss with grad
    else:
        loss = 0
        for output, weight in zip(outputs, aux_weights):
            loss += criterion_list[0](output, targets) * weight
        outputs = outputs[0]

    with torch.no_grad():
        metrics = [calculate_single_loss(criterion, outputs, targets, present_mask) for criterion in criterion_list[1:]]

    return outputs, loss, metrics


def forward_pass(data_img, data_class, model, device, criterion_list, pseudo_label=False, use_fp=False,
                 aux_weights=None):
    inputs = data_img.to(device)

    # Forward pass
    outputs = model(inputs)  # .reshape(targets.shape)

    if criterion_list is None:
        return outputs, None, None

    return calculate_losses(outputs, data_class, device, criterion_list, use_fp, aux_weights)


def forward_pass_by_crops(data_img, data_class: torch.Tensor, model, device, criterion_list, validate_by_crops,
                          validate_crops_offset, apply_results_filter, record_ids, use_fp, aux_weights=None):
    inputs = data_img.to(device)  # type:torch.Tensor

    # validate by crops
    full_length = inputs.shape[3]
    repeat_count = int((full_length - validate_by_crops) / validate_crops_offset) + 2
    final_output = []

    for i in range(repeat_count):
        start = i * validate_crops_offset
        end = start + validate_by_crops
        if end > full_length:
            start = full_length - validate_by_crops
            end = full_length

        results_offset = None
        if apply_results_filter:
            results_filter = apply_results_filter(record_ids, start, end)
            if np.count_nonzero(results_filter) == 0:
                continue
            results_offset = -1000 * torch.tensor(1 - results_filter, dtype=data_class.dtype)
            results_offset = results_offset.to(device)

        input_item = inputs.narrow(-1, start, end - start)
        outputs = model(input_item)  # .reshape(targets.shape)

        # reduce values for the samples without targets in cuts
        if results_offset is not None:
            if aux_weights is None:
                outputs = outputs + results_offset
                final_output.append(outputs)
            else:
                if len(final_output) == 0:
                    final_output = [[] for _ in outputs]
                for idx, output in enumerate(outputs):
                    final_output[idx].append(output + results_offset)

    if aux_weights is None:
        final_output = torch.stack(final_output, dim=0)
        final_output, _ = torch.max(final_output, dim=0)
    else:
        final_output = [torch.stack(output, dim=0) for output in final_output]
        final_output = [torch.max(output, dim=0)[0] for output in final_output]

    return calculate_losses(final_output, data_class, device, criterion_list, use_fp, aux_weights)

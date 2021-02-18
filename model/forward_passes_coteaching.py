from typing import Optional

import numpy as np
import torch

from model.losses import CombinedNoisyLoss


def prepare_fp_targets_coteaching(data_class, device):
    targets = data_class.reshape((-1, 2, data_class.shape[-1] // 2))
    ones = torch.ones(targets.shape)
    present_targets = torch.where(targets > 0, ones, targets)
    present_targets, _ = present_targets.max(dim=1)
    targets = (targets[:, 0, :] + 1 - targets[:, 1, :]) / 2
    targets = targets * present_targets

    targets = targets.to(device)
    present_targets = present_targets.to(device)

    return present_targets, targets


def calculate_single_loss_coteaching(criterion: torch.nn.Module, outputs, target,
                                     confident_mask: Optional[torch.Tensor]):
    if isinstance(criterion, CombinedNoisyLoss):
        if confident_mask is None:
            confident_mask = torch.where(target >= 0.5, 1, 0)
        return criterion(outputs, target, confident_mask)

    return criterion(outputs, target)


def calculate_losses_coteaching(outputs_1: torch.Tensor, outputs_2: torch.Tensor, data_class: torch.Tensor, device_1,
                                device_2, criterion_list_1, criterion_list_2, remember_rate, high_loss_train,
                                use_fp: bool):
    if not use_fp:
        present_mask_1 = None
        present_mask_2 = None
        targets_1 = data_class.to(device_1)  # type:torch.Tensor
        targets_2 = data_class.to(device_2) if device_1 != device_2 else targets_1  # type:torch.Tensor
    else:
        present_mask_1, targets_1 = prepare_fp_targets_coteaching(data_class, device_1)
        if device_1 != device_2:
            present_mask_2, targets_2 = prepare_fp_targets_coteaching(data_class, device_2)
        else:
            present_mask_2 = present_mask_1
            targets_2 = targets_1

    loss_1_upd, loss_2_upd = calculate_coteach_double_loss(outputs_1, outputs_2, targets_1, targets_2, present_mask_1,
                                                           present_mask_2, data_class, device_1, device_2,
                                                           criterion_list_1, criterion_list_2, remember_rate,
                                                           high_loss_train)

    with torch.no_grad():
        metrics_1 = [torch.mean(calculate_single_loss_coteaching(criterion, outputs_1, targets_1, present_mask_1))
                     for criterion in criterion_list_1[1:]]
        metrics_2 = [torch.mean(calculate_single_loss_coteaching(criterion, outputs_2, targets_2, present_mask_2))
                     for criterion in criterion_list_2[1:]]

    return outputs_1, outputs_2, loss_1_upd, loss_2_upd, metrics_1, metrics_2


def calculate_coteach_double_loss(outputs_1, outputs_2, targets_1, targets_2, present_mask_1, present_mask_2,
                                  data_class, device_1, device_2, criterion_list_1, criterion_list_2, remember_rate,
                                  high_loss_train):
    outputs_1 = outputs_1.flatten()
    outputs_2 = outputs_2.flatten()
    # Copy targets tensors to avoid wrong high metrics on later measurements after modifying it for the Co-Teaching
    targets_1 = targets_1.clone().detach().flatten()
    targets_2 = targets_2.clone().detach().flatten()
    if present_mask_1 is not None:
        present_mask_1 = present_mask_1.flatten()
        present_mask_2 = present_mask_2.flatten()

    with torch.no_grad():
        loss_1 = calculate_single_loss_coteaching(criterion_list_1[0], outputs_1, targets_1, present_mask_1).flatten()
        loss_2 = calculate_single_loss_coteaching(criterion_list_2[0], outputs_2, targets_2, present_mask_2).flatten()

    if present_mask_1 is None:
        confident_mask = torch.tensor(np.where(data_class >= 0.5, 1, 0).flatten(), device='cpu')
    else:
        confident_mask = present_mask_1.cpu().flatten()
    noisy_mask = 1 - confident_mask
    confident_indexes = confident_mask.nonzero(as_tuple=False).squeeze(1)

    # set 0 to the confident mask to make them be in the begining of the sorted indexes
    loss_1 = loss_1.cpu() * noisy_mask
    loss_2 = loss_2.cpu() * noisy_mask

    ind_1_sorted = torch.argsort(loss_1)  # type:torch.Tensor
    ind_2_sorted = torch.argsort(loss_2)  # type:torch.Tensor

    num_remember = int(remember_rate * ind_1_sorted.shape[0])
    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    # adding high loss samples with results 0 or 1 according to it's predicted output (use prediction, not labels)
    num_high_loss_train = int(high_loss_train * (ind_1_sorted.shape[0] - num_remember))
    if num_high_loss_train > 0:
        ind_1_high_loss = ind_1_sorted[-num_high_loss_train:]
        ind_2_high_loss = ind_2_sorted[-num_high_loss_train:]

        with torch.no_grad():
            pred_1 = torch.where(outputs_1[ind_1_high_loss] >= 0, 1.0, 0.0).to(targets_1.device)
            pred_2 = torch.where(outputs_2[ind_2_high_loss] >= 0, 1.0, 0.0).to(targets_2.device)
            # set predicted results with exchange
            targets_2[ind_1_high_loss] = pred_1[:]
            targets_1[ind_2_high_loss] = pred_2[:]

        ind_1_update = torch.cat((ind_1_update, ind_1_high_loss), dim=0)
        ind_2_update = torch.cat((ind_2_update, ind_2_high_loss), dim=0)

    ind_1_update = ind_1_update.to(device_2)
    ind_2_update = ind_2_update.to(device_1)

    # exchange
    outputs_1_upd = outputs_1[ind_2_update]
    targets_1_upd = targets_1[ind_2_update]
    present_mask_1_upd = present_mask_1[ind_2_update] if present_mask_1 is not None else None

    outputs_2_upd = outputs_2[ind_1_update]
    targets_2_upd = targets_2[ind_1_update]
    present_mask_2_upd = present_mask_2[ind_1_update] if present_mask_2 is not None else None

    loss_1_upd = calculate_single_loss_coteaching(criterion_list_1[0], outputs_1_upd, targets_1_upd, present_mask_1_upd)
    loss_2_upd = calculate_single_loss_coteaching(criterion_list_2[0], outputs_2_upd, targets_2_upd, present_mask_2_upd)

    # scale confident losses by 2
    confident_len = confident_indexes.shape[0]
    loss_1_upd[:confident_len] = loss_1_upd[:confident_len] * 2
    loss_2_upd[:confident_len] = loss_2_upd[:confident_len] * 2

    loss_1_upd = torch.mean(loss_1_upd)
    loss_2_upd = torch.mean(loss_2_upd)
    return loss_1_upd, loss_2_upd


def forward_pass_coteaching(data_img, data_class, model_1, device_1, model_2, device_2, criterion_list_1,
                            criterion_list_2, remember_rate_value, high_loss_train, use_fp=False):
    inputs_1 = data_img.to(device_1)
    inputs_2 = data_img.to(device_2) if device_1 != device_2 else inputs_1

    outputs_1 = model_1(inputs_1)  # .reshape(targets.shape)
    outputs_2 = model_2(inputs_2)  # .reshape(targets.shape)

    if criterion_list_1 is None:
        return outputs_1, outputs_2, None, None, None, None

    return calculate_losses_coteaching(outputs_1, outputs_2, data_class, device_1, device_2, criterion_list_1,
                                       criterion_list_2, remember_rate_value, high_loss_train, use_fp)


def forward_pass_by_crops_coteaching(data_img, data_class: torch.Tensor, model_1, device_1, model_2, device_2,
                                     criterion_list_1, criterion_list_2, remember_rate_value, high_loss_train,
                                     validate_by_crops, validate_crops_offset, apply_results_filter, record_ids,
                                     use_fp):
    inputs_1 = data_img.to(device_1)  # type:torch.Tensor
    inputs_2 = data_img.to(device_2) if device_1 != device_2 else inputs_1  # type:torch.Tensor

    # validate by crops
    full_length = inputs_1.shape[3]
    repeat_count = int((full_length - validate_by_crops) / validate_crops_offset) + 2
    final_output_1, final_output_2 = [], []

    for i in range(repeat_count):
        start = i * validate_crops_offset
        end = start + validate_by_crops
        if end > full_length:
            start = full_length - validate_by_crops
            end = full_length

        results_offset_1, results_offset_2 = None, None
        if apply_results_filter:
            results_filter = apply_results_filter(record_ids, start, end)
            if np.count_nonzero(results_filter) == 0:
                continue
            results_offset_cpu = -1000 * torch.tensor(1 - results_filter, dtype=data_class.dtype)
            results_offset_1 = results_offset_cpu.to(device_1)
            results_offset_2 = results_offset_cpu.to(device_2)

        input_item_1 = inputs_1.narrow(-1, start, end - start)
        input_item_2 = inputs_2.narrow(-1, start, end - start)

        outputs_1 = model_1(input_item_1)  # .reshape(targets.shape)
        outputs_2 = model_2(input_item_2)  # .reshape(targets.shape)

        # reduce values for the samples without targets in cuts
        if results_offset_1 is not None:
            outputs_1 = outputs_1 + results_offset_1
            final_output_1.append(outputs_1)

        if results_offset_2 is not None:
            outputs_2 = outputs_2 + results_offset_2
            final_output_2.append(outputs_2)

    final_output_1 = torch.stack(final_output_1, dim=0)
    final_output_1, _ = torch.max(final_output_1, dim=0)

    final_output_2 = torch.stack(final_output_2, dim=0)
    final_output_2, _ = torch.max(final_output_2, dim=0)

    return calculate_losses_coteaching(final_output_1, final_output_2, data_class, device_1, device_2,
                                       criterion_list_1, criterion_list_2, remember_rate_value, high_loss_train, use_fp)

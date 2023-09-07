# coding=utf-8
# Copyright (c) 2023 ADEPT AI LABS INC.
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""General utilities."""

import re
import os
import sys
import warnings

import torch
from torch.nn.parallel import DistributedDataParallel as torchDDP
from typing import Optional

from megatron import get_args
from megatron import get_adlr_autoresume
from megatron import mpu
from megatron.model.module import param_is_not_shared
from megatron.mpu.layers import param_is_tensor_parallel_unique


def unwrap_model(model, module_instances=(torchDDP)):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def calculate_num_params_and_params_l2_norm(model):
    """Calculate l2 norm of parameters"""
    import amp_C
    from apex.multi_tensor_apply import multi_tensor_applier

    args = get_args()
    if not isinstance(model, list):
        model = [model]
    # Remove duplicate params.
    params_data = []
    num_params = 0
    for model_ in model:
        for param in model_.parameters():
            is_not_shared = param_is_not_shared(param)
            is_tp_unique = param_is_tensor_parallel_unique(param)
            if is_not_shared and is_tp_unique:
                num_params += param.numel()
                if args.bf16:
                    params_data.append(param.data.float())
                else:
                    params_data.append(param.data)
    # Calculate norm
    dummy_overflow_buf = torch.cuda.IntTensor([0])
    norm, _ = multi_tensor_applier(
        amp_C.multi_tensor_l2norm,
        dummy_overflow_buf,
        [params_data],
        False,  # no per-parameter norm
    )
    norm_2 = norm * norm
    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(norm_2, op=torch.distributed.ReduceOp.SUM, group=mpu.get_model_parallel_group())

    # Sum num params across all model-parallel GPUs.
    num_params = torch.tensor(num_params, dtype=torch.int64, device=torch.cuda.current_device())
    torch.distributed.all_reduce(num_params, op=torch.distributed.ReduceOp.SUM, group=mpu.get_model_parallel_group())

    return num_params.item(), norm_2.item() ** 0.5


def calculate_per_layer_grad_norms(model, normalizing_factor):
    """Calculate per-layer grad norms. Assumes gradients have already been all-reduced across data group."""
    import amp_C
    from apex.multi_tensor_apply import multi_tensor_applier

    if mpu.get_pipeline_model_parallel_world_size() > 1:
        warnings.warn(
            "calculate_per_layer_grad_norms is not yet implemented for pipeline parallel models, skipping calculation!"
        )
        return {}
    if mpu.get_tensor_model_parallel_world_size() > 1:
        warnings.warn(
            "calculate_per_layer_grad_norms is not yet implemented for tensor parallel models, skipping calculation!"
        )
        return {}

    if not isinstance(model, list):
        model = [model]
    # Remove duplicate params.
    names = []
    grads = []
    for model_ in model:
        for name, param in model_.named_parameters():
            if hasattr(param, "main_grad"):
                grad = param.main_grad
            elif param.grad is not None:
                grad = param.grad
            else:
                continue
            names.append(name)
            grads.append(grad)
    # Calculate norms.
    dummy_overflow_buf = torch.cuda.IntTensor([0])
    _, norms = multi_tensor_applier(
        amp_C.multi_tensor_l2norm,
        dummy_overflow_buf,
        [grads],
        True,  # include per-parameter norm
    )

    norms_2 = norms * norms
    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(norms_2, op=torch.distributed.ReduceOp.SUM, group=mpu.get_model_parallel_group())
    norms = norms_2**0.5

    grad_norms = {}
    for i, name in enumerate(names):
        # Consolidate stats for individual layers into a single combined stat.
        name = re.sub(r"\.layers\.[0-9]+", ".all_layers_combined", name)
        # Consolidate stats for individual qk_layernorms into a single combined stat.
        name = re.sub(r"\.q_layernorms\.[0-9]+", ".all_q_layernorms_combined", name)
        name = re.sub(r"\.k_layernorms\.[0-9]+", ".all_k_layernorms_combined", name)
        if name in grad_norms:
            grad_norms[name] += norms[i]
        else:
            grad_norms[name] = norms[i]

    for k in grad_norms.keys():
        grad_norms[k] = grad_norms[k] / normalizing_factor

    return grad_norms


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses, group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    return averaged_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + " memory (MB)"
    string += " | allocated: {}".format(torch.cuda.memory_allocated() / mega_bytes)
    string += " | max allocated: {}".format(torch.cuda.max_memory_allocated() / mega_bytes)
    string += " | reserved: {}".format(torch.cuda.memory_reserved() / mega_bytes)
    string += " | max reserved: {}".format(torch.cuda.max_memory_reserved() / mega_bytes)
    if mpu.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string), flush=True)


def check_adlr_autoresume_termination(iteration, model, optimizer, opt_param_scheduler):
    """Check for autoresume signal and exit if it is received."""
    from megatron.checkpointing import save_checkpoint

    args = get_args()
    autoresume = get_adlr_autoresume()
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if autoresume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            autoresume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def get_ltor_masks_and_position_ids(data, eod_token, reset_position_ids, reset_attention_mask, eod_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)).view(
        att_mask_batch, 1, seq_length, seq_length
    )

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):
            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1) :] -= i + 1 - prev_index
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


def print_rank_0(*args):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args, flush=True)
    elif os.environ.get("RANK", "0") == "0":
        print(*args, flush=True)


def print_once_per_data_rank(*args):
    if not torch.distributed.is_initialized() or (
        mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_pipeline_model_parallel_rank() == 0
    ):
        print(*args, flush=True)


def is_last_rank():
    return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)


def print_rank_last(*args):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(*args, flush=True)
    else:
        print(*args, flush=True)


def get_total_eval_iters(train_iters: Optional[int], eval_interval: int, iter_per_eval: int) -> Optional[int]:
    """Helper function to calculate evaluation iters for iteration based training, otherwise ignore."""
    if train_iters is not None:
        return (train_iters // eval_interval + 1) * iter_per_eval

    return None

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

"""Pretrain utilities."""

import math
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import (
    get_args,
    mpu,
)
from megatron.checkpointing import (
    load_state_dicts_and_update_args,
    update_model_and_optim_from_loaded_data,
)
from megatron.initialize import initialize_megatron
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module, ModelType

from megatron.utils import (
    unwrap_model,
)


def get_model(
    model_provider_func,
    model_type=ModelType.encoder_or_decoder,
    wrap_with_ddp=True,
):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    # Build model.
    if (
        mpu.get_pipeline_model_parallel_world_size() > 1
        and args.virtual_pipeline_model_parallel_size is not None
    ):
        assert (
            model_type != ModelType.encoder_and_decoder
        ), "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process, post_process=post_process
            )
            this_model.model_type = model_type
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                assert (
                    args.pipeline_model_parallel_split_rank is not None
                ), "Split rank needs to be specified for model with both encoder and decoder"
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = args.pipeline_model_parallel_split_rank
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder,
            )
        else:
            model = model_provider_func(
                pre_process=pre_process, post_process=post_process
            )
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            mpu.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(
            " > number of parameters on (tensor, pipeline) "
            "model parallel rank ({}, {}): {}".format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                sum(
                    [
                        sum([p.nelement() for p in model_module.parameters()])
                        for model_module in model
                    ]
                ),
            ),
            flush=True,
        )

    # GPU allocation.
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    if wrap_with_ddp:
        if args.DDP_impl == "torch":
            i = torch.cuda.current_device()
            model = [
                torchDDP(
                    model_module,
                    device_ids=[i],
                    output_device=i,
                    process_group=mpu.get_data_parallel_group(),
                )
                for model_module in model
            ]

        elif args.DDP_impl == "local":
            model = [
                LocalDDP(
                    model_module,
                    args.accumulate_allreduce_grads_in_fp32,
                    args.use_contiguous_buffers_in_local_ddp,
                )
                for model_module in model
            ]
            # broad cast params from data parallel src rank to other data parallel ranks
            if args.data_parallel_random_init:
                for model_module in model:
                    model_module.broadcast_params()
        else:
            raise NotImplementedError(
                "Unknown DDP implementation specified: "
                "{}. Exiting.".format(args.DDP_impl)
            )

    return model

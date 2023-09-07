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

"""Megatron initialization."""

import argparse
import git
import random
import os
import sys
import time

import numpy as np
import torch
from datetime import timedelta

from megatron import fused_kernels
from megatron import get_adlr_autoresume
from megatron import get_args
from megatron import get_tensorboard_writer
from megatron import mpu
from megatron.arguments import parse_args, validate_args
from megatron.checkpointing import load_args_from_checkpoint
from megatron.global_vars import set_global_variables
from megatron.mpu import (
    set_tensor_model_parallel_rank,
    set_tensor_model_parallel_world_size,
)
from megatron.model.transformer import bias_dropout_add_fused_train
from megatron.model.fused_bias_gelu import bias_gelu
import requests
import json

ENDPOINT = "http://experiments.research.adept.ai/run/slurm_job_slack_monitor"

# This is needed here to avoid a circular dependency on the main version
def _local_print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def initialize_megatron(
    extra_args_provider=None,
    args_defaults={},
    ignore_unknown_args=False,
    allow_no_cuda=False,
):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """
    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), "Megatron requires CUDA."

    args = parse_args(extra_args_provider, ignore_unknown_args=ignore_unknown_args)
    if not args.untie_embeddings:
        # Set untie-embeddings to None by default so it's not checked when a model checkpoint which doesn't use this param, is loaded.
        # Note: We might need to do this for a lot of other args as well.
        args.untie_embeddings = None

    # Initialize basic torch distributed
    _initialize_torch_distributed(args)

    if args.load:
        # if a checkpoint is specified, we always load arguments from it
        args = load_args_from_checkpoint(args)
    args = validate_args(args, args_defaults)

    if args.deterministic:
        torch.use_deterministic_algorithms(mode=True)

    # set global args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(args)

    # Initialize megatron's customizations to the distributed setup (including parallelism)
    _initialize_megatron_distributed()


def _initialize_torch_distributed(args: argparse.Namespace):
    """Initialize torch.distributed and mpu."""

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():

        if args.rank == 0:
            print(
                "torch distributed is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
    else:

        if args.rank == 0:
            print("> initializing torch distributed ...", flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert (
                    args.local_rank == device
                ), "expected local-rank to be the same as rank % device-count."
            else:
                args.local_rank = device
            torch.cuda.set_device(device)
    # Call the init process
    if args.cross_block_networking or args.force_socket_networking:
        os.environ["NCCL_NET"] = "Socket"
    else:
        os.environ["NCCL_NET"] = "IB"
    torch.distributed.init_process_group(
        backend=args.default_backend,
        world_size=args.world_size,
        rank=args.rank,
        timeout=timedelta(minutes=args.distributed_comms_timeout),
    )

    mpu.force_communicator_creation(
        rank=torch.distributed.get_rank(),
        world_size=torch.distributed.get_world_size(),
        all_reduce=True,
        all_gather=True,
        barrier=True,
    )
    _local_print_rank_0("_initialize_torch_distributed init")


def _initialize_megatron_distributed():
    """Initialize distributed."""
    # Parse arguments

    # torch.distributed initialization
    args = get_args()

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if torch.cuda.device_count() > 0:
        if mpu.model_parallel_is_initialized():
            _local_print_rank_0(
                "mpu.initialize_model_parallel skipped, model parallel is already initialized"
            )
        else:
            mpu.initialize_model_parallel(
                tensor_model_parallel_size_=args.tensor_model_parallel_size,
                pipeline_model_parallel_size_=args.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size_=args.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank_=args.pipeline_model_parallel_split_rank,
                default_backend=args.default_backend,
                p2p_backend="nccl",
                cross_block_networking=args.cross_block_networking,
                share_word_embeddings=not args.untie_embeddings,
                force_socket_networking=args.force_socket_networking,
            )

            _local_print_rank_0("mpu.initialize_model_parallel init")

    # Random seeds for reproducibility.
    if args.rank == 0:
        print("> setting random seeds to {} ...".format(args.seed))
    _set_random_seed(args.seed, args.data_parallel_random_init)

    # Autoresume.
    _init_autoresume()

    # Compile dependencies.
    _compile_dependencies()
    _local_print_rank_0("_compile_dependencies init")


def _compile_dependencies():

    args = get_args()

    # ==================
    # Load fused kernels
    # ==================

    compilation_succeeded = torch.tensor([0]).cuda()  # Communicate success/failure
    # Always build on rank zero first.
    assert (
        args.force_build_fused_kernels == False
    ), "force_build_fused_kernels is not supported anymore."
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        try:
            fused_kernels.load(args.force_build_fused_kernels)
            compilation_succeeded[0] = 1
            torch.distributed.broadcast(compilation_succeeded, 0)
        except fused_kernels.CompilationTimeoutError as e:
            print("ERROR", e, flush=True)
            torch.distributed.broadcast(compilation_succeeded, 0)
            sys.exit(1)
    else:
        torch.distributed.broadcast(compilation_succeeded, 0)
        if compilation_succeeded[0] == 0:
            print(
                "ERROR: fused kernel compilation on rank 0 timed out, exiting...",
                flush=True,
            )
            sys.exit(1)
        fused_kernels.load(args.force_build_fused_kernels)

    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(
            ">>> done with compiling and loading fused kernels. "
            "Compilation time: {:.3f} seconds".format(time.time() - start_time),
            flush=True,
        )


def _init_autoresume():
    """Set autoresume start time."""
    autoresume = get_adlr_autoresume()
    if autoresume:
        torch.distributed.barrier()
        autoresume.init()
        torch.distributed.barrier()


def _set_random_seed(seed_, data_parallel_random_init=False):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:
            seed = seed + (10 * mpu.get_data_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            mpu.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed_))


def set_jit_fusion_options():
    """Set PyTorch JIT layer fusion options."""
    # flags required to enable jit fusion kernels
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])
    if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):
        # nvfuser
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
        torch._C._debug_set_autodiff_subgraph_inlining(False)
    else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)

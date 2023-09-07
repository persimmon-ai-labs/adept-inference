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

"""Communications utilities."""


from typing import Sequence, Union
import collections
import torch

from megatron import mpu
from megatron.model.utils import check_shapes


# TODO: use functions from megatron/p2p
def recv_from_prev_pipeline_rank_(recv_buffer=None):
    """Receive from previous pipeline stage and update the
    input buffer inplace."""
    if not mpu.is_pipeline_first_stage():
        assert recv_buffer is not None
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv, recv_buffer, mpu.get_pipeline_model_parallel_prev_rank()
        )
        reqs = torch.distributed.batch_isend_irecv([recv_prev_op])
        for req in reqs:
            req.wait()
        # To protect against race condition when using batch_isend_irecv().
        torch.cuda.synchronize()


# TODO: use functions from megatron/p2p
def send_to_next_pipeline_rank(tensor=None):
    """Send output to the next pipeline stage."""
    if not mpu.is_pipeline_last_stage():
        assert tensor is not None
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend, tensor, mpu.get_pipeline_model_parallel_next_rank()
        )
        reqs = torch.distributed.batch_isend_irecv([send_next_op])
        for req in reqs:
            req.wait()
        # To protect against race condition when using batch_isend_irecv().
        torch.cuda.synchronize()


def _is_cuda(tensor):
    """Check if a tensor is not none and is cuda."""
    assert tensor is not None
    assert tensor.is_cuda


def _is_cuda_contiguous(tensor):
    """Check if a tensor is not none, is cuda, and is contiguous."""
    _is_cuda(tensor)
    assert tensor.is_contiguous()


def _check_shapes(tensor: torch.Tensor, size: Union[int, Sequence[int]], dtype: torch.dtype):
    """Check that the sending and receiver tensors will be compatible."""
    # This logic is required because torch.empty will promote a size of int to [int] and there are lots
    # of callers that rely on this convention.
    shape = size if isinstance(size, collections.abc.Sequence) else [size]

    check_shapes(tensor, expected_shapes=shape, expected_ndim=len(shape), expected_dtype=dtype)


def broadcast_from_last_pipeline_stage(size, dtype, tensor=None):
    """Broadcast a tensor from last pipeline stage to all ranks."""

    is_last_stage = mpu.is_pipeline_last_stage(ignore_virtual=True)
    # If first stage and last state are the same, then there is no
    # pipeline parallelism and no need to communicate.
    if mpu.is_pipeline_first_stage() and is_last_stage:
        return tensor

    if is_last_stage:
        _is_cuda_contiguous(tensor)
        _check_shapes(tensor, size, dtype)
    else:
        tensor = torch.empty(size, dtype=dtype, device=torch.cuda.current_device())
    # Get the group and corresponding source rank.
    src = mpu.get_pipeline_model_parallel_last_rank()
    group = mpu.get_pipeline_model_parallel_group()
    torch.distributed.broadcast(tensor, src, group)

    return tensor


def broadcast_from_last_to_first_pipeline_stage(size, dtype, tensor=None):
    """Broadcast tensor values from last stage into the first stage."""

    is_last_stage = mpu.is_pipeline_last_stage()
    is_first_stage = mpu.is_pipeline_first_stage()
    # If first stage and last state are the same, then there is no
    # pipeline parallelism and no need to communicate.
    if is_first_stage and is_last_stage:
        return tensor
    # Only first and last stage pipeline stages need to be involved.
    if is_last_stage or is_first_stage:
        if is_last_stage:
            _is_cuda_contiguous(tensor)
            _check_shapes(tensor, size, dtype)
        else:
            tensor = torch.empty(size, dtype=dtype, device=torch.cuda.current_device())
        src = mpu.get_pipeline_model_parallel_last_rank()
        group = mpu.get_embedding_group()
        # Broadcast from last stage into the first stage.
        torch.distributed.broadcast(tensor, src, group)
    else:
        tensor = None

    return tensor


def copy_from_last_to_first_pipeline_stage(size, dtype, tensor=None):
    """Copy tensor values from last stage into the first stage.
    Note that the input tensor is updated in place."""

    is_last_stage = mpu.is_pipeline_last_stage()
    is_first_stage = mpu.is_pipeline_first_stage()
    # If first stage and last state are the same, then there is no
    # pipeline parallelism and no need to communicate.
    if is_first_stage and is_last_stage:
        return
    # Only first and last stage pipeline stages need to be involved.
    if is_last_stage or is_first_stage:
        _is_cuda(tensor)
        is_contiguous = tensor.is_contiguous()
        src = mpu.get_pipeline_model_parallel_last_rank()
        group = mpu.get_embedding_group()
        if is_contiguous:
            tensor_ = tensor
        else:
            if is_last_stage:
                tensor_ = tensor.contiguous()
            else:
                tensor_ = torch.empty(size, dtype=dtype, device=torch.cuda.current_device())
        # Broadcast from last stage into the first stage.
        torch.distributed.broadcast(tensor_, src, group)
        # Update the first stage tensor
        if is_first_stage and not is_contiguous:
            tensor[...] = tensor_


def broadcast_tensor(size, dtype, tensor=None, rank=0, device=0):
    """Given size and type of a tensor on all ranks and the tensor value
    only on a specific rank, broadcast from that rank to all other ranks.
    Args:
        size: size of the tensor
        dtype: type of the tensor
        tensor: tensor to be broadcasted
        rank: primary rank for broadcasting
        device: device of the tensor. If not set to None, then we use cuda.current_device().
            Default is 0, since we use cuda.current_device() to get the device.
    """
    if device is not None:
        device = torch.cuda.current_device()
    if torch.distributed.get_rank() == rank:
        if device is not None:
            _is_cuda_contiguous(tensor)
        _check_shapes(tensor, size, dtype)
    else:
        tensor = torch.empty(size, dtype=dtype, device=device)
    torch.distributed.broadcast(tensor, rank)
    return tensor


def broadcast_list(size, dtype, list_values=None, rank=0, device=0):
    """Broadcast a list of values with a given type.
    Args:
        size: size of the list
        dtype: dtype of the list
        list_values: list of values to be broadcasted
        rank: primary rank for broadcasting
        device: device of the tensor. If not set to None, then we use cuda.current_device().
            Default is 0, since we use cuda.current_device() to get the device.
    """
    tensor = None
    if device is not None:
        device = torch.cuda.current_device()
    if torch.distributed.get_rank() == rank:
        tensor = torch.tensor(list_values, dtype=dtype, device=device)
    return broadcast_tensor(size, dtype, tensor=tensor, rank=rank, device=device)


def broadcast_int_list(size, int_list=None, rank=0, device=0):
    """Broadcast a list of interger values.
    Args:
        size: size of the list
        int_list: list of values to be broadcasted
        rank: primary rank for broadcasting
        device: device of the tensor. If not set to None, then we use cuda.current_device().
            Default is 0, since we use cuda.current_device() to get the device.
    """
    if device is not None:
        device = torch.cuda.current_device()
    return broadcast_list(size, torch.int64, list_values=int_list, rank=rank, device=device)


def broadcast_float_list(size, float_list=None, rank=0, device=0):
    """Broadcast a list of float values.
    Args:
        size: size of the list
        float_list: list of values to be broadcasted
        rank: primary rank for broadcasting
        device: device of the tensor. If not set to None, then we use cuda.current_device().
            Default is 0, since we use cuda.current_device() to get the device.
    """
    if device is not None:
        device = torch.cuda.current_device()
    return broadcast_list(size, torch.float32, list_values=float_list, rank=rank, device=device)

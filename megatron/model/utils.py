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

"""Utilities for models."""

import math
import socket

import torch
from numpy.random import default_rng, SeedSequence
import multiprocessing
import concurrent.futures
import numpy as np
from typing import Sequence, List
import functools

from megatron import get_args

from megatron import mpu


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""

    def fill(rng: np.random.Generator, shape: Sequence[int]):
        n_threads = 16
        n_elements = np.prod(shape)
        values = np.empty(n_elements)
        assert n_elements % n_threads == 0, "Number of elements must be a multiple of number of threads!"
        step = np.ceil(n_elements / n_threads).astype(np.int_)

        # TODO(erich): hacky way to get new generator seeds, we should use spawn as soon as we have
        # numpy 1.25 and not this!!!
        seeds = rng.integers(0, 1 << 63, n_threads)

        _random_generators = [np.random.default_rng(s) for s in seeds]

        executor = concurrent.futures.ThreadPoolExecutor(n_threads)

        def _fill(generator, out, first, last):
            out[first:last] = generator.normal(loc=0.0, scale=sigma, size=step)

        futures = {}
        for i in range(n_threads):
            args = (_fill, _random_generators[i], values, i * step, (i + 1) * step)
            futures[executor.submit(*args)] = i
        concurrent.futures.wait(futures)
        return np.reshape(values, shape)

    return fill


# TODO(erich): combine these two functions to reduce code duplication
def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def fill(rng: np.random.Generator, shape: Sequence[int]):
        n_threads = 16
        n_elements = np.prod(shape)
        values = np.empty(n_elements)
        assert n_elements % n_threads == 0, "Number of elements must be a multiple of number of threads!"
        step = np.ceil(n_elements / n_threads).astype(np.int_)

        # TODO(erich): hacky way to get new generator seeds, we should use spawn as soon as we have
        # numpy 1.25 and not this!!!
        seeds = rng.integers(0, 1 << 63, n_threads)

        _random_generators = [np.random.default_rng(s) for s in seeds]
        executor = concurrent.futures.ThreadPoolExecutor(n_threads)

        def _fill(generator, out, first, last):
            out[first:last] = generator.normal(loc=0.0, scale=std, size=step)

        futures = {}
        for i in range(n_threads):
            args = (_fill, _random_generators[i], values, i * step, (i + 1) * step)
            futures[executor.submit(*args)] = i
        concurrent.futures.wait(futures)
        return np.reshape(values, shape)

    return fill


def check_shapes(tensor, expected_shapes=None, expected_ndim=None, expected_dtype=None):
    """Check that the passed-in `tensor` has the expected shape and ndim. Should an expected dim be None, it is not checked.

    Args:
        tensor: The tensor to shape-check.
        expected_shapes: The prefix of the shapes to expect in `tensor`.
        expected_ndim: The expected number of dimensions in `tensor`.
        expected_dtype: Expected dtype.
    """
    assert tensor is not None, "tensor is None"

    if expected_shapes is None and expected_ndim is None and expected_dtype is None:
        return

    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype

    if expected_ndim is not None:
        assert tensor.ndim == expected_ndim, f"Unexpected ndims detected. Got: {tensor.ndim} expected: {expected_ndim}"

    expected_shapes = list(expected_shapes or [])

    if expected_ndim is not None:
        assert (
            len(expected_shapes) <= tensor.ndim
        ), f"Asking to check too many shapes. Got: {expected_shapes} expected: <= {tensor.ndim}"

    t_shapes = list(tensor.shape[: len(expected_shapes)])
    for i, e in enumerate(expected_shapes):
        if e is None:
            t_shapes[i] = None
    assert t_shapes == expected_shapes, f"Mismatch detected. Got: {t_shapes} expected: {expected_shapes}"


def get_linear_layer(rows, columns, init_method):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    if get_args().perform_initialization:
        init_method(layer.weight)
    with torch.no_grad():
        layer.bias.zero_()
    return layer


@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


def openai_gelu(x):
    return gelu_impl(x)


# This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype) + torch.ones_like(x).to(dtype=x.dtype))


def print_named_parameters(model):
    """Print a summary of the parameters in a model."""
    prefix = ""
    if torch.distributed.is_initialized():
        # Print on only the first data parallel rank, but on all tensor/pipeline parallel ranks.
        should_print = mpu.get_data_parallel_rank() == 0
        if mpu.get_tensor_model_parallel_world_size() > 1:
            prefix = f"tensor-rank: {mpu.get_tensor_model_parallel_rank()} | "
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            prefix = f"pipeline-rank: {mpu.get_pipeline_model_parallel_rank()} | "
    else:
        should_print = get_args().rank == 0

    if not should_print:
        return

    print(f"{prefix} > {type(model).__name__} parameters: ", flush=True)
    for name, param in model.named_parameters():
        if mpu.param_is_tensor_parallel_unique(param):
            print(f"{prefix}{name=}, {param.shape=}, norm={torch.norm(param.data.float()).item()}", flush=True)


def sync_data_parallel_replicated_parameters(a, b):
    pass


def sync_tensor_parallel_replicated_parameters(a, b):
    pass


def _sync_replicated_parameters(a, b, c, d, e):

    def _sync(p):
        pass


class ReplicationMismatchError(Exception):
    pass


class TensorParallelReplicationMismatchError(ReplicationMismatchError):
    pass


class DataParallelReplicationMismatchError(ReplicationMismatchError):
    pass


def validate_replicated_parameters(m, o):

    def collect_and_validate(a, b, c, d, prefix):
        pass


def _validate_replicated_parameters(
    *, a, b, c, d, prefix=""
):
    pass

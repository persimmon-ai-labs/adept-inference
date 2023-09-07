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


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import numpy as np
import hashlib

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from megatron import get_args, get_global_memory_buffer

from .initialize import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from .mappings import (
    reduce_backward_from_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_forward_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from .utils import VocabUtility, divide

# tensor_model_parallel_unique: True if the parameter weights are unique on different tensor parallel shards.
#  Examples of the True case would be dense layers that are sharded across tensor parallel ranks.
#  Examples of the False case would be LayerNorm layers that are just replicated across tensor parallel ranks.
# reduce_tensor_model_parallel_gradients: True if the gradients should be reduced across tensor parallel ranks.
#  This happens in optimizer.py.
#  Examples of the True case would be shared QK LayerNorms that are replicated across tensor parallel ranks, but
#  operate on different activations and need to be kept in sync.
_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    "tensor_model_parallel_unique": False,
    "reduce_tensor_model_parallel_gradients": False,
}


def param_is_tensor_parallel_unique(param):
    """Return true if the parameter is sharded across tensor parallel ranks or is on rank 0.

    Typically used to determine if a given parameter's weights should count toward something like
    the total params norm. The goal is to avoid double counting parameters that have exact copies
    on multiple tensor parallel ranks.
    """
    return getattr(param, "tensor_model_parallel_unique", False) or get_tensor_model_parallel_rank() == 0


def param_is_tensor_parallel_replicated(param):
    """Retruns true if the parameter should be the same across all tensor parallel ranks."""
    return not getattr(param, "tensor_model_parallel_unique", False)


def set_tensor_model_parallel_attributes(tensor, *, is_tensor_parallel_unique, reduce_tensor_parallel_grads):
    if reduce_tensor_parallel_grads and is_tensor_parallel_unique:
        raise ValueError(f"Cannot set both is_tensor_parallel_unique and reduce_tensor_parallel_grads to True")
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, "tensor_model_parallel_unique", is_tensor_parallel_unique)
    setattr(tensor, "reduce_tensor_model_parallel_gradients", reduce_tensor_parallel_grads)
    # Make sure the attributes are all set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert hasattr(tensor, attribute)


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)

    for attribute, default in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS.items():
        maybe_set(attribute, default)


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute, getattr(source_tensor, attribute))

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def _get_parameter_placement() -> torch.device:
    """Figure out where to place a parameter."""
    args = get_args()
    if args.place_on_cpu:
        return torch.device("cpu")
    else:
        return torch.cuda.current_device()


USED_RNG_KEYS = set()


def get_rng_for_key(rng_key: str) -> np.random.Generator:
    """Get a numpy random number generator for a given key."""
    if rng_key in USED_RNG_KEYS:
        raise ValueError(f"RNG key {rng_key} already used")
    USED_RNG_KEYS.add(rng_key)

    args = get_args()
    msg = hashlib.sha256(str(args.seed).encode("utf-8"))
    msg.update(rng_key.encode("utf-8"))
    seed = list(msg.digest())

    return np.random.default_rng(seed=seed)


def _initialize_affine_weight(
    weight,
    output_size,
    input_size,
    per_partition_size,
    partition_dim,
    init_method,
    rng_key: str,
    perform_initialization: bool,
    return_master_weight=False,
):
    """Initialize affine weight for model parallel.

    Build the master weight on CPU using numpy. This saves GPU memory and is fully deterministic.
    Then select the relevant chunk (for tensor parallelism) and move to GPU.

    Arguments:
        weight: weight tensor to initialize.
        output_size: first dimension of weight tensor.
        input_size: second dimension of weight tensor.
        per_partition_size: size of the partitioned dimension.
        partition_dim: dimension to partition along.
        init_method: weight initialization function.
        rng_key: a unique string for the weight to be initialized. Should be the same regardless of tensor or
            pipeline parallelism.
        perform_initialization: if False, skip initialization.
        return_master_weight: if True, return the master weight.
    """

    set_tensor_model_parallel_attributes(
        tensor=weight, is_tensor_parallel_unique=True, reduce_tensor_parallel_grads=False
    )

    if not perform_initialization:
        return None

    # Initialize master weight on CPU to save GPU memory and use fully deterministic numpy.
    rng = get_rng_for_key(rng_key)
    master_weight = torch.tensor(
        init_method(rng=rng, shape=[output_size, input_size]), dtype=torch.float, requires_grad=False
    )
    args = get_args()

    # Select the relevant chunk and move to GPU.
    weight_list = torch.split(master_weight, per_partition_size, dim=partition_dim)
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        combined_weight = torch.cat(my_weight_list, dim=partition_dim)
        combined_weight = combined_weight.to(dtype=args.params_dtype)
        weight.copy_(combined_weight)
    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim, init_method, init_rng_key: str):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the defaults for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocabulary dimension.
        self.vocab_start_index, self.vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
            self.num_embeddings, get_tensor_model_parallel_rank(), self.tensor_model_parallel_size
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        # Allocate weights and initialize.
        args = get_args()
        self.weight = Parameter(
            torch.empty(
                self.num_embeddings_per_partition,
                self.embedding_dim,
                device=_get_parameter_placement(),
                dtype=args.params_dtype,
            )
        )
        _initialize_affine_weight(
            weight=self.weight,
            output_size=self.num_embeddings,
            input_size=self.embedding_dim,
            per_partition_size=self.num_embeddings_per_partition,
            partition_dim=0,
            init_method=init_method,
            rng_key=init_rng_key,
            perform_initialization=args.perform_initialization,
        )

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_forward_from_tensor_model_parallel_region(output_parallel)
        return output


class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """
    Linear layer execution with asynchronous communication and gradient accumulation
    fusion in backprop.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, gradient_accumulation_fusion, async_grad_allreduce, sequence_parallel):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel = sequence_parallel

        if sequence_parallel:
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            torch.distributed._all_gather_base(all_gather_buffer, input, group=get_tensor_model_parallel_group())
            total_input = all_gather_buffer
        else:
            total_input = input

        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias

        if ctx.sequence_parallel:
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            handle = torch.distributed._all_gather_base(
                all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True
            )

            # Delay the start of intput gradient computation shortly (3us) to have
            # gather scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1
            total_input = all_gather_buffer
        else:
            total_input = input
        grad_input = grad_output.matmul(weight)

        if ctx.sequence_parallel:
            handle.wait()

        # Convert the tensor shapes to 2D for execution compatibility
        # Necessary when using cross-attention and the flash attention module.
        grad_output = grad_output.contiguous()
        grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2])
        total_input = total_input.view(total_input.shape[0] * total_input.shape[1], total_input.shape[2])

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(grad_input, group=get_tensor_model_parallel_group(), async_op=True)
            # Delay the start of weight gradient computation shortly (3us) to have
            # all-reduce scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1

        if ctx.sequence_parallel:
            assert not ctx.async_grad_allreduce
            dim_size = list(input.size())
            sub_grad_input = torch.empty(
                dim_size, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False
            )
            # reduce_scatter
            handle = torch.distributed._reduce_scatter_base(
                sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=True
            )
            # Delay the start of weight gradient computation shortly (3us) to have
            # reduce scatter scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1

        if ctx.gradient_accumulation_fusion:
            from megatron_fused_kernels import fused_dense_cuda

            fused_dense_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, weight.main_grad)
            grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel:
            handle.wait()
            return sub_grad_input, grad_weight, grad_bias, None, None, None

        if ctx.async_grad_allreduce:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        init_method,
        init_rng_key: str,
        sequence_parallel: bool,
        bias=True,
        gather_output=True,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
    ):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        self.weight = Parameter(
            torch.empty(
                self.output_size_per_partition,
                self.input_size,
                device=_get_parameter_placement(),
                dtype=args.params_dtype,
            )
        )
        _initialize_affine_weight(
            weight=self.weight,
            output_size=self.output_size,
            input_size=self.input_size,
            per_partition_size=self.output_size_per_partition,
            partition_dim=0,
            init_method=init_method,
            rng_key=init_rng_key,
            perform_initialization=args.perform_initialization,
            return_master_weight=keep_master_weight_for_test,
        )
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition, device=_get_parameter_placement(), dtype=args.params_dtype)
            )
            set_tensor_model_parallel_attributes(
                self.bias, is_tensor_parallel_unique=True, reduce_tensor_parallel_grads=False
            )
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)
        self.async_tensor_model_parallel_allreduce = args.async_tensor_model_parallel_allreduce and world_size > 1
        self.sequence_parallel = sequence_parallel
        assert not self.async_tensor_model_parallel_allreduce or not self.sequence_parallel
        self.gradient_accumulation_fusion = args.gradient_accumulation_fusion

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        if self.async_tensor_model_parallel_allreduce or self.sequence_parallel:
            input_parallel = input_
        else:
            input_parallel = reduce_backward_from_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = LinearWithGradAccumulationAndAsyncCommunication.apply(
            input_parallel,
            self.weight,
            bias,
            self.gradient_accumulation_fusion,
            self.async_tensor_model_parallel_allreduce,
            self.sequence_parallel,
        )
        if self.gather_output:
            assert not self.sequence_parallel
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class ParallelLinear(ColumnParallelLinear):
    """Drop-in replacement for torch.nn.Linear, with tensor parallelism support."""

    def __init__(
        self,
        input_size,
        output_size,
        *,
        init_method,
        init_rng_key: str,
        sequence_parallel: bool,
        bias=True,
    ):
        super(ParallelLinear, self).__init__(
            input_size,
            output_size,
            bias=bias,
            init_method=init_method,
            init_rng_key=init_rng_key,
            gather_output=True,
            sequence_parallel=sequence_parallel,
        )

    def forward(self, input_):
        # normal linear layer doesn't return tuples, make this behave the same way
        output, _ = super().forward(input_)
        return output


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        init_method,
        init_rng_key: str,
        sequence_parallel: bool,
        bias=True,
        input_is_parallel=False,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
    ):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        self.weight = Parameter(
            torch.empty(
                self.output_size,
                self.input_size_per_partition,
                device=_get_parameter_placement(),
                dtype=args.params_dtype,
            )
        )
        _initialize_affine_weight(
            weight=self.weight,
            output_size=self.output_size,
            input_size=self.input_size,
            per_partition_size=self.input_size_per_partition,
            partition_dim=1,
            init_method=init_method,
            rng_key=init_rng_key,
            perform_initialization=args.perform_initialization,
            return_master_weight=keep_master_weight_for_test,
        )
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, device=_get_parameter_placement(), dtype=args.params_dtype)
            )
            setattr(self.bias, "sequence_parallel", sequence_parallel)

            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)
        self.sequence_parallel = sequence_parallel
        self.gradient_accumulation_fusion = args.gradient_accumulation_fusion

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = LinearWithGradAccumulationAndAsyncCommunication.apply(
            input_parallel, self.weight, None, self.gradient_accumulation_fusion, None, None
        )
        # All-reduce across all the partitions.
        if self.sequence_parallel:
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output_ = reduce_forward_from_tensor_model_parallel_region(output_parallel)
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias


def argmax_parallel(input_: torch.Tensor) -> torch.Tensor:
    """Argmax that supports tensor parallel inputs. Assumes dim=-1. Reverts to local argmax when not using parallelism."""
    local_max, local_index = torch.max(input_, dim=-1, keepdim=True)
    local_index += input_.shape[-1] * get_tensor_model_parallel_rank()

    gathered_max = gather_from_tensor_model_parallel_region(local_max)
    gathered_index = gather_from_tensor_model_parallel_region(local_index)

    global_argmax = torch.take_along_dim(
        gathered_index, torch.argmax(gathered_max, dim=-1, keepdim=True), dim=-1
    ).squeeze(-1)
    assert global_argmax.shape == input_.shape[:-1]
    return global_argmax

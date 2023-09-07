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

"""Transformer."""
from typing import Tuple

import math
from contextlib import nullcontext
import dataclasses
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron import get_args, get_global_memory_buffer
from megatron.config import Config, DISABLE_PREFIX
from megatron import mpu
from .module import MegatronModule
from megatron.model.enums import AttnMaskType, ModelType, LayerType, AttnType
from megatron.model import LayerNorm
from megatron.model.fused_softmax import FusedScaleSoftmax
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.fused_bias_sqrelu import bias_sqrelu_impl
from megatron.model.positional_embeddings import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_torch,
)
from megatron.model.utils import openai_gelu, erf_gelu

try:
    from einops import rearrange, repeat
except ImportError:
    rearrange, repeat = None, None

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func
    from flash_attn.layers.rotary import RotaryEmbedding as RotaryEmbeddingFlash
except ImportError:
    flash_attn_varlen_qkvpacked_func = None
    flash_attn_varlen_kvpacked_func = None
    RotaryEmbeddingFlash = None

# FT's single-query attention kernel from flash-attention repo
try:
    import ft_attention
except ImportError:
    ft_attention = None


""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""


class DropPath(MegatronModule):
    """Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_state):
        if self.drop_prob == 0.0 or not self.training:
            return hidden_state
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (hidden_state.shape[0],) + (1,) * (hidden_state.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=hidden_state.dtype, device=hidden_state.device)
        random_tensor.floor_()  # binarize
        output = hidden_state.div(keep_prob) * random_tensor
        return output


@dataclasses.dataclass
class ParallelMLPConfig(Config):
    bias_gelu_fusion: bool
    ffn_hidden_size: int
    hidden_size: int
    onnx_safe: bool = DISABLE_PREFIX()
    openai_gelu: bool = DISABLE_PREFIX()
    sequence_parallel: bool = DISABLE_PREFIX()
    sq_relu: bool


class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(
        self,
        parallel_mlp_config: ParallelMLPConfig,
        init_method,
        output_layer_init_method,
        init_rng_key: str,
    ):
        super(ParallelMLP, self).__init__()

        self.sequence_parallel = parallel_mlp_config.sequence_parallel

        # Project to 4h.
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            parallel_mlp_config.hidden_size,
            parallel_mlp_config.ffn_hidden_size,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            init_rng_key=f"{init_rng_key}_dense_h_to_4h",
            sequence_parallel=self.sequence_parallel,
        )

        self.bias_gelu_fusion = parallel_mlp_config.bias_gelu_fusion
        self.bias_sqrelu_fusion = parallel_mlp_config.sq_relu
        self.activation_func = F.gelu
        if parallel_mlp_config.openai_gelu:
            self.activation_func = openai_gelu
        elif parallel_mlp_config.onnx_safe:
            self.activation_func = erf_gelu

        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            parallel_mlp_config.ffn_hidden_size,
            parallel_mlp_config.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            init_rng_key=f"{init_rng_key}_dense_4h_to_h",
            sequence_parallel=self.sequence_parallel,
        )

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
            intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
        elif self.bias_sqrelu_fusion:
            intermediate_parallel = bias_sqrelu_impl(intermediate_parallel, bias_parallel)
        else:
            intermediate_parallel = self.activation_func(intermediate_parallel + bias_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


@dataclasses.dataclass
class CoreAttentionConfig(Config):
    apply_query_key_layer_scaling: bool = DISABLE_PREFIX()
    attention_dropout: float
    attention_softmax_in_fp32: bool = DISABLE_PREFIX()
    bf16: bool = DISABLE_PREFIX()
    fp16: bool = DISABLE_PREFIX()
    kv_channels: int
    masked_softmax_fusion: bool = DISABLE_PREFIX()
    num_attention_heads: int
    pos_emb: str
    sequence_parallel: bool = DISABLE_PREFIX()
    tensor_model_parallel_size: int = DISABLE_PREFIX()


class CoreAttention(MegatronModule):
    def __init__(
        self,
        core_attention_config: CoreAttentionConfig,
        layer_number,
        attn_mask_type=AttnMaskType.padding,
    ):
        super(CoreAttention, self).__init__()
        self.fp16 = core_attention_config.fp16
        self.bf16 = core_attention_config.bf16

        self.apply_query_key_layer_scaling = core_attention_config.apply_query_key_layer_scaling
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = core_attention_config.sequence_parallel

        projection_size = core_attention_config.kv_channels * core_attention_config.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(projection_size, world_size)
        self.hidden_size_per_attention_head = mpu.divide(projection_size, core_attention_config.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(core_attention_config.num_attention_heads, world_size)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleSoftmax(
            self.fp16,
            self.bf16,
            self.attn_mask_type,
            core_attention_config.masked_softmax_fusion,
            self.attention_softmax_in_fp32,
            coeff,
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(core_attention_config.attention_dropout)

    def compute_attention_scores(self, query_layer, key_layer, matmul_result, output_size, transpose_sb=False):
        """Compute attention probabilities.

        Args:
            query_layer (torch.Tensor): query tensor
            key_layer (torch.Tensor): key tensor
            matmul_result (torch.Tensor): matmul result
            output_size (tuple): output size
            transpose_sb (bool): transpose the first two dimensions of q and k

        Returns:
            torch.Tensor: attention probabilities
        """
        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1) if transpose_sb else query_layer,
            key_layer.transpose(0, 1).transpose(1, 2) if transpose_sb else key_layer.transpose(1, 2),
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.sequence_parallel:
            with mpu.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)
        return attention_probs

    def forward(self, query_layer, key_layer, value_layer):

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================
        # [b, np, sq, sk]
        output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]), query_layer.dtype, "mpu"
        )

        attention_probs = self.compute_attention_scores(
            query_layer, key_layer, matmul_input_buffer, output_size, transpose_sb=True
        )

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class FlashSelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        assert flash_attn_varlen_qkvpacked_func is not None, "FlashAttention is not installed"
        assert flash_attn_varlen_kvpacked_func is not None, "FlashAttention is not installed"
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv=None, kv=None, q=None):
        """Implements the multihead softmax attention.
        In the case of self-attention, the qkv matrix must be provided.
        In the case of cross-attention, however, the kv and q matrices must
            be provided separately.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            kv: The tensor containing the key and value. (B, S, 2, H, D)
            q: The tensor containing the query. (B, S, H, D)
        """
        if qkv is not None:
            assert kv is None and q is None
            return self._self_attn(qkv)
        else:
            assert kv is not None and q is not None
            return self._cross_attn(kv, q)

    def _cross_attn(self, kv, q):
        assert kv.dtype in [torch.float16, torch.bfloat16] and q.dtype in [torch.float16, torch.bfloat16]
        assert kv.is_cuda and q.is_cuda

        batch_size, seqlen_kv = kv.shape[0], kv.shape[1]
        kv = rearrange(kv, "b s ... -> (b s) ...")

        seqlen_q = q.shape[1]
        q = rearrange(q, "b s ... -> (b s) ...")

        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device)
        cu_seqlens_kv = torch.arange(
            0, (batch_size + 1) * seqlen_kv, step=seqlen_kv, dtype=torch.int32, device=kv.device
        )
        output = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_seqlens_q,
            cu_seqlens_kv,
            seqlen_q,
            seqlen_kv,
            self.dropout_p if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
        output = rearrange(output, "(b s) ... -> b s ...", b=batch_size)
        return output

    def _self_attn(self, qkv):
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        qkv = rearrange(qkv, "b s ... -> (b s) ...")
        max_s = seqlen
        cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=qkv.device)
        output = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_s,
            self.dropout_p if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
        output = rearrange(output, "(b s) ... -> b s ...", b=batch_size)
        return output


@dataclasses.dataclass
class ParallelAttentionConfig(Config):
    attention_dropout: float
    bf16: bool = DISABLE_PREFIX()
    core_attention_config: CoreAttentionConfig
    fp16: bool = DISABLE_PREFIX()
    hidden_size: int
    kv_channels: int
    num_attention_heads: int
    params_dtype: torch.dtype = DISABLE_PREFIX()
    pos_emb: str
    recompute_granularity: str = DISABLE_PREFIX()
    rotary_emb_base: int
    enable_rotary_base_check: bool = DISABLE_PREFIX()
    rotary_pct: float
    use_flash_attn: bool = DISABLE_PREFIX()
    qk_layernorm: str = DISABLE_PREFIX()
    layernorm_epsilon: float = DISABLE_PREFIX()
    no_persist_layer_norm: bool = DISABLE_PREFIX()
    sequence_parallel: bool = DISABLE_PREFIX()


class LayerNormPerAttentionHead(nn.Module):
    def __init__(self, layernorm_builder, num_attention_heads_per_partition):
        super().__init__()
        self.num_attention_heads_per_partition = num_attention_heads_per_partition
        self.layernorms = torch.nn.ModuleList(
            [layernorm_builder() for _ in range(self.num_attention_heads_per_partition)]
        )

    def forward(self, layer_input):
        layer_heads = torch.split(layer_input, 1, dim=2)
        assert (
            len(layer_heads) == len(self.layernorms) == self.num_attention_heads_per_partition
        ), f"{len(layer_heads)}, {len(self.layernorms)}, {self.num_attention_heads_per_partition}"
        normed_heads = [ln(ln_input) for ln, ln_input in zip(self.layernorms, layer_heads)]
        return torch.cat(normed_heads, dim=2)


class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        parallel_attention_config: ParallelAttentionConfig,
        init_method,
        output_layer_init_method,
        layer_number,
        module_name: str,
        attention_type=AttnType.self_attn,
        attn_mask_type=AttnMaskType.padding,
    ):
        super(ParallelAttention, self).__init__()
        self.module_name = module_name
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = parallel_attention_config.params_dtype
        # Set positional embeddings
        self.pos_emb = parallel_attention_config.pos_emb
        self.rotary_emb_base = parallel_attention_config.rotary_emb_base
        self.enable_rotary_base_check = parallel_attention_config.enable_rotary_base_check
        self.qk_layernorm = parallel_attention_config.qk_layernorm
        self.fp16 = parallel_attention_config.fp16
        self.bf16 = parallel_attention_config.bf16
        self.use_flash_attn = parallel_attention_config.use_flash_attn
        if self.use_flash_attn:
            assert flash_attn_varlen_qkvpacked_func is not None, "FlashAttention is not installed"
            assert flash_attn_varlen_kvpacked_func is not None, "FlashAttention is not installed"
            if rearrange is None:
                raise ImportError("einops is not installed")

        if parallel_attention_config.pos_emb == "rotary" and attention_type == AttnType.cross_attn:
            raise AssertionError("Can't use rotary embeddings and cross attention.")

        projection_size = parallel_attention_config.kv_channels * parallel_attention_config.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = mpu.divide(projection_size, parallel_attention_config.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(parallel_attention_config.num_attention_heads, world_size)

        self.sequence_parallel = parallel_attention_config.sequence_parallel

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = mpu.ColumnParallelLinear(
                parallel_attention_config.hidden_size,
                3 * projection_size,
                gather_output=False,
                init_method=init_method,
                init_rng_key=f"{self.module_name}_layer_{layer_number}_query_key_value",
                sequence_parallel=self.sequence_parallel,
            )
        else:
            assert attention_type == AttnType.cross_attn
            self.query = mpu.ColumnParallelLinear(
                parallel_attention_config.hidden_size,
                projection_size,
                gather_output=False,
                init_method=init_method,
                init_rng_key=f"{self.module_name}_layer_{layer_number}_query",
                sequence_parallel=self.sequence_parallel,
            )

            self.key_value = mpu.ColumnParallelLinear(
                parallel_attention_config.hidden_size,
                2 * projection_size,
                gather_output=False,
                init_method=init_method,
                init_rng_key=f"{self.module_name}_layer_{layer_number}_key_value",
                sequence_parallel=self.sequence_parallel,
            )

        # See "Scaling Vision Transformers to 22 Billion Parameters" https://arxiv.org/abs/2302.05442
        if self.qk_layernorm != "none":

            def qk_layernorm_builder():
                # Modes available are:
                # weights_per_layer: QK LayerNorm weights are the same for every attention head on a given layer.
                # weights_per_tensor_parallel_rank_and_layer: QK LayerNorm weights are the same for every attention
                #   head on a given tensor parallel rank. This version was implemented by accident and used in the 4B
                #   flagship model. It is preserved for backwards compatibility, but should probably not be used going
                #   forward.
                # weights_per_attention_head_and_layer: QK LayerNorm weights are unique per attention head.

                # Heads are sharded by tensor parallel ranks. If we're using unique weights per head or tensor
                # parallel rank, then the weights for these LayerNorms will be unique per tensor parallel rank and
                # are marked is_tensor_parallel_unique.
                # Otherwise, they will be shared (via reduce_tensor_parallel_grads) and should not be counted
                # multiple times when calculating params norm for logging or gradient clipping.
                if self.qk_layernorm == "weights_per_layer":
                    is_tensor_parallel_unique = False
                    reduce_tensor_parallel_grads = True
                elif self.qk_layernorm == "weights_per_tensor_parallel_rank_and_layer":
                    is_tensor_parallel_unique = True
                    reduce_tensor_parallel_grads = False
                elif self.qk_layernorm == "weights_per_attention_head_and_layer":
                    is_tensor_parallel_unique = True
                    reduce_tensor_parallel_grads = False
                else:
                    raise ValueError(f"Unknown qk_layernorm mode: {self.qk_layernorm}")

                return LayerNorm(
                    self.hidden_size_per_attention_head,
                    eps=parallel_attention_config.layernorm_epsilon,
                    no_persist_layer_norm=parallel_attention_config.no_persist_layer_norm,
                    sequence_parallel=False,
                    is_tensor_parallel_unique=is_tensor_parallel_unique,
                    reduce_tensor_parallel_grads=reduce_tensor_parallel_grads,
                )

            if self.qk_layernorm == "weights_per_attention_head_and_layer":
                self.q_layernorm = LayerNormPerAttentionHead(
                    layernorm_builder=qk_layernorm_builder,
                    num_attention_heads_per_partition=self.num_attention_heads_per_partition,
                )
                self.k_layernorm = LayerNormPerAttentionHead(
                    layernorm_builder=qk_layernorm_builder,
                    num_attention_heads_per_partition=self.num_attention_heads_per_partition,
                )
            elif self.qk_layernorm in ("weights_per_layer", "weights_per_tensor_parallel_rank_and_layer"):
                self.q_layernorm = qk_layernorm_builder()
                self.k_layernorm = qk_layernorm_builder()
            else:
                raise ValueError(f"Unknown qk_layernorm mode: {self.qk_layernorm}")
        else:
            self.q_layernorm = None
            self.k_layernorm = None

        self.core_attention = CoreAttention(
            parallel_attention_config.core_attention_config,
            self.layer_number,
            self.attn_mask_type,
        )
        self.checkpoint_core_attention = parallel_attention_config.recompute_granularity == "selective"

        if self.use_flash_attn:
            is_causal = self.attn_mask_type == AttnMaskType.causal
            self.core_attention_flash = FlashSelfAttention(
                causal=is_causal, attention_dropout=parallel_attention_config.attention_dropout
            )
            if parallel_attention_config.pos_emb == "rotary":
                self.rotary_ndims = int(parallel_attention_config.rotary_pct * self.hidden_size_per_attention_head)
                self.rotary_emb = RotaryEmbeddingFlash(
                    self.rotary_ndims,
                    base=parallel_attention_config.rotary_emb_base,
                    pos_idx_in_fp32=False,  # For backward compatibility with models already trained
                )

        # Output.
        self.dense = mpu.RowParallelLinear(
            projection_size,
            parallel_attention_config.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            init_rng_key=f"{self.module_name}_layer_{layer_number}_dense",
            sequence_parallel=self.sequence_parallel,
        )

        if parallel_attention_config.pos_emb == "rotary" and not self.use_flash_attn:
            if parallel_attention_config.rotary_pct == 1:
                self.rotary_ndims = None
            else:
                assert parallel_attention_config.rotary_pct < 1
                self.rotary_ndims = int(self.hidden_size_per_attention_head * parallel_attention_config.rotary_pct)
            rotary_dim = self.rotary_ndims if self.rotary_ndims is not None else self.hidden_size_per_attention_head
            self.rotary_emb = RotaryEmbedding(
                rotary_dim,
                base=parallel_attention_config.rotary_emb_base,
                precision=parallel_attention_config.params_dtype,
            )

    def _checkpointed_attention_forward(self, query_layer, key_layer, value_layer):
        """Forward method with activation checkpointing."""

        def custom_forward(*inputs):
            query_layer = inputs[0]
            key_layer = inputs[1]
            value_layer = inputs[2]
            output_ = self.core_attention(query_layer, key_layer, value_layer)
            return output_

        hidden_states = mpu.checkpoint(custom_forward, False, query_layer, key_layer, value_layer)

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_len, batch_size):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            dtype=self.params_dtype,
            device=torch.cuda.current_device(),
        )

    def _check_rotary_base(self, sequence_len: int) -> None:
        if not self.enable_rotary_base_check:
            return
        min_rotary_emb_base = math.ceil((2 * sequence_len) / (math.pi))
        if min_rotary_emb_base > self.rotary_emb_base:
            raise ValueError(
                f"To support sequence length {sequence_len}, a minimum rotary base of "
                f"{min_rotary_emb_base} is required, but current base is {self.rotary_emb_base}"
            )

    def _apply_rotary_embedding(self, query_layer, key_layer, offset=0):
        """
        query_layer, key_layer: (seqlen, batch, nheads, headdim)
        """
        if self.rotary_ndims is not None:
            # partial rotary
            query_rot, query_pass = (
                query_layer[..., : self.rotary_ndims],
                query_layer[..., self.rotary_ndims :],
            )
            key_rot, key_pass = (
                key_layer[..., : self.rotary_ndims],
                key_layer[..., self.rotary_ndims :],
            )
        else:
            # full rotary
            query_rot, key_rot = query_layer, key_layer
        apply_rotary_fn = apply_rotary_pos_emb_torch if self.bf16 else apply_rotary_pos_emb

        # For inference, add the offset value to sequence length
        sequence_len = key_layer.shape[0] + offset
        # Apply the rotary embeddings
        cos, sin = self.rotary_emb(key_layer, seq_len=sequence_len)
        query_layer, key_layer = apply_rotary_fn(query_rot, key_rot, cos, sin, offset=offset)

        # query [sq, b, np, hn], key [sk, b, np, hn]
        if self.rotary_ndims is not None:
            query_layer = torch.cat((query_layer, query_pass), dim=-1)
            key_layer = torch.cat((key_layer, key_pass), dim=-1)
        return query_layer, key_layer

    def _update_kv_cache(self, inference_params, key_layer: Tensor, value_layer: Tensor) -> Tuple[Tensor]:
        """k, v: (seqlen, batch_size, nheads, head_dim)"""
        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        if self.layer_number not in inference_params.key_value_memory_dict:
            inf_max_seq_len = inference_params.max_sequence_len
            inf_max_batch_size = inference_params.max_batch_size
            inference_key_memory = self._allocate_memory(inf_max_seq_len, inf_max_batch_size)
            inference_value_memory = self._allocate_memory(inf_max_seq_len, inf_max_batch_size)
            inference_params.key_value_memory_dict[self.layer_number] = (
                inference_key_memory,
                inference_value_memory,
            )
            preallocated = False
        else:
            # It's possible that we allocate kv_cache larger than necessary, we'll slice to
            # take a subset of the right batch size and seqlen layer
            # If the memory is pre-allocated, and we're using FT's fused kernel,
            # then k_cache has shape (b h d/packsize seqlen packsize) where packsize = 4 or 8 (fp32 vs fp16,bf16),
            # and v_cache has shape (b h s d).
            inference_key_memory, inference_value_memory = inference_params.key_value_memory_dict[self.layer_number]
            preallocated = True
        # ==================================
        # Adjust key and value for inference
        # ==================================
        seqlen_dim_k = 0 if not (inference_params.fused_ft_kernel and preallocated) else 3
        batch_dim = 1 if not (inference_params.fused_ft_kernel and preallocated) else 0
        batch_start = inference_params.batch_size_offset
        batch_end = batch_start + key_layer.size(1)
        sequence_start = inference_params.sequence_len_offset
        sequence_end = sequence_start + key_layer.size(0)
        assert batch_end <= inference_key_memory.size(batch_dim)
        assert sequence_end <= inference_key_memory.size(seqlen_dim_k)
        # Copy key and values.
        if not inference_params.fused_ft_kernel:
            inference_key_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = key_layer
            inference_value_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = value_layer
            key_layer = inference_key_memory[:sequence_end, batch_start:batch_end, ...]
            value_layer = inference_value_memory[:sequence_end, batch_start:batch_end, ...]
            return key_layer, value_layer
        else:
            assert ft_attention is not None
            assert rearrange is not None
            assert inference_params.sequence_len_offset == 0
            # FT kernel requires different layouts for the k_cache and v_cache.
            assert inference_key_memory.dtype in [torch.float16, torch.bfloat16, torch.float32]
            assert inference_value_memory.dtype == inference_key_memory.dtype
            packsize = 4 if inference_key_memory.dtype == torch.float32 else 8
            if not preallocated:
                inference_key_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = key_layer
                inference_value_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = value_layer
                inference_key_memory = rearrange(
                    inference_key_memory, "s b h (d packsize) -> b h d s packsize", packsize=packsize
                ).contiguous()
                inference_value_memory = rearrange(inference_value_memory, "s b h d -> b h s d").contiguous()
                inference_params.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory,
                    inference_value_memory,
                )
            else:
                inference_key_memory[batch_start:batch_end, :, :, sequence_start:sequence_end, :] = rearrange(
                    key_layer, "s b h (d packsize) -> b h d s packsize", packsize=packsize
                )
                inference_value_memory[batch_start:batch_end, :, sequence_start:sequence_end, :] = rearrange(
                    value_layer, "s b h d -> b h s d"
                )
            return key_layer, value_layer

    def forward(self, hidden_states, encoder_output=None, inference_params=None):
        """
        Wrt to inference_params, there are 4 code paths:
        1. inference_params=None: training, we don't touch inference_params at all.
        2. inference_params is not None, inference_params.sequence_len_offset == 0: prompt processing.
            We do the same ops as during training, except we store the key_layer and value_layer to the kv cache
            in inference_params.
        3. inference_params is not None, inference_params.sequence_len_offset > 0: iterative decoding.
            3a. inference_params.fused_ft_kernel == False: we do QKV projection, rotary embedding,
                add k and v to the kv_cache, then load the kv_cache from inference_params, do single-query attention,
                then do output projection.
            3b. inference_params.fused_ft_kernel == True: we do QKV projection (but no rotary embedding),
                load the kv_cache from inference_params, then call ft_attention (which fuses rotary embedding,
                adding k and v to the kv_cache, and single-query attention), then do output projection.
        """
        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3)

            if self.q_layernorm:
                assert self.k_layernorm
                query_layer = self.q_layernorm(query_layer)
                key_layer = self.k_layernorm(key_layer)

                if self.use_flash_attn:
                    mixed_x_layer = torch.cat([query_layer, key_layer, value_layer], dim=-1)
                    assert mixed_x_layer.shape == new_tensor_shape

            if self.use_flash_attn:  # FlashAttention needs to reshape things
                qkv = rearrange(mixed_x_layer, "s b h (three d) -> b s three h d", three=3).contiguous()
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                2 * self.hidden_size_per_attention_head,
            )
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer, value_layer) = mpu.split_tensor_along_last_dim(mixed_kv_layer, 2)

            assert not self.q_layernorm and not self.k_layernorm, "qk_layernorm not yet implemented for cross-attention"

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            query_layer = query_layer.view(*new_tensor_shape)
            if self.use_flash_attn:  # FlashAttention needs to reshape things
                kv = rearrange(mixed_kv_layer, "s b h (two d) -> b s two h d", two=2).contiguous()
                q = rearrange(query_layer, "s b h d -> b s h d").contiguous()

        if self.pos_emb == "rotary":
            if (
                inference_params is None
                or inference_params.sequence_len_offset == 0
                or not inference_params.fused_ft_kernel
            ):  # Code path 1, 2, & 3a above
                offset = 0 if inference_params is None else inference_params.sequence_len_offset
                sequence_len = key_layer.shape[0] + offset
                self._check_rotary_base(sequence_len)
                if not self.use_flash_attn:
                    query_layer, key_layer = self._apply_rotary_embedding(
                        query_layer=query_layer, key_layer=key_layer, offset=offset
                    )
                else:
                    assert self.attention_type == AttnType.self_attn
                    qkv = self.rotary_emb(qkv, seqlen_offset=offset)
                    # For inference_params
                    # TD [2023-03-28]: We need to assign to query_layer here, otherwise later on when
                    # we use query_layer, we would be using the "unrotated" query_layer.
                    query_layer = rearrange(qkv[:, :, 0], "b s h d -> s b h d")
                    key_layer = rearrange(qkv[:, :, 1], "b s h d -> s b h d")
                    value_layer = rearrange(qkv[:, :, 2], "b s h d -> s b h d")

        # Code path 2 & 3a
        if inference_params is not None and (
            inference_params.sequence_len_offset == 0 or not inference_params.fused_ft_kernel
        ):
            key_layer, value_layer = self._update_kv_cache(inference_params, key_layer, value_layer)

        # ==================================
        # core attention computation
        # ==================================

        if (
            inference_params is None
            or inference_params.sequence_len_offset == 0
            or not inference_params.fused_ft_kernel
        ):  # Code path 1, 2, & 3a above
            # For iterative decoding (code path 3a), we don't use FlashAttention (since it has causal=True and
            # will mask out things)
            if not self.use_flash_attn or (inference_params is not None and inference_params.sequence_len_offset > 0):
                if self.checkpoint_core_attention:
                    context_layer = self._checkpointed_attention_forward(query_layer, key_layer, value_layer)
                else:
                    context_layer = self.core_attention(query_layer, key_layer, value_layer)
            else:
                if self.attention_type == AttnType.self_attn:
                    context_layer = rearrange(self.core_attention_flash(qkv=qkv), "b s h d -> s b (h d)").contiguous()
                else:
                    context_layer = rearrange(
                        self.core_attention_flash(kv=kv, q=q), "b s h d -> s b (h d)"
                    ).contiguous()
        else:
            # Code path 3b
            assert ft_attention is not None
            assert query_layer.shape[0] == key_layer.shape[0] == value_layer.shape[0] == 1
            q, k, v = [rearrange(x, "1 b h d -> b h d").contiguous() for x in [query_layer, key_layer, value_layer]]
            k_cache, v_cache = inference_params.key_value_memory_dict[self.layer_number]
            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + q.size(0)
            assert batch_end <= k_cache.size(0)
            k_cache = k_cache[batch_start:batch_end]
            v_cache = v_cache[batch_start:batch_end]
            lengths_per_sample = (
                inference_params.lengths_per_sample[batch_start:batch_end]
                if inference_params.lengths_per_sample is not None
                else None
            )
            # For now, we're emulating the behavior where rotary_emb calculation is done in bf16
            # This is because we have a lot of models already trained with bf16 rotary, and the
            # output of bf16 rotary is quite different from fp32 rotary.
            # If rotary_bf16, compute rotary_cos / sin and pass to ft_attention.
            # If not rotary_bf16, then just set rotary_cos / sin to None.
            if self.pos_emb == "rotary":
                dtype = self.rotary_emb.inv_freq.dtype
                if lengths_per_sample is not None:
                    # angle should have shape (batch, rotary_dim // 2) since that's what ft_attention expects
                    angle = self.rotary_emb.inv_freq * rearrange(lengths_per_sample.to(dtype), "b -> b 1")
                else:
                    angle = repeat(
                        self.rotary_emb.inv_freq * inference_params.sequence_len_offset, "d -> b d", b=q.size(0)
                    ).contiguous()
                rotary_cos, rotary_sin = angle.cos(), angle.sin()
            else:
                rotary_cos, rotary_sin = None, None
            rotary_emb_dim = 0 if self.pos_emb != "rotary" else self.rotary_ndims
            rotary_emb_base = float(self.rotary_emb_base) if self.pos_emb == "rotary" else 0.0
            context_layer = ft_attention.single_query_attention(
                q,
                k,
                v,
                k_cache,
                v_cache,
                lengths_per_sample,
                rotary_cos,
                rotary_sin,
                None,  # nnz_head_idx, unrelated option that we don't ever need
                inference_params.sequence_len_offset,
                rotary_emb_dim,
                rotary_emb_base,
            )
            context_layer = rearrange(context_layer, "b h d -> 1 b (h d)")

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        return output, bias


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)

    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(
    x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(
    x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)


@dataclasses.dataclass
class ParallelTransformerLayerConfig(Config):
    bf16: bool = DISABLE_PREFIX()
    apply_residual_connection_post_layernorm: bool = DISABLE_PREFIX()
    fp32_residual_connection: bool = DISABLE_PREFIX()
    hidden_size: int
    layernorm_epsilon: float = DISABLE_PREFIX()
    no_persist_layer_norm: bool = DISABLE_PREFIX()
    sequence_parallel: bool = DISABLE_PREFIX()
    hidden_dropout: float
    bias_dropout_fusion: bool = DISABLE_PREFIX()
    num_experts: int = DISABLE_PREFIX()
    parallel_mlp_config: ParallelMLPConfig
    parallel_attention_config: ParallelAttentionConfig


class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        parallel_transformer_layer_config: ParallelTransformerLayerConfig,
        init_method,
        output_layer_init_method,
        layer_number,
        module_name: str,
        layer_type=LayerType.encoder,
        self_attn_mask_type=AttnMaskType.padding,
        drop_path_rate=0.0,
    ):
        super(ParallelTransformerLayer, self).__init__()
        self.module_name = module_name
        self.layer_number = layer_number
        self.layer_type = layer_type
        self.sequence_parallel = parallel_transformer_layer_config.sequence_parallel

        self.apply_residual_connection_post_layernorm = (
            parallel_transformer_layer_config.apply_residual_connection_post_layernorm
        )

        self.bf16 = parallel_transformer_layer_config.bf16
        self.fp32_residual_connection = parallel_transformer_layer_config.fp32_residual_connection

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            parallel_transformer_layer_config.hidden_size,
            eps=parallel_transformer_layer_config.layernorm_epsilon,
            no_persist_layer_norm=parallel_transformer_layer_config.no_persist_layer_norm,
            sequence_parallel=self.sequence_parallel,
        )

        # Self attention.
        self.self_attention = ParallelAttention(
            parallel_transformer_layer_config.parallel_attention_config,
            init_method,
            output_layer_init_method,
            layer_number,
            module_name=self.module_name,
            attention_type=AttnType.self_attn,
            attn_mask_type=self_attn_mask_type,
        )
        self.hidden_dropout = parallel_transformer_layer_config.hidden_dropout
        self.bias_dropout_fusion = parallel_transformer_layer_config.bias_dropout_fusion
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNorm(
            parallel_transformer_layer_config.hidden_size,
            eps=parallel_transformer_layer_config.layernorm_epsilon,
            no_persist_layer_norm=parallel_transformer_layer_config.no_persist_layer_norm,
            sequence_parallel=self.sequence_parallel,
        )

        if self.layer_type == LayerType.decoder:
            self.inter_attention = ParallelAttention(
                parallel_transformer_layer_config.parallel_attention_config,
                init_method,
                output_layer_init_method,
                layer_number,
                module_name=self.module_name,
                attention_type=AttnType.cross_attn,
            )
            # Layernorm on the attention output.
            self.post_inter_attention_layernorm = LayerNorm(
                parallel_transformer_layer_config.hidden_size,
                eps=parallel_transformer_layer_config.layernorm_epsilon,
                no_persist_layer_norm=parallel_transformer_layer_config.no_persist_layer_norm,
                sequence_parallel=self.sequence_parallel,
            )

        # MLP
        self.mlp = ParallelMLP(
            parallel_transformer_layer_config.parallel_mlp_config,
            init_method,
            output_layer_init_method,
            init_rng_key=f"{self.module_name}_layer_{layer_number}_mlp",
        )

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split(".")[0])
        TORCH_MINOR = int(torch.__version__.split(".")[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad

    def forward(self, hidden_states, encoder_output=None, inference_params=None):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, attention_bias = self.self_attention(layernorm_output, inference_params=inference_params)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        if self.drop_path is None:
            # jit scripting for a nn.module (with dropout) is not
            # trigerring the fusion kernel. For now, we use two
            # different nn.functional routines to account for varying
            # dropout semantics during training and inference phases.
            if self.bias_dropout_fusion:
                if self.training:
                    bias_dropout_add_func = bias_dropout_add_fused_train
                else:
                    bias_dropout_add_func = bias_dropout_add_fused_inference
            else:
                bias_dropout_add_func = get_bias_dropout_add(self.training)

            with self.bias_dropout_add_exec_handler():
                layernorm_input = bias_dropout_add_func(
                    attention_output, attention_bias.expand_as(residual), residual, self.hidden_dropout
                )
        else:
            out = torch.nn.functional.dropout(
                attention_output + attention_bias, p=self.hidden_dropout, training=self.training
            )
            layernorm_input = residual + self.drop_path(out)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        if self.layer_type == LayerType.decoder:
            attention_output, attention_bias = self.inter_attention(
                layernorm_output, enc_dec_attn_mask, encoder_output=encoder_output
            )
            # residual connection
            if self.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = layernorm_input

            with self.bias_dropout_add_exec_handler():
                layernorm_input = bias_dropout_add_func(
                    attention_output, attention_bias.expand_as(residual), residual, self.hidden_dropout
                )

            # Layer norm post the decoder attention
            layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        if self.drop_path is None:
            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func(mlp_output, mlp_bias.expand_as(residual), residual, self.hidden_dropout)

            # Jit compiled function creates 'view' tensor. This tensor
            # potentially gets saved in the MPU checkpoint function context,
            # which rejects view tensors. While making a viewless tensor here
            # won't result in memory savings (like the data loader, or
            # p2p_communication), it serves to document the origin of this
            # 'view' tensor.
            output = mpu.make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)

        else:
            out = torch.nn.functional.dropout(mlp_output + mlp_bias, p=self.hidden_dropout, training=self.training)
            output = residual + self.drop_path(out)

        return output


class NoopTransformerLayer(MegatronModule):
    """A single 'no-op' transformer layer.

    The sole purpose of this layer is for when a standalone embedding layer
    is used (i.e., args.standalone_embedding_stage == True). In this case,
    zero transformer layers are assigned when pipeline rank == 0. Additionally,
    when virtual pipeline rank >= 1, zero total model parameters are created
    (virtual rank 0 contains the input embedding). This results in the model's
    input and output tensors being the same, which causes an error when
    performing certain memory optimiations on the output tensor (e.g.,
    deallocating it). Thus, this layer disconnects the input from the output
    via a clone. Since ranks containing a no-op layer are generally under-
    utilized (both compute and memory), there's no worry of any performance
    degredation.
    """

    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number

    def forward(self, hidden_states, encoder_output=None, inference_params=None):
        return hidden_states.clone()


@dataclasses.dataclass
class ParallelTransformerConfig(Config):
    bf16: bool = DISABLE_PREFIX()
    decoder_num_layers: int
    distribute_saved_activations: bool = DISABLE_PREFIX()
    encoder_num_layers: int
    fp32_residual_connection: bool = DISABLE_PREFIX()
    hidden_size: int
    layernorm_epsilon: float = DISABLE_PREFIX()
    model_type: ModelType = DISABLE_PREFIX()
    no_persist_layer_norm: bool = DISABLE_PREFIX()
    num_layers: int
    parallel_transformer_layer_config: ParallelTransformerLayerConfig
    pipeline_model_parallel_split_rank: int = DISABLE_PREFIX()
    recompute_granularity: str = DISABLE_PREFIX()
    recompute_method: str = DISABLE_PREFIX()
    recompute_num_layers: int = DISABLE_PREFIX()
    sequence_parallel: bool = DISABLE_PREFIX()
    standalone_embedding_stage: bool = DISABLE_PREFIX()
    transformer_pipeline_model_parallel_size: int = DISABLE_PREFIX()
    virtual_pipeline_model_parallel_size: int = DISABLE_PREFIX()


class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(
        self,
        parallel_transformer_config: ParallelTransformerConfig,
        init_method,
        output_layer_init_method,
        module_name: str,
        layer_type=LayerType.encoder,
        self_attn_mask_type=AttnMaskType.padding,
        post_layer_norm=True,
        pre_process=True,
        post_process=True,
        drop_path_rate=0.0,
        num_layers=None,
    ):
        super(ParallelTransformer, self).__init__()

        self.module_name = (module_name,)
        self.layer_type = layer_type
        self.model_type = parallel_transformer_config.model_type
        self.bf16 = parallel_transformer_config.bf16
        self.fp32_residual_connection = parallel_transformer_config.fp32_residual_connection
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.drop_path_rate = drop_path_rate

        # Store activation checkpointing flag.
        self.recompute_granularity = parallel_transformer_config.recompute_granularity
        self.recompute_method = parallel_transformer_config.recompute_method
        self.recompute_num_layers = parallel_transformer_config.recompute_num_layers
        self.sequence_parallel = parallel_transformer_config.sequence_parallel
        self.distribute_saved_activations = (
            parallel_transformer_config.distribute_saved_activations and not self.sequence_parallel
        )

        # Number of layers.
        if num_layers is None:
            self.num_layers = mpu.get_num_layers(
                pipeline_model_parallel_split_rank=parallel_transformer_config.pipeline_model_parallel_split_rank,
                standalone_embedding_stage=parallel_transformer_config.standalone_embedding_stage,
                transformer_pipeline_model_parallel_size=parallel_transformer_config.transformer_pipeline_model_parallel_size,
                encoder_num_layers=parallel_transformer_config.encoder_num_layers,
                decoder_num_layers=parallel_transformer_config.decoder_num_layers,
                num_layers=parallel_transformer_config.num_layers,
                is_encoder_and_decoder_model=parallel_transformer_config.model_type == ModelType.encoder_and_decoder,
                is_decoder=layer_type == LayerType.decoder,
            )
        else:
            self.num_layers = num_layers

        if layer_type == LayerType.encoder:
            drop_path_layers = parallel_transformer_config.num_layers
        else:
            drop_path_layers = (
                parallel_transformer_config.decoder_num_layers
                if parallel_transformer_config.decoder_num_layers is not None
                else parallel_transformer_config.num_layers
            )
        self.drop_path_rates = [rate.item() for rate in torch.linspace(0, self.drop_path_rate, drop_path_layers)]

        # Transformer layers.
        def build_layer(layer_number):
            return ParallelTransformerLayer(
                parallel_transformer_config.parallel_transformer_layer_config,
                init_method,
                output_layer_init_method,
                layer_number,
                self.module_name,
                layer_type=layer_type,
                self_attn_mask_type=self_attn_mask_type,
                drop_path_rate=self.drop_path_rates[layer_number - 1],
            )

        if parallel_transformer_config.virtual_pipeline_model_parallel_size is not None:
            assert (
                parallel_transformer_config.num_layers
                % parallel_transformer_config.virtual_pipeline_model_parallel_size
                == 0
            ), ("num_layers_per_stage must be divisible by " "virtual_pipeline_model_parallel_size")
            assert parallel_transformer_config.model_type != ModelType.encoder_and_decoder
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // parallel_transformer_config.virtual_pipeline_model_parallel_size
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = mpu.get_virtual_pipeline_model_parallel_rank() * (
                parallel_transformer_config.num_layers
                // parallel_transformer_config.virtual_pipeline_model_parallel_size
            ) + (mpu.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            if (
                parallel_transformer_config.model_type == ModelType.encoder_and_decoder
                and mpu.get_pipeline_model_parallel_world_size() > 1
            ):
                pipeline_rank = mpu.get_pipeline_model_parallel_rank()
                if layer_type == LayerType.encoder:
                    offset = pipeline_rank * self.num_layers
                else:
                    num_ranks_in_enc = parallel_transformer_config.pipeline_model_parallel_split_rank
                    offset = (pipeline_rank - num_ranks_in_enc) * self.num_layers
            else:
                offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers

        if self.num_layers == 0:
            # When a standalone embedding stage is used (e.g.,
            # args.standalone_embedding_stage == True), virtual pipeline ranks
            # on pipeline rank 0 will have zero transformer layers assigned to
            # them. This results in the model's input and output tensors to be
            # the same, which will cause failure for certain output tensor
            # optimizations (e.g., pipeline output deallocation). To remedy
            # this, we assign a 'no-op' layer on these ranks, which will
            # disconnect the input tensor from the output tensor.
            self.num_layers = 1
            self.layers = torch.nn.ModuleList([NoopTransformerLayer(1)])
        else:
            self.layers = torch.nn.ModuleList([build_layer(i + 1 + offset) for i in range(self.num_layers)])

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            self.final_layernorm = LayerNorm(
                parallel_transformer_config.hidden_size,
                eps=parallel_transformer_config.layernorm_epsilon,
                no_persist_layer_norm=parallel_transformer_config.no_persist_layer_norm,
                sequence_parallel=self.sequence_parallel,
            )

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(self, hidden_states, encoder_output):
        """Forward method with activation checkpointing."""

        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                encoder_output = inputs[1]
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, encoder_output)
                return x_

            return custom_forward

        if self.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            l = 0
            while l < self.num_layers:
                hidden_states = mpu.checkpoint(
                    custom(l, l + self.recompute_num_layers),
                    self.distribute_saved_activations,
                    hidden_states,
                    encoder_output,
                )
                l += self.recompute_num_layers

        elif self.recompute_method == "block":
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            for l in range(self.num_layers):
                if l < self.recompute_num_layers:
                    hidden_states = mpu.checkpoint(
                        custom(l, l + 1),
                        self.distribute_saved_activations,
                        hidden_states,
                        encoder_output,
                    )
                else:
                    hidden_states = custom(l, l + 1)(hidden_states, encoder_output)
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self, hidden_states, encoder_output=None, inference_params=None):
        # hidden_states: [s, b, h]

        # Checks.
        if inference_params:
            assert self.recompute_granularity is None, "inference does not work with activation checkpointing"

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = mpu.make_viewless_tensor(
            hidden_states,
            requires_grad=True,
            keep_graph=True,
        )

        if self.sequence_parallel:
            rng_context = mpu.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        with rng_context:
            # Forward pass.
            if self.recompute_granularity == "full":
                hidden_states = self._checkpointed_forward(hidden_states, encoder_output)
            else:
                for index in range(self.num_layers):
                    layer = self._get_layer(index)
                    hidden_states = layer(
                        hidden_states,
                        encoder_output=encoder_output,
                        inference_params=inference_params,
                    )

        # Final layer norm.
        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states

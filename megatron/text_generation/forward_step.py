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

"""Forward step utilities."""

from typing import Optional, Dict, Any, List
from collections.abc import Iterable

from torch import dtype
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args, mpu
from megatron.text_generation.inference_params import InferenceParams
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module
from megatron.model.module import MegatronModule
from megatron.model.gpt_model import GPTModel
from megatron.utils import unwrap_model

from megatron.mpu.communication import recv_from_prev_pipeline_rank_, send_to_next_pipeline_rank


class ForwardStep:
    """Forward step function with all the communications.
    We use a class here to hide the inference parameters
    from the outside caller."""

    def __init__(
        self,
        model: MegatronModule,
        max_batch_size: int,
        max_sequence_len: int,
        inference_params: InferenceParams,
    ):
        """Set values so we don't need to do it multiple times."""
        # Make sure model is in eval mode.
        assert not isinstance(model, Iterable), "interleaving schedule is not supported for inference"
        model.eval()
        self.model = model
        # Initialize inference parameters.
        self.inference_params = inference_params
        # Pipelining arguments.
        args = get_args()
        self.pipeline_size_larger_than_one = args.pipeline_model_parallel_size > 1
        # Threshold of pipelining.
        self.pipelining_batch_x_seqlen = args.inference_batch_times_seqlen_threshold

    def __call__(
        self,
        tokens: torch.Tensor,
        position_ids: torch.Tensor,
        lm_logits_mask: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, Any]]:
        """Invocation of the forward methods. Note that self.inference_params
        is being modified by the forward step."""
        # Pipelining case.
        if self.pipeline_size_larger_than_one:
            current_batch_x_seqlen = tokens.size(0) * tokens.size(1)
            if current_batch_x_seqlen >= self.pipelining_batch_x_seqlen:
                raise ValueError("We deleted _with_pipelining_forward_step")

        return _no_pipelining_forward_step(
            self.model,
            tokens,
            position_ids,
            self.inference_params,
            lm_logits_mask=lm_logits_mask,
        )


def _get_recv_buffer_dtype(args: Any) -> dtype:
    """Receive happens between the layers."""
    if args.fp32_residual_connection:
        return torch.float
    return args.params_dtype


def _allocate_recv_buffer(batch_size: int, sequence_length: int) -> torch.Tensor:
    """Receive happens between the layers with size [s, b, h]."""
    if mpu.is_pipeline_first_stage():
        return None
    args = get_args()
    recv_size = (sequence_length, batch_size, args.hidden_size)
    return torch.empty(recv_size, dtype=_get_recv_buffer_dtype(args), device=torch.cuda.current_device())


def _model_forward_step(
    model: MegatronModule,
    tokens: torch.Tensor,
    position_ids: torch.Tensor,
    inference_params: InferenceParams,
    lm_logits_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    # Run a simple forward pass.
    unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))
    outputs: Dict[str, Any] = {}
    if isinstance(unwrapped_model, GPTModel):
        outputs = model(tokens, position_ids, lm_logits_mask=lm_logits_mask, inference_params=inference_params)
    else:
        assert False, "Unknown model type!" + str(type(unwrapped_model))

    if not isinstance(outputs, dict):
        outputs = {"logits": outputs}

    return outputs


def _forward_step_helper(
    model: MegatronModule,
    tokens: torch.Tensor,
    position_ids: torch.Tensor,
    inference_params: InferenceParams,
    recv_buffer: Optional[torch.Tensor] = None,
    lm_logits_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Single forward step. Update the allocate memory flag so
    only the first time the memory is allocated."""
    batch_size = tokens.size(0)
    sequence_length = tokens.size(1)
    if recv_buffer is None:
        recv_buffer = _allocate_recv_buffer(batch_size, sequence_length)

    # Receive from previous stage.
    recv_from_prev_pipeline_rank_(recv_buffer)

    # Forward pass through the model.
    model.set_input_tensor(recv_buffer)
    outputs = _model_forward_step(
        model,
        tokens,
        position_ids,
        inference_params=inference_params,
        lm_logits_mask=lm_logits_mask,
    )

    # Send output to the next stage.
    send_to_next_pipeline_rank(outputs)

    return outputs


def _no_pipelining_forward_step(
    model: MegatronModule,
    tokens: torch.Tensor,
    position_ids: torch.Tensor,
    inference_params: InferenceParams,
    recv_buffer: Optional[torch.Tensor] = None,
    lm_logits_mask: Optional[torch.Tensor] = None,
) -> Optional[Dict[str, Any]]:
    """If recv_buffer is none, we will allocate one on the fly."""
    # Run a simple forward pass.
    outputs = _forward_step_helper(
        model,
        tokens,
        position_ids,
        inference_params,
        recv_buffer=recv_buffer,
        lm_logits_mask=lm_logits_mask,
    )
    # Update the sequence length offset.
    if inference_params is not None:
        inference_params.sequence_len_offset += tokens.size(1)

    lm_output = None
    if mpu.is_pipeline_last_stage():
        lm_output = outputs

    return lm_output

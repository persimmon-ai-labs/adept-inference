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

"""Megatron Module"""

import argparse
from typing import Any, Callable, Dict, Optional, TypeVar
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from megatron import get_args
from megatron import mpu


_FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
_HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)
_BF16_TYPES = (torch.BFloat16Tensor, torch.cuda.BFloat16Tensor)


def param_is_not_shared(param: Parameter) -> bool:
    return not hasattr(param, "shared") or not param.shared


class MegatronModule(torch.nn.Module):  # type: ignore
    """Megatron specific extensions of torch Module with support
    for pipelining."""

    def __init__(self, share_word_embeddings: bool = True) -> None:
        super().__init__()
        self.share_word_embeddings = share_word_embeddings
        # Initialize the key name for the output word embeddings.
        # This variable is referred at multiple places inside the code,
        # and based on untie_embeddings, it may be present or not.
        # When loading we need to check if they key is present, so this
        # serves as a central place to refer to the variable.
        self._word_embeddings_for_head_key = "word_embeddings_for_head"

    def state_dict_for_save_checkpoint(
        self, destination: Optional[Dict[str, Any]] = None, prefix: str = "", keep_vars: bool = False
    ) -> Dict[str, Any]:
        """Use this function to override the state dict for
        saving checkpoints."""
        state_dict: Dict[str, Any] = self.state_dict(destination, prefix, keep_vars)
        return state_dict

    def word_embeddings_weight(self) -> Parameter:
        """
        This function is responsible for returning the right word_embeddings var
        based on which pipeline stage this is called from and also if we are
        sharing word_embeddings between the input and the output.
        Megatron uses two vars called post_process and pre_process.

        When there is pipelining,
        - post_process is True for last pipeline stage
        - pre_process is True for all stages except the last

        When there is no pipelining,
        - post_process and pre_process is True

        We create a separate var called self.word_embeddings when we are at the
        last pipeline stage or we have share_embeddings False (untie_embeddings is True.)
        """
        args = get_args()

        # When not using pipelining, return the input embedding var if we
        # have share_word_embeddings to be True, and return the new embedding var,
        # word_embeddings, if False.
        if args.pipeline_model_parallel_size == 1:
            if self.share_word_embeddings:
                weight: Parameter = self.language_model.embedding.word_embeddings.weight
                return weight
            else:
                weight = self.word_embeddings.weight
                return weight

        # First condition triggered for last pipeline stage
        if self.post_process and not self.pre_process:
            weight = self.word_embeddings.weight
            return weight
        # Triggered when we are not in the last pipeline stage or when we are
        # sharing word embeddings, which will matter when there is no pipelining.
        elif self.pre_process or self.share_word_embeddings:
            weight = self.language_model.embedding.word_embeddings.weight
            return weight
        # This will trigger for cases when there is no pipelining and
        # no sharing of word_embedding.
        else:
            weight = self.word_embeddings.weight
            return weight

    def initialize_word_embeddings(self, init_method_normal: Callable[..., Any]) -> None:
        args = get_args()

        # This function just initializes the word embeddings in the final stage
        # when we are using pipeline parallelism. Nothing to do if we aren't
        # using pipeline parallelism.

        if args.pipeline_model_parallel_size == 1:
            if not self.share_word_embeddings:
                self.word_embeddings = mpu.VocabParallelEmbedding(
                    args.padded_vocab_size,
                    args.hidden_size,
                    init_method=init_method_normal(args.init_method_std),
                    init_rng_key="output_word_embeddings",
                )
            return

        # Parameters are shared between the word embeddings layers, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.
        if self.post_process and (not self.pre_process or mpu.is_pipeline_last_stage()):
            assert not mpu.is_pipeline_first_stage()
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.word_embeddings = mpu.VocabParallelEmbedding(
                args.padded_vocab_size,
                args.hidden_size,
                init_method=init_method_normal(args.init_method_std),
                init_rng_key="output_word_embeddings",
            )
            if self.share_word_embeddings:
                self.word_embeddings.weight.data.fill_(0)
                self.word_embeddings.weight.shared = True

        # Zero out initial weights for decoder embedding.
        # NOTE: We don't currently support T5 with the interleaved schedule.
        # NOTE: Can we remove the pre_process flag check here?
        if not mpu.is_pipeline_first_stage(ignore_virtual=True) and self.pre_process:
            self.language_model.embedding.zero_parameters()

        if not torch.distributed.is_initialized():
            if not getattr(MegatronModule, "embedding_warning_printed", False):
                print(
                    "WARNING! Distributed processes aren't initialized, so "
                    "word embeddings in the last layer are not initialized. "
                    "If you are just manipulating a model this is fine, but "
                    "this needs to be handled manually. If you are training "
                    "something is definitely wrong."
                )
                MegatronModule.embedding_warning_printed = True
            return

        # Ensure that first and last stages have the same initial parameter
        # values.
        if not args.is_pipeline_conversion:
            if self.share_word_embeddings and mpu.is_rank_in_embedding_group():
                torch.distributed.all_reduce(self.word_embeddings_weight().data, group=mpu.get_embedding_group())

            # Ensure that encoder(first stage) and decoder(split stage) position
            # embeddings have the same initial parameter values
            # NOTE: We don't currently support T5 with the interleaved schedule.
            if mpu.is_rank_in_position_embedding_group() and args.pipeline_model_parallel_split_rank is not None:
                self.language_model.embedding.cuda()
                position_embeddings = self.language_model.embedding.position_embeddings
                torch.distributed.all_reduce(position_embeddings.weight.data, group=mpu.get_position_embedding_group())


T1 = TypeVar("T1")
T2 = TypeVar("T2")


def conversion_helper(val: Any, conversion: Callable[[Any], Any]) -> Any:
    """Apply conversion to val. Recursively apply conversion if `val`
    #is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        return tuple(rtn)
    return rtn


def fp32_to_float16(val: Any, float16_convertor: Callable[[Any], Any]) -> Any:
    """Convert fp32 `val` to fp16/bf16"""

    def half_conversion(val: Any) -> Any:
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, _FLOAT_TYPES):
            val = float16_convertor(val)
        return val

    return conversion_helper(val, half_conversion)


def float16_to_fp32(val: Any) -> Any:
    """Convert fp16/bf16 `val` to fp32"""

    def float_conversion(val: Any) -> Any:
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, (_BF16_TYPES, _HALF_TYPES)):
            val = val.float()
        return val

    return conversion_helper(val, float_conversion)


class Float16Module(MegatronModule):
    def __init__(self, module: torch.nn.Module, args: argparse.Namespace) -> None:
        super().__init__()

        if args.fp16:
            self.add_module("module", module.half())

            def float16_convertor(val: Any) -> Any:
                return val.half()

        elif args.bf16:
            self.add_module("module", module.bfloat16())

            def float16_convertor(val: Any) -> Any:
                return val.bfloat16()

        else:
            raise Exception("should not be here")

        self.float16_convertor = float16_convertor

    def set_input_tensor(self, input_tensor: torch.Tensor) -> Any:
        return self.module.set_input_tensor(input_tensor)

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        if mpu.is_pipeline_first_stage():
            inputs = fp32_to_float16(inputs, self.float16_convertor)
        outputs = self.module(*inputs, **kwargs)
        if mpu.is_pipeline_last_stage():
            outputs = float16_to_fp32(outputs)
        return outputs

    def state_dict(
        self, destination: Optional[Dict[str, Any]] = None, prefix: str = "", keep_vars: bool = False
    ) -> Dict[str, Any]:
        return self.module.state_dict(destination, prefix, keep_vars)  # type: ignore

    def state_dict_for_save_checkpoint(
        self, destination: Optional[Dict[str, Any]] = None, prefix: str = "", keep_vars: bool = False
    ) -> Dict[str, Any]:
        return self.module.state_dict_for_save_checkpoint(destination, prefix, keep_vars)  # type: ignore

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        self.module.load_state_dict(state_dict, strict=strict)

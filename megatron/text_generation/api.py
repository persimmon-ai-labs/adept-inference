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

"""Inference API."""


import base64
from typing import List, Optional, Tuple, Any, Dict

import numpy as np
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron.model.module import MegatronModule
from megatron import mpu
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module

from megatron.tokenizer.tokenizer import AbstractTokenizer
from megatron.mpu.communication import broadcast_float_list, broadcast_int_list
from megatron.text_generation.generation import (
    generate_tokens_probs_and_return_on_first_stage,
    score_and_return_on_first_stage,
)
from megatron.text_generation.inference_params import InferenceParams
from megatron.text_generation.tokenization import (
    convert_generations_to_human_readable_tokens,
    tokenize_prompts,
)
from megatron.utils import unwrap_model
import numpy.typing as npt

def preprocess_prompts(prompts: Optional[List[List[str]]]) -> Optional[List[List[str]]]:
    """
    Accepts a list of list of subprompts, returns a list of list of processed subprompts.
    """
    if prompts is None:
        return None
    processed_prompts = []
    for prompt in prompts:
        processed_subprompts = []
        for subprompt in prompt:
            processed_subprompts.append(f"human: {subprompt.strip()}\n\nadept:")
        processed_prompts.append(processed_subprompts)
    return processed_prompts


def generate_and_post_process(
    model: MegatronModule,
    params_dtype: torch.dtype,
    max_position_embeddings: int,
    termination_id: int,
    tokenizer: AbstractTokenizer,
    prompts: Optional[List[List[str]]] = None,
    max_tokens_to_generate: int = 0,
    inference_params: Optional[InferenceParams] = None,
    return_output_log_probs: bool = False,
    return_all_log_probs: bool = False,
    log_prob_tokens: Optional[torch.Tensor] = None,
    top_k_sampling: int = 0,
    top_p_sampling: float = 0.0,
    temperature: float = 1.0,
    add_BOS: bool = False,
    random_seed: int = -1,
    process_prompts_for_chat: bool = False
) -> Optional[
    Tuple[
        List[str],
        Optional[Any],
        List[List[int]],
        List[str],
        Optional[str],
        Optional[Any],
        List[List[str]],
        Optional[Any],
    ]
]:
    """Run inference and post-process outputs, i.e., detokenize,
    move to cpu and convert to list.

    prompts: a list of list of strings, where each element in the outer list represents a single sample that
    is an item in the batch. A single sample is represented as a list of strings.
    """
    # Pre-process the prompts.
    if process_prompts_for_chat:
        prompts = preprocess_prompts(prompts)

    # Main inference.
    outputs = generate(
        model,
        max_position_embeddings=max_position_embeddings,
        params_dtype=params_dtype,
        termination_id=termination_id,
        tokenizer=tokenizer,
        prompts=prompts,
        max_tokens_to_generate=max_tokens_to_generate,
        inference_params=inference_params,
        return_output_log_probs=return_output_log_probs,
        return_all_log_probs=return_all_log_probs,
        log_prob_tokens=log_prob_tokens,
        top_k_sampling=top_k_sampling,
        top_p_sampling=top_p_sampling,
        temperature=temperature,
        add_BOS=add_BOS,
        random_seed=random_seed,
    )

    # Only post-process on first stage.
    if mpu.is_pipeline_first_stage():

        all_tokens = outputs["tokens"].cpu().numpy().tolist()

        raw_prompts_plus_generations = [tokenizer.detokenize(ts) for ts in all_tokens]
        processed_generations = [
            tokenizer.detokenize(toks) for toks in outputs.get("generated_tokens", [])
        ]

        processed_generated_tokens = [
            tokenizer.encode(g) for g in processed_generations
        ]

        human_readable_tokens = convert_generations_to_human_readable_tokens(
            processed_generated_tokens, "<s>"
        )

        output_log_probs = None
        if return_output_log_probs:
            output_log_probs = outputs["output_log_probs"].cpu().numpy().tolist()

        all_log_probs = None
        if return_all_log_probs:
            all_log_probs = outputs["all_log_probs"].cpu().numpy().tolist()

        return (
            raw_prompts_plus_generations,
            output_log_probs,
            all_tokens,
            processed_generations,
            all_log_probs,
            human_readable_tokens,
        )

    return None


def generate(
    model: MegatronModule,
    max_position_embeddings: int,
    params_dtype: torch.dtype,
    termination_id: int,
    tokenizer: AbstractTokenizer,
    prompts: Optional[List[List[str]]] = None,
    max_tokens_to_generate: int = 0,
    inference_params: Optional[InferenceParams] = None,
    return_output_log_probs: bool = False,
    return_all_log_probs: bool = False,
    log_prob_tokens: Optional[torch.Tensor] = None,
    top_k_sampling: int = 0,
    top_p_sampling: float = 0.0,
    temperature: float = 1.0,
    add_BOS: bool = False,
    random_seed: int = -1,
) -> Dict[str, Any]:
    """Given prompts and input parameters, run inference and return:
    tokens: prompts plus the generated tokens.
    lengths: length of the prompt + generations. Note that we can
        discard tokens in the tokens tensor that are after the
        corresponding length.
    output_log_probs: log probs of the tokens.
    all_log_probs: log probs of all the vocab (or just log_prob_tokens if provided)
        tokens for each generated token position.
    """
    num_log_prob_tokens = 0 if log_prob_tokens is None else len(log_prob_tokens)

    # Make sure input params are avaialble to all ranks.
    values = [
        max_tokens_to_generate,
        return_output_log_probs,
        top_k_sampling,
        top_p_sampling,
        temperature,
        add_BOS,
        random_seed,
        return_all_log_probs,
        num_log_prob_tokens,
    ]
    values_float_tensor = broadcast_float_list(len(values), float_list=values)
    max_tokens_to_generate = int(values_float_tensor[0].item())
    return_output_log_probs = bool(values_float_tensor[1].item())
    top_k_sampling = int(values_float_tensor[2].item())
    top_p_sampling = values_float_tensor[3].item()
    temperature = values_float_tensor[4].item()
    add_BOS = bool(values_float_tensor[5].item())
    random_seed = int(values_float_tensor[6].item())
    return_all_log_probs = bool(values_float_tensor[7].item())
    num_log_prob_tokens = int(values_float_tensor[8].item())

    if return_all_log_probs and num_log_prob_tokens > 0:
        # Do another broadcast for the log_prob_tokens.
        log_prob_tokens = broadcast_int_list(
            num_log_prob_tokens, int_list=log_prob_tokens
        )
    else:
        log_prob_tokens = None

    if random_seed != -1:
        torch.random.manual_seed(random_seed)

    # Tokenize prompts and get the batch.
    # Note that these tensors are broadcasted to all ranks.
    if torch.distributed.get_rank() == 0:
        assert prompts is not None

    context_tokens_tensor, context_length_tensor = tokenize_prompts(
        prompts=prompts,
        max_tokens_to_generate=max_tokens_to_generate,
        max_position_embeddings=max_position_embeddings,
        add_BOS=add_BOS,
    )

    batch_size = context_tokens_tensor.shape[0]
    num_sub_sequences = context_tokens_tensor.shape[1]

    assert num_sub_sequences == 1
    # Remove subsequence dim
    context_length_tensor = torch.squeeze(context_length_tensor, dim=1)
    context_tokens_tensor = torch.squeeze(context_tokens_tensor, dim=1)

    if max_tokens_to_generate == 0:
        assert inference_params is not None
        return score_and_return_on_first_stage(
            model,
            context_tokens_tensor,
            context_length_tensor,
            inference_params,
            max_position_embeddings=max_position_embeddings,
        )

    # Main inference function.
    # Note that the outputs are available on the first stage.
    assert inference_params is not None
    # Added termination_id to support the case that we want to terminate the
    # generation once that id is generated.

    outputs = generate_tokens_probs_and_return_on_first_stage(
        model,
        context_tokens_tensor,
        context_length_tensor,
        inference_params=inference_params,
        max_position_embeddings=max_position_embeddings,
        termination_id=termination_id,
        vocab_size=tokenizer.vocab_size,
        return_output_log_probs=return_output_log_probs,
        log_prob_tokens=log_prob_tokens,
        return_all_log_probs=return_all_log_probs,
        top_k=top_k_sampling,
        top_p=top_p_sampling,
        temperature=temperature,
    )
    # Now we can figure out what actually got generated and return that too
    generated_tokens = []
    context_lengths = context_length_tensor.detach().cpu().numpy().tolist()
    contexts = context_tokens_tensor.detach().cpu().numpy().tolist()
    lengths_cpu = outputs["lengths"].detach().cpu().numpy().tolist()

    for b in range(batch_size):
        assert lengths_cpu[b] > context_lengths[b]
        gen = contexts[b][context_lengths[b] : lengths_cpu[b]]
        generated_tokens.append(gen)

    outputs["generated_tokens"] = generated_tokens

    return outputs

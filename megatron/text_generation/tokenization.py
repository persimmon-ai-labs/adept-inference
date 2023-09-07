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

"""Tokenization utilities."""

import re
from typing import List, Tuple, Optional, Any, NamedTuple, Dict
import numpy as np

import torch

from megatron import get_args, get_tokenizer
from megatron.tokenizer import AbstractTokenizer
from megatron.mpu.communication import broadcast_int_list, broadcast_tensor

TEXT_REPR_BBOX_OPEN = "<box>"
TEXT_REPR_BBOX_CLOSE = "</box>"
TEXT_REPR_POINT_OPEN = "<point>"
TEXT_REPR_POINT_CLOSE = "</point>"


def convert_generations_to_human_readable_tokens(
    generations: List[List[int]], bos_token: str
) -> List[List[str]]:
    """Convert the list of integers that a model outputs into a human-readable list of tokens.
    Args:
        generations: One list per batch, each of which contains a list of integers to detokenize.
        bos_token: The BOS token that we are using.
    Return:
        A list of lists.
    """
    new_generations = []
    tokenizer = get_tokenizer()

    for generation in generations:
        tokens: List[str] = []
        for i, int_token in enumerate(generation):
            token = tokenizer.inv_vocab[int_token]
            # convert underscore into an empty string when it is first.
            if token[0] == "▁" and (i == 0 or tokens[i - 1] == bos_token):
                token = token[1:]
            # continue processing normally.
            token = re.sub("▁", " ", token)
            tokens.append(token)
        new_generations.append(tokens)
    return new_generations


# ====================================================================== TOKENIZATION  #


def tokenize_prompts(
    prompts: Optional[List[List[str]]],
    max_tokens_to_generate: int,
    max_position_embeddings: int,
    add_BOS: bool,
    rank: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize prompts and make them avaiable on all ranks."""
    assert add_BOS is not None

    # On all ranks set to None so we can pass them to functions
    sizes_list = None
    prompts_tokens_cuda_long_tensor = None
    prompts_length_cuda_long_tensor = None

    # On the specified rank, build the above.
    if torch.distributed.get_rank() == rank:
        assert prompts is not None
        assert max_tokens_to_generate is not None
        # Tensor of tokens padded and their unpadded length.
        (
            prompts_tokens_cuda_long_tensor,
            prompts_length_cuda_long_tensor,
        ) = _tokenize_prompts_and_batch(
            prompts,
            max_tokens_to_generate,
            max_position_embeddings,
            add_BOS,
        )
        # We need the sizes of these tensors for the broadcast
        sizes_list = [
            prompts_tokens_cuda_long_tensor.size(0),  # Batch size
            prompts_tokens_cuda_long_tensor.size(1),  # Num subsequences
            prompts_tokens_cuda_long_tensor.size(2),  # Sequence length
        ]
    # First, broadcast the sizes.
    sizes_tensor = broadcast_int_list(3, int_list=sizes_list, rank=rank)

    # Now that we have the sizes, we can broadcast the tokens
    # and length tensors.
    sizes = sizes_tensor.tolist()
    prompts_tokens_cuda_long_tensor = broadcast_tensor(
        sizes, torch.int64, tensor=prompts_tokens_cuda_long_tensor, rank=rank
    )
    prompts_length_cuda_long_tensor = broadcast_tensor(
        sizes[:2], torch.int64, tensor=prompts_length_cuda_long_tensor, rank=rank
    )

    return prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor


def _tokenize_prompts_and_batch(
    prompts: List[List[str]],
    max_tokens_to_generate: int,
    max_position_embeddings: int,
    add_BOS: bool,  # Same issue with types as above
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a set of prompts and number of tokens to generate:
    - tokenize prompts
    - set the sequence length to be the max of length of prompts
      plus the number of tokens we would like to generate
    - pad all the sequences to this length so we can convert them
      into a 3D tensor.
    """
    args = get_args()
    # Tokenize all the prompts.
    tokenizer = get_tokenizer()

    transformed_prompt_tokens = [
        [tokenizer.tokenize(prompt) for prompt in prompt_seq] for prompt_seq in prompts
    ]

    if add_BOS:
        if args.add_bos_prompt_token is not None:
            bos_token = tokenizer.vocab[args.add_bos_prompt_token]
        else:
            bos_token = tokenizer.eod
        prompts_tokens = [
            [[bos_token] + x for x in prompt_seq]
            for prompt_seq in transformed_prompt_tokens
        ]
    else:
        prompts_tokens = transformed_prompt_tokens

    # Now we have a list of list of tokens which each list has a different
    # size. We want to extend this list to:
    #   - incorporate the tokens that need to be generated
    #   - make all the sequences equal length.
    # Get the prompts length.

    prompts_length = [
        [len(x) for x in prompts_tokens_seq] for prompts_tokens_seq in prompts_tokens
    ]
    # Get the max prompts length.
    max_prompt_len: int = np.max(prompts_length)
    # Number of tokens in the each sample of the batch.
    samples_length = min(
        max_prompt_len + max_tokens_to_generate, max_position_embeddings
    )
    if (
        max_prompt_len + max_tokens_to_generate > max_position_embeddings
        and torch.distributed.get_rank() == 0
    ):
        print(
            f"Max subsequence prompt length of {max_prompt_len} + max tokens to generate {max_tokens_to_generate}",
            f"exceeds context length of {max_position_embeddings}. Will generate as many tokens as possible.",
        )
    # Now update the list of list to be of the same size: samples_length.
    for prompt_tokens_seq, prompts_length_seq in zip(prompts_tokens, prompts_length):
        for prompt_tokens, prompt_length in zip(prompt_tokens_seq, prompts_length_seq):
            if len(prompt_tokens) > samples_length:
                raise ValueError(
                    "Length of subsequence prompt exceeds sequence length."
                )
            padding_size = samples_length - prompt_length
            prompt_tokens.extend([tokenizer.eod] * padding_size)

    # Now we are in a structured format, we can convert to tensors.
    prompts_tokens_tensor = torch.cuda.LongTensor(prompts_tokens)
    prompts_length_tensor = torch.cuda.LongTensor(prompts_length)

    return prompts_tokens_tensor, prompts_length_tensor

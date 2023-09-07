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

"""Generation utilities."""

from typing import Optional, Dict, Tuple, List

import torch
import torch.nn.functional as F

from megatron import mpu
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.model.module import MegatronModule

from megatron.mpu.communication import (
    broadcast_from_last_pipeline_stage,
    broadcast_from_last_to_first_pipeline_stage,
    copy_from_last_to_first_pipeline_stage,
)
from .inference_params import InferenceParams
from .forward_step import ForwardStep
from .sampling import sample


def score_and_return_on_first_stage(
    model: MegatronModule,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    inference_params: InferenceParams,
    max_position_embeddings: int,
) -> Dict[str, torch.Tensor]:
    """Function for just scoring.
    Arguments:
        model: no interleaving is supported.
        tokens: prompt tokens extended to be of size [b, max_prompt_length]
        lengths: original prompt length, size: [b]
    Note: Outside of model, other parameters only need to be available on
          rank 0.
    Outputs:
        output_log_probs: log probability of the selected tokens. size: [b, s]
    """

    batch_size = tokens.size(0)
    max_prompt_length = lengths.max().item()
    assert max_prompt_length == tokens.size(1)
    max_sequence_length = min(max_prompt_length, max_position_embeddings)

    # forward step.
    forward_step = ForwardStep(model, batch_size, max_sequence_length, inference_params)

    # ===================
    # Pre-allocate memory
    # ===================

    # Log probability of the sequence (prompt + generated tokens).
    output_log_probs = None
    output_log_probs_size = (batch_size, max_sequence_length - 1)

    if mpu.is_pipeline_last_stage():
        output_log_probs = torch.empty(
            output_log_probs_size,
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        )

    # =============
    # Run inference
    # =============
    with torch.no_grad():
        _, position_ids = _build_attention_mask_and_position_ids(tokens)

        # logits will be meanigful only in the last pipeline stage.
        outputs = forward_step(tokens, position_ids)
        assert outputs is not None
        logits = outputs["logits"]

        if mpu.is_pipeline_last_stage():
            # Always the last stage should have an output.
            assert logits is not None
            assert output_log_probs is not None
            log_probs = F.log_softmax(logits, dim=2)

            # Pick the tokens that we need to get the log
            # probabilities for. Note that next input token is
            # the token which we selected in the current logits,
            # so shift by 1.
            indices = torch.unsqueeze(tokens[:, 1:], 2)
            output_log_probs[:] = torch.gather(log_probs, 2, indices).squeeze(2)

    # ======================================
    # Broadcast to the first pipeline stage.
    # ======================================
    output_log_probs = broadcast_from_last_to_first_pipeline_stage(
        output_log_probs_size, torch.float32, output_log_probs
    )

    return {
        "tokens": tokens,
        "lengths": lengths,
        "output_log_probs": output_log_probs,
    }


def generate_tokens_probs_and_return_on_first_stage(
    model: MegatronModule,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    inference_params: InferenceParams,
    max_position_embeddings: int,
    termination_id: int,
    vocab_size: int,
    *,
    return_output_log_probs: bool = False,
    return_all_log_probs: bool = False,
    log_prob_tokens: Optional[torch.Tensor] = None,
    top_k: int = 0,
    top_p: float = 0.0,
    temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Main token generation function.
    Arguments:
        model: no interleaving is supported.
        tokens: prompt tokens extended to be of size [b, max-sequence-length]
        lengths: original prompt length, size: [b]
        return_output_log_probs: flag to calculate the log probability of
            the generated tokens. Note that the log probability is the one
            from the original logit.
        return_all_log_probs: flag to return log probabilities of all the
            vocab tokens for each generated token. Note that unlike output
            log probs, these are only calculated for the generated tokens
            for efficiency.
        log_prob_tokens: if return_all_log_probs is True, then the log probs
            are restricted to the vocab tokens in this tensor.
        top_k, top_p: top-k and top-p sampling parameters.
            Note that top-k = 1 is gready. Also, these paramters are
            exclusive meaning that:
                if top-k > 0 then we expect top-p=0.
                if top-p > 0 then we check for top-k=0.
        temperature: sampling temperature.
    Note: Outside of model, other parameters only need to be available on
          rank 0.
    Outputs: Note that is size is adjusted to a lower value than
             max-sequence-length if generation is terminated early.
        tokens: prompt and generated tokens. size: [b, :]
        generated_sequence_lengths: total length (including prompt) of
            the generated sequence. size: [b]
        output_log_probs: log probability of the selected tokens. size: [b, s]
    """

    batch_size = tokens.size(0)
    min_prompt_length = lengths.min().item()
    max_prompt_length = lengths.max().item()
    max_sequence_length = tokens.size(1)
    tokens_to_generate = max_sequence_length - max_prompt_length

    if max_sequence_length > max_position_embeddings:
        raise ValueError("Length of prompt + tokens_to_generate longer than allowed")

    # forward step.
    forward_step = ForwardStep(model, batch_size, max_sequence_length, inference_params)

    # ===================
    # Pre-allocate memory
    # ===================

    # Log probability of the sequence (prompt + generated tokens).
    output_log_probs = None
    output_log_probs_size = (batch_size, max_sequence_length - 1)

    # Log probability of all vocab tokens for each generated token. We only do generated tokens so that memory
    # usage is not too high.
    all_log_probs = None
    log_prob_vocab_size = (
        vocab_size if log_prob_tokens is None else len(log_prob_tokens)
    )
    all_log_probs_size = (batch_size, tokens_to_generate, log_prob_vocab_size)

    # Lengths of generated seuquence including including prompts.
    generated_sequence_lengths = None
    if mpu.is_pipeline_last_stage():
        if return_output_log_probs:
            output_log_probs = torch.zeros(
                output_log_probs_size,
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
        if return_all_log_probs:
            all_log_probs = torch.empty(
                all_log_probs_size,
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
        generated_sequence_lengths = (
            torch.ones(
                batch_size, dtype=torch.int64, device=torch.cuda.current_device()
            )
            * max_sequence_length
        )

    # Whether we have reached a termination id.
    is_generation_done = torch.zeros(
        batch_size, dtype=torch.uint8, device=torch.cuda.current_device()
    )

    # Keep track of the indices of the generated token for each prompt in the batch.
    gen_token_indices = torch.full(
        (batch_size,), -1, dtype=torch.int64, device=torch.cuda.current_device()
    )

    # =============
    # Run inference
    # =============

    with torch.inference_mode():
        _, position_ids = _build_attention_mask_and_position_ids(tokens)
        prev_context_length = 0

        if return_output_log_probs:
            lm_logits_mask = None
        else:
            # We do not care about the logits for the prompt tokens, so we can ignore them.
            # This is strictly a performance optimization.
            lm_logits_mask = torch.ones(
                max_sequence_length,
                dtype=torch.bool,
                device=torch.cuda.current_device(),
            )
            lm_logits_mask[: min_prompt_length - 1] = False

        for context_length in range(min_prompt_length, max_sequence_length):
            # Pick the slice that we need to pass through the network.
            tokens2use = tokens[:, prev_context_length:context_length]
            positions2use = position_ids[:, prev_context_length:context_length]
            lm_logits_mask2use = (
                None
                if lm_logits_mask is None
                else lm_logits_mask[prev_context_length:context_length]
            )

            # logits will be meaningful only in the last pipeline stage.
            if (
                not hasattr(model, "_cg_cache")
                or inference_params.sequence_len_offset == 0
            ):
                outputs = forward_step(
                    tokens2use,
                    positions2use,
                    lm_logits_mask=lm_logits_mask2use,
                )
            else:
                outputs = model._cg_cache.run(
                    tokens2use, positions2use, inference_params.sequence_len_offset
                )
                # This is needed since it's on CPU side, and CUDA graph reply doesn't execute this step
                inference_params.sequence_len_offset += 1

            assert outputs is not None
            logits = outputs["logits"]

            if mpu.is_pipeline_last_stage():
                # Always the last stage should have an output.
                assert logits is not None

                # Sample.
                last_token_logits = logits[:, -1, :].float()

                new_sample = sample(
                    last_token_logits,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    vocab_size=vocab_size,
                )

                # If a prompt length is smaller or equal to the current context
                # length, it means we have started generating tokens
                started = lengths <= context_length
                # Whether we've finished generating the tokens of interest
                finished = context_length >= (lengths + tokens_to_generate)
                # Increment the gen_token_indices if started and not finished
                gen_token_indices[started & ~finished] += 1

                # Update the tokens.
                tokens[started, context_length] = new_sample[started]

                # Calculate the log probabilities.
                if return_output_log_probs:
                    # [b, v]
                    log_probs = F.log_softmax(logits, dim=2)

                    # Pick the tokens that we need to get the log
                    # probabilities for. Note that next input token is
                    # the token which we selected in the current logits,
                    # so shift by 1.
                    indices = torch.unsqueeze(
                        tokens[:, (prev_context_length + 1) : (context_length + 1)], 2
                    )
                    assert output_log_probs is not None
                    output_log_probs[
                        :, prev_context_length:context_length
                    ] = torch.gather(log_probs, 2, indices).squeeze(2)

                if return_all_log_probs:
                    last_token_log_softmax = F.log_softmax(
                        last_token_logits, dim=-1
                    )  # Shape: [b, v]
                    if log_prob_tokens is not None:
                        # Pick only the log probabilities of the tokens in log_prob_tokens
                        last_token_log_softmax = last_token_log_softmax[
                            :, log_prob_tokens
                        ]

                    # Update all_log_probs for the cases where we've started but not finished generating tokens
                    # of interest
                    indices = started & ~finished
                    assert all_log_probs is not None
                    all_log_probs[
                        indices, gen_token_indices[indices], :
                    ] = last_token_log_softmax[indices, :]

            # Update the tokens on the first stage so the next input to
            # the network is correct.
            copy_from_last_to_first_pipeline_stage(
                batch_size, torch.int64, tokens[:, context_length]
            )

            if inference_params is not None:
                # Update the context length for the next token generation if using KV cache.
                prev_context_length = context_length

            # Check if all the sequences have hit the termination_id.
            done = None
            if mpu.is_pipeline_last_stage():
                done_token = (new_sample == termination_id).byte() & started.byte()

                just_finished = (done_token & ~is_generation_done).bool()
                assert generated_sequence_lengths is not None
                generated_sequence_lengths[just_finished.view(-1)] = context_length + 1
                is_generation_done = is_generation_done | done_token
                done = torch.all(is_generation_done)
            done = broadcast_from_last_pipeline_stage(1, torch.uint8, tensor=done)
            if done:  # always use eod token for early termination
                break

    # ===================================================
    # Update the length of based on max generated length.
    # ===================================================

    tokens = tokens[:, : (context_length + 1)]
    if mpu.is_pipeline_last_stage():
        if return_output_log_probs:
            assert output_log_probs is not None
            output_log_probs = output_log_probs[:, :context_length]

    # ======================================
    # Broadcast to the first pipeline stage.
    # ======================================

    generated_sequence_lengths = broadcast_from_last_to_first_pipeline_stage(
        batch_size, torch.int64, generated_sequence_lengths
    )
    if return_output_log_probs:
        output_log_probs_size = (batch_size, context_length)
        output_log_probs = broadcast_from_last_to_first_pipeline_stage(
            output_log_probs_size, torch.float32, output_log_probs
        )
    if return_all_log_probs:
        all_log_probs = broadcast_from_last_to_first_pipeline_stage(
            all_log_probs_size, torch.float32, all_log_probs
        )

    return {
        "tokens": tokens,
        "lengths": generated_sequence_lengths,
        "output_log_probs": output_log_probs,
        "all_log_probs": all_log_probs,
    }


def _build_attention_mask_and_position_ids(
    tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the attention mask and postition ids for the input tokens."""

    # Since we are not interested in loss-mask and reset attention/position
    # is also False, eod_token is not used so it is safe to set it to None.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=None,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )

    return attention_mask, position_ids

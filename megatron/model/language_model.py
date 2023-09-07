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

"""Transformer based language model."""

from typing import List, Optional

import torch

from megatron import get_args, mpu
from megatron.model.enums import AttnMaskType, LayerType
from megatron.model.transformer import ParallelTransformer, ParallelTransformerConfig
from megatron.model.utils import (
    check_shapes,
    get_linear_layer,
    init_method_normal,
    scaled_init_method_normal,
)
from .module import MegatronModule


def parallel_lm_logits(input_, word_embeddings_weight, parallel_output, bias=None):
    """LM logits using word embedding weights."""
    args = get_args()
    # Parallel logits.
    if args.async_tensor_model_parallel_allreduce or args.sequence_parallel:
        input_parallel = input_
        model_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        async_grad_allreduce = (
            args.async_tensor_model_parallel_allreduce and model_parallel and not args.sequence_parallel
        )
    else:
        input_parallel = mpu.reduce_backward_from_tensor_model_parallel_region(input_)
        async_grad_allreduce = False

    # Matrix multiply.
    logits_parallel = mpu.LinearWithGradAccumulationAndAsyncCommunication.apply(
        input_parallel,
        word_embeddings_weight,
        bias,
        args.gradient_accumulation_fusion,
        async_grad_allreduce,
        args.sequence_parallel,
    )
    # Gather if needed.

    if parallel_output:
        return logits_parallel

    return mpu.gather_from_tensor_model_parallel_region(logits_parallel)


def get_language_model(
    num_tokentypes,
    add_pooler,
    encoder_attn_mask_type,
    init_method=None,
    scaled_init_method=None,
    add_encoder=True,
    add_decoder=False,
    decoder_attn_mask_type=AttnMaskType.causal,
    pre_process=True,
    post_process=True,
    continuous_embed_input_size=None,
):
    """Build language model and return along with the key to save."""
    args = get_args()

    if init_method is None:
        init_method = init_method_normal(args.init_method_std)

    if scaled_init_method is None:
        scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)

    # Language model.
    language_model = TransformerLanguageModel(
        init_method,
        scaled_init_method,
        encoder_attn_mask_type,
        num_tokentypes=num_tokentypes,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        decoder_attn_mask_type=decoder_attn_mask_type,
        add_pooler=add_pooler,
        pre_process=pre_process,
        post_process=post_process,
        continuous_embed_input_size=continuous_embed_input_size,
    )
    # key used for checkpoints.
    language_model_key = "language_model"

    return language_model, language_model_key


class Embedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(
        self,
        hidden_size,
        vocab_size,
        max_sequence_length,
        embedding_dropout_prob,
        init_method,
        num_tokentypes=0,
        pos_emb=None,
        continuous_embed_input_size=None,
    ):
        super(Embedding, self).__init__()

        self.hidden_size = hidden_size
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes
        self.pos_emb = pos_emb
        self.continuous_embed_input_size = continuous_embed_input_size

        args = get_args()

        from megatron import print_rank_0  # pylint: disable=import-outside-toplevel

        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size, self.hidden_size, init_method=self.init_method, init_rng_key="input_word_embeddings"
        )
        self._word_embeddings_key = "word_embeddings"

        # Position embedding (serial).
        if pos_emb == "sinusoidal" or pos_emb == "learned":
            if pos_emb == "sinusoidal":
                print_rank_0(
                    "WARNING: sinusoidal positional embedding argument is deprecated. It never "
                    "meant sinusoidal and always meant learned. Please use 'learned' in the future."
                )
            self.position_embeddings = mpu.VocabParallelEmbedding(
                max_sequence_length,
                self.hidden_size,
                init_method=self.init_method,
                init_rng_key="input_position_embeddings",
            )
            self._position_embeddings_key = "position_embeddings"
        else:
            self.position_embeddings = None

        self.continuous_embed_linear = None
        if continuous_embed_input_size:
            if continuous_embed_input_size != hidden_size:
                assert False, (
                    f"All our current configs have continuous_embed_input_size == hidden_size, but got "
                    f"{continuous_embed_input_size=} and {hidden_size=}. This is probably a mistake. "
                    f"(Feel free to delete this assert if we're doing this on purpose.)"
                )
                print_rank_0(
                    f"Using linear layer to project from continuous embedding size of {continuous_embed_input_size} to hidden size of {hidden_size}."
                )
                self.continuous_embed_linear = mpu.ParallelLinear(
                    self.continuous_embed_input_size,
                    self.hidden_size,
                    bias=False,
                    init_method=self.init_method,
                    init_rng_key="input_continuous_embed_linear",
                    sequence_parallel=args.sequence_parallel,
                )
                self._continuous_embed_linear_key = "continuous_embed_linear"
            else:
                print_rank_0(
                    f"Skipping linear layer to project from continuous embedding size of {continuous_embed_input_size} to hidden size of {hidden_size}."
                )

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self._tokentype_embeddings_key = "tokentype_embeddings"
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes, self.hidden_size)
            # Initialize the token-type embeddings.
            if args.perform_initialization:
                self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        self.fp32_residual_connection = args.fp32_residual_connection
        self.sequence_parallel = args.sequence_parallel
        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        if self.pos_emb == "sinusoidal" or self.pos_emb == "learned":
            self.position_embeddings.weight.data.fill_(0)
            self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            self.tokentype_embeddings.weight.data.fill_(0)
            self.tokentype_embeddings.weight.shared = True
        if self.continuous_embed_linear:
            raise ValueError("Zeroing out parameters not supported when continuous_embed_linear is used.")

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception("tokentype embeddings is already initialized")
        from megatron import print_rank_0  # pylint: disable=import-outside-toplevel

        print_rank_0("adding embedding for {} tokentypes".format(num_tokentypes))
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = torch.nn.Embedding(num_tokentypes, self.hidden_size)
        # Initialize the token-type embeddings.
        args = get_args()
        self.init_method(self.tokentype_embeddings.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        tokentype_ids: Optional[torch.Tensor] = None,
        continuous_embeddings: Optional[List[torch.Tensor]] = None,
    ):
        """Embeddings layer forward pass.

        Args:
            input_ids: input token ids. Shape: [batch_size, sequence_length].
            position_ids: Input position ids. Shape: [batch_size, sequence_length].
            tokentype_ids: token type ids. Shape: [batch_size, sequence_length].
            continuous_embeddings: continuous embeddings.
        """
        # Embeddings.
        word_embeddings: torch.Tensor = self.word_embeddings(input_ids)

        assert continuous_embeddings is None

        if self.pos_emb == "sinusoidal" or self.pos_emb == "learned":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = word_embeddings + position_embeddings
        else:
            embeddings = word_embeddings

        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        else:
            assert self.tokentype_embeddings is None

        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.sequence_parallel:
            embeddings = mpu.scatter_to_sequence_parallel_region(embeddings)
            with mpu.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)

        return embeddings

    def state_dict_for_save_checkpoint(self, destination=None, prefix="", keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._word_embeddings_key] = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        if self.pos_emb == "sinusoidal" or self.pos_emb == "learned":
            state_dict_[self._position_embeddings_key] = self.position_embeddings.state_dict(
                destination, prefix, keep_vars
            )
        if self.num_tokentypes > 0:
            state_dict_[self._tokentype_embeddings_key] = self.tokentype_embeddings.state_dict(
                destination, prefix, keep_vars
            )
        if self.continuous_embed_linear is not None:
            state_dict_[self._continuous_embed_linear_key] = self.continuous_embed_linear.state_dict(
                destination, prefix, keep_vars
            )

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""
        from megatron import print_rank_0  # pylint: disable=import-outside-toplevel

        # Word embedding.
        if self._word_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
            self.word_embeddings.load_state_dict(state_dict_, strict=strict)
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if "word_embeddings" in key:
                    state_dict_[key.split("word_embeddings.")[1]] = state_dict[key]
                    print_rank_0("Load word embeddings in state dict by splitting original key")
            if len(state_dict_.keys()) > 1:
                self.word_embeddings.load_state_dict(state_dict_, strict=strict)

        # Position embedding.
        if hasattr(self, "_position_embeddings_key"):
            if self._position_embeddings_key in state_dict:
                state_dict_ = state_dict[self._position_embeddings_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if "position_embeddings" in key:
                        state_dict_[key.split("position_embeddings.")[1]] = state_dict[key]
            self.position_embeddings.load_state_dict(state_dict_, strict=strict)

        # Tokentype embedding.
        if self.num_tokentypes > 0:
            state_dict_ = {}
            if self._tokentype_embeddings_key in state_dict:
                state_dict_ = state_dict[self._tokentype_embeddings_key]
            else:
                # for backward compatibility.
                for key in state_dict.keys():
                    if "tokentype_embeddings" in key:
                        state_dict_[key.split("tokentype_embeddings.")[1]] = state_dict[key]
            if len(state_dict_.keys()) > 0:
                self.tokentype_embeddings.load_state_dict(state_dict_, strict=strict)
            else:
                print_rank_0("***WARNING*** expected tokentype embeddings in the " "checkpoint but could not find it")

        # Continuous embedding linear layer.
        if self.continuous_embed_linear:
            self.continuous_embed_linear.load_state_dict(state_dict[self._continuous_embed_linear_key], strict=strict)


class TransformerLanguageModel(MegatronModule):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(
        self,
        init_method,
        output_layer_init_method,
        encoder_attn_mask_type,
        num_tokentypes=0,
        add_encoder=True,
        add_decoder=False,
        decoder_attn_mask_type=AttnMaskType.causal,
        add_pooler=False,
        pre_process=True,
        post_process=True,
        continuous_embed_input_size=None,
    ):
        super(TransformerLanguageModel, self).__init__()
        args = get_args()

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = args.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method = init_method
        self.add_encoder = add_encoder
        self.encoder_attn_mask_type = encoder_attn_mask_type
        self.add_decoder = add_decoder
        self.decoder_attn_mask_type = decoder_attn_mask_type
        self.add_pooler = add_pooler
        self.encoder_hidden_state = None
        self.pos_emb = args.pos_emb

        # Embeddings.
        if self.pre_process:
            self.embedding = Embedding(
                self.hidden_size,
                args.padded_vocab_size,
                args.max_position_embeddings,
                args.hidden_dropout,
                self.init_method,
                self.num_tokentypes,
                pos_emb=args.pos_emb,
                continuous_embed_input_size=continuous_embed_input_size,
            )
            self._embedding_key = "embedding"

        # Transformer.
        # Encoder (usually set to True, False if part of an encoder-decoder
        # architecture and in encoder-only stage).
        if self.add_encoder:
            self._encoder_key = "encoder"
            self.encoder = ParallelTransformer(
                ParallelTransformerConfig.from_args(args),
                self.init_method,
                output_layer_init_method,
                self_attn_mask_type=self.encoder_attn_mask_type,
                pre_process=self.pre_process,
                post_process=self.post_process,
                module_name=self._encoder_key,
            )
        else:
            self.encoder = None

        # Decoder (usually set to False, True if part of an encoder-decoder
        # architecture and in decoder-only stage).
        if self.add_decoder:
            self._decoder_key = "decoder"
            self.decoder = ParallelTransformer(
                ParallelTransformerConfig.from_args(args),
                self.init_method,
                output_layer_init_method,
                layer_type=LayerType.decoder,
                self_attn_mask_type=self.decoder_attn_mask_type,
                pre_process=self.pre_process,
                post_process=self.post_process,
                module_name=self._decoder_key,
            )
        else:
            self.decoder = None


    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""

        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        if self.add_encoder and self.add_decoder:
            assert (
                len(input_tensor) == 1
            ), "input_tensor should only be length 1 for stage with both encoder and decoder"
            self.encoder.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            assert len(input_tensor) == 1, "input_tensor should only be length 1 for stage with only encoder"
            self.encoder.set_input_tensor(input_tensor[0])
        elif self.add_decoder:
            if len(input_tensor) == 2:
                self.decoder.set_input_tensor(input_tensor[0])
                self.encoder_hidden_state = input_tensor[1]
            elif len(input_tensor) == 1:
                self.decoder.set_input_tensor(None)
                self.encoder_hidden_state = input_tensor[0]
            else:
                raise Exception("input_tensor must have either length 1 or 2")
        else:
            raise Exception("Stage must have at least either encoder or decoder")

    def forward(
        self,
        enc_input_ids,
        enc_position_ids,
        dec_input_ids=None,
        dec_position_ids=None,
        tokentype_ids=None,
        inference_params=None,
        pooling_sequence_index=0,
        enc_hidden_states=None,
        output_enc_hidden=False,
        continuous_embeddings: Optional[List[torch.Tensor]] = None,
    ):
        args = get_args()

        # Encoder embedding.
        if self.pre_process and self.add_encoder:
            encoder_input = self.embedding(
                enc_input_ids,
                enc_position_ids,
                tokentype_ids=tokentype_ids,
                continuous_embeddings=continuous_embeddings,
            )
        else:
            encoder_input = None

        # Run encoder.
        if enc_hidden_states is None:
            if self.encoder is not None:
                encoder_output = self.encoder(encoder_input, inference_params=inference_params)
            else:
                encoder_output = self.encoder_hidden_state
        else:
            assert False, "T5 models are no longer supported"

        # output_enc_hidden refers to when we just need the encoder's
        # output. For example, it is helpful to compute
        # similarity between two sequences by average pooling
        if not self.add_decoder or output_enc_hidden:
            return encoder_output

        # Decoder embedding.
        if self.pre_process:
            decoder_input = self.embedding(dec_input_ids, dec_position_ids)
        else:
            decoder_input = None

        # Run decoder.
        decoder_output = self.decoder(
            decoder_input,
            encoder_output=encoder_output,
            inference_params=inference_params,
        )

        return decoder_output, encoder_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix="", keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        if self.pre_process:
            state_dict_[self._embedding_key] = self.embedding.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars
            )
        if self.add_encoder:
            state_dict_[self._encoder_key] = self.encoder.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        if self.post_process:
            if self.add_pooler:
                state_dict_[self._pooler_key] = self.pooler.state_dict_for_save_checkpoint(
                    destination, prefix, keep_vars
                )
        if self.add_decoder:
            state_dict_[self._decoder_key] = self.decoder.state_dict_for_save_checkpoint(destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        args = get_args()
        # Embedding.
        if self.pre_process:
            if self._embedding_key in state_dict:
                state_dict_ = state_dict[self._embedding_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if "_embeddings" in key:
                        state_dict_[key] = state_dict[key]
            self.embedding.load_state_dict(state_dict_, strict=strict)

        # Encoder.
        if self.add_encoder:
            if self._encoder_key in state_dict:
                state_dict_ = state_dict[self._encoder_key]
            # For backward compatibility.
            elif "transformer" in state_dict:
                state_dict_ = state_dict["transformer"]
            else:
                # For backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if "transformer." in key:
                        state_dict_[key.split("transformer.")[1]] = state_dict[key]

            # For backward compatibility.
            state_dict_self_attention = {}
            for key in state_dict_.keys():
                if ".attention." in key:
                    state_dict_self_attention[key.replace(".attention.", ".self_attention.")] = state_dict_[key]
                else:
                    state_dict_self_attention[key] = state_dict_[key]
            state_dict_ = state_dict_self_attention

            # Compatibility for pipeline conversion. Add random layernorm weight and bias to the model
            if args.is_pipeline_conversion:
                if "final_layernorm.weight" not in state_dict_:
                    from megatron import print_rank_0  # pylint: disable=import-outside-toplevel

                    print_rank_0(
                        "Create a random final_layernorm weight and bias tensor when not present in the checkpoint"
                    )
                    state_dict_["final_layernorm.weight"] = torch.rand((self.hidden_size))
                    state_dict_["final_layernorm.bias"] = torch.rand((self.hidden_size))

            self.encoder.load_state_dict(state_dict_, strict=strict)

        # Decoder.
        if self.add_decoder:
            assert "decoder" in state_dict, "could not find data for pooler in the checkpoint"
            self.decoder.load_state_dict(state_dict[self._decoder_key], strict=strict)

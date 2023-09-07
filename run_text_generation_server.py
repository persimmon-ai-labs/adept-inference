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

"""Sample Generate GPT"""
import os
import sys
from typing import Any, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import torch

from megatron import get_args, get_tokenizer, mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.model.module import MegatronModule
from megatron.model.utils import print_named_parameters
from megatron.text_generation.api import generate_and_post_process
from megatron.text_generation.inference_params import InferenceParams
from megatron.text_generation_server import (MegatronServer,
                                             add_text_generate_args,
                                             setup_model)
from megatron.training import get_model

MAX_BATCH_SIZE = 1  # You can increase this depending on your desired max sequence length and GPU memory
MAX_SEQLEN = 16 * 1024


def model_provider(
        pre_process: bool=True,
        post_process: bool=True
    ) -> MegatronModule:
    """Build the model."""

    args = get_args()
    if args.model_architecture == "GPTModel":
        model : MegatronModule = GPTModel(num_tokentypes=0, parallel_output=False, pre_process=pre_process, post_process=post_process)
    else:
        raise ValueError(f"Unsupported model type: {args.model_architecture}")
    print_named_parameters(model)
    return model


def initialize_model_from_args() -> Tuple[Any, Optional[InferenceParams]]:
    # Needed for tensor parallel inference with CUDA graph
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = "0"

    initialize_megatron(
        extra_args_provider=add_text_generate_args,
        args_defaults={
            "tokenizer_type": "GPT2BPETokenizer",
            "no_load_rng": True,
            "no_load_optim": True,
            "inference_max_batch_size": MAX_BATCH_SIZE,
            "inference_max_seqlen": MAX_SEQLEN,
        },
    )

    args = get_args()
    if not args.fused_ft_kernel:
        args.use_cuda_graph = False  # CUDA graph requires fused FT kernel
    if hasattr(args, "iteration"):
        args.curr_iteration = args.iteration
        print("curr_iteration", args.curr_iteration)
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    args.model_architecture = "GPTModel"
    inference_params = setup_model(
        model,
        args.model_architecture,
        args.use_inference_kv_cache,
        args.fused_ft_kernel,
        args.use_cuda_graph,
        args.inference_max_batch_size,
        args.inference_max_seqlen,
    )

    return model, inference_params


if __name__ == "__main__":
    model, inference_params = initialize_model_from_args()

    args = get_args()
    tokenizer = get_tokenizer()
    if hasattr(args, "eos_id"):
        termination_id = args.eos_id
    else:
        termination_id = tokenizer.eod
    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        assert inference_params is not None
        server = MegatronServer(
            model=model,
            inference_params=inference_params,
            params_dtype=args.params_dtype,
            max_position_embeddings=args.max_position_embeddings,
            termination_id=termination_id,
            tokenizer=tokenizer,
            port=args.port,
            )
        server.run("0.0.0.0")

    while True:
        if inference_params is not None:
            inference_params.reset()
        choice = torch.cuda.LongTensor(1)
        torch.distributed.broadcast(choice, 0)
        if choice[0].item() == 0:
            try:
                assert inference_params is not None
                generate_and_post_process(
                    model=model,
                    params_dtype=args.params_dtype,
                    max_position_embeddings=args.max_position_embeddings,
                    termination_id=termination_id,
                    tokenizer=tokenizer,
                    inference_params=inference_params)
            except ValueError as ve:
                pass

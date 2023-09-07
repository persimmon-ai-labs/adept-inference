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
import collections
import datetime
import json
import os
import threading
import gc
import numbers
from pathlib import Path
import urllib.request
import copy
import traceback
import argparse

from typing import Optional, Callable, Tuple, Dict, Union, Any, List

from dataclasses import dataclass, field

import torch
from torch import Tensor

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from flask_restful import Api, Resource  # type: ignore
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron.tokenizer.tokenizer import AbstractTokenizer
from megatron.model.module import MegatronModule
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module
from megatron.text_generation.api import generate_and_post_process
from megatron.text_generation.inference_params import InferenceParams
from megatron.text_generation.forward_step import ForwardStep
from megatron.utils import unwrap_model


GENERATE_NUM = 0
BEAM_NUM = 1
lock = threading.Lock()


def add_text_generate_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="text generation")

    group.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature."
    )
    group.add_argument("--top_p", type=float, default=0.0, help="Top p sampling.")
    group.add_argument("--top_k", type=int, default=0, help="Top k sampling.")
    group.add_argument("--port", type=int, default=5000, help="Server port.")
    group.add_argument(
        "--out-seq-length",
        type=int,
        default=1024,
        help="Size of the output generated text.",
    )

    group.add_argument(
        "--use-inference-kv-cache",
        dest="use_inference_kv_cache",
        action="store_true",
        help="Use a KV cache",
    )
    group.add_argument(
        "--no-use-inference-kv-cache",
        dest="use_inference_kv_cache",
        action="store_false",
        help="Don't use a KV cache",
    )
    group.set_defaults(use_inference_kv_cache=True)

    group.add_argument(
        "--fused-ft-kernel",
        dest="fused_ft_kernel",
        action="store_true",
        help="Use fused FT kernel during inference",
    )
    group.add_argument(
        "--no-fused-ft-kernel",
        dest="fused_ft_kernel",
        action="store_false",
        help="Don't use fused FT kernel during inference",
    )
    group.set_defaults(fused_ft_kernel=True)
    group.add_argument(
        "--use-cuda-graph",
        dest="use_cuda_graph",
        action="store_true",
        help="Use CUDA graph",
    )
    group.add_argument(
        "--no-use-cuda-graph",
        dest="use_cuda_graph",
        action="store_false",
        help="Don't use CUDA graph",
    )
    group.set_defaults(use_cuda_graph=True)

    group.add_argument(
        "--add-bos-prompt-token",
        type=str,
        default=None,
        help="Custom BOS token to use instead of EOD",
    )
    group.add_argument(
        "--model-architecture",
        default="GPTModel",
        type=str,
        choices=["GPTModel"],
        help="Model type for Inference.",
    )

    group.add_argument(
        "--inference-max-batch-size",
        type=int,
        help="Maximum batch size for inference.",
    )
    group.add_argument(
        "--inference-max-seqlen",
        type=int,
        help="Maximum sequence length for inference.",
    )

    return parser


def seqlen_to_seqlen_type(seqlen: int) -> int:
    """Convert sequence length to a seqlen_type.
    This is used to determine which cuda graph to use.
    The case work is due to the fact that the attention kernel from FasterTransformer has 3 cases,
    for seqlen < 32, seqlen < 2048, and seqlen >= 2048.
    Arguments:
        seqlen: int
    """
    if seqlen < 32:
        return 0
    elif seqlen < 2048:
        return 1
    else:
        return 2


def seqlen_type_to_max_seqlen(seqlen_type: int) -> int:
    if seqlen_type == 0:
        return 32
    elif seqlen_type == 1:
        return 2048
    elif seqlen_type == 2:
        return 2**32  # Will take the min of this and the actual max_seqlen
    else:
        raise ValueError(f"seqlen_type {seqlen_type} not supported")


BatchSize = int
SeqLenType = int
SeqLenOffset = int
ForwardStepOutput = Optional[Dict[str, Any]]


@dataclass
class DecodingCGCache:
    max_batch_size: int = 0
    max_seqlen: int = 0
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    callables: Dict[
        Tuple[BatchSize, SeqLenType],
        Callable[[Tensor, Tensor, SeqLenOffset], ForwardStepOutput],
    ] = field(default_factory=dict)
    mempool: Optional[torch.Tensor] = None
    inference_params: Optional[InferenceParams] = None
    run: Optional[
        Callable[
            [torch.Tensor, torch.Tensor, int],
            Callable[[torch.Tensor, torch.Tensor, int], Optional[Dict[str, Any]]],
        ]
    ] = None


@torch.inference_mode()  # type: ignore
def update_graph_cache(
    model: MegatronModule,
    cg_cache: Optional[DecodingCGCache],
    inference_params: InferenceParams,
    batch_size: int,
    seqlen_og: int,
    max_seqlen: int,
    dtype: Optional[torch.dtype] = None,
    n_warmups: int = 2,
):
    # PyTorch docs says at least 2 warmups are needed.
    if cg_cache is None:
        cg_cache = DecodingCGCache()
    param_example = next(iter(model.parameters()))
    device = param_example.device
    if dtype is None:
        dtype = param_example.dtype
    if (
        (device, dtype) != (cg_cache.device, cg_cache.dtype)
        or batch_size > cg_cache.max_batch_size
        or max_seqlen > cg_cache.max_seqlen
    ):  # Invalidate the cg_cache
        cg_cache.callables = {}
        cg_cache.mempool = None
        gc.collect()
        cg_cache.device, cg_cache.dtype = device, dtype
        cg_cache.max_batch_size, cg_cache.max_seqlen = batch_size, max_seqlen
        lengths_per_sample = torch.full(
            (batch_size,), seqlen_og, dtype=torch.int32, device=device
        )
        cg_cache.inference_params = InferenceParams(
            max_sequence_len=max_seqlen,
            max_batch_size=batch_size,
            fused_ft_kernel=True,
            lengths_per_sample=lengths_per_sample,
        )
        cg_cache.inference_params.sequence_len_offset = seqlen_og
        cg_cache.inference_params.key_value_memory_dict = (
            inference_params.key_value_memory_dict
        )
        cg_cache.mempool = torch.cuda.graphs.graph_pool_handle()
    for s_type in range(
        seqlen_to_seqlen_type(seqlen_og), seqlen_to_seqlen_type(max_seqlen) + 1
    ):
        if (batch_size, s_type) not in cg_cache.callables:
            max_seqlen_ = min(
                max(seqlen_og, seqlen_type_to_max_seqlen(s_type)), max_seqlen
            )
            assert cg_cache.inference_params is not None
            cg_cache.callables[batch_size, s_type] = capture_graph(
                model,
                cg_cache.inference_params,
                batch_size,
                max_seqlen_,
                mempool=cg_cache.mempool,
                n_warmups=n_warmups,
            )

    def dispatch(
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        seqlen: int,
    ) -> Any:
        batch_size = input_ids.shape[0]
        assert cg_cache is not None
        return cg_cache.callables[batch_size, seqlen_to_seqlen_type(seqlen)](
            input_ids, position_ids, seqlen
        )

    assert cg_cache.inference_params is not None
    cg_cache.run = dispatch
    cg_cache.inference_params.reset()  # Reset so it's not confusing
    return cg_cache


def capture_graph(
    model: MegatronModule,
    inference_params: InferenceParams,
    batch_size: int,
    max_seqlen: int,
    mempool: Optional[torch.Tensor] = None,
    n_warmups: int = 2,
) -> Callable[[torch.Tensor, torch.Tensor, int], Optional[Dict[str, Any]]]:
    device = next(iter(model.parameters())).device
    input_ids = torch.full((batch_size, 1), 0, dtype=torch.long, device=device)
    position_ids = torch.full((batch_size, 1), 0, dtype=torch.long, device=device)
    sequence_len_offset_og = inference_params.sequence_len_offset
    # TD [2023-04-14]: important for correctness of the FT's attention kernel, as seqlen_cpu is
    # used to determine the size of smem. Hence seqlen_cpu must be >= lengths_per_sample.
    inference_params.sequence_len_offset = max_seqlen - 1
    inference_params.lengths_per_sample[:] = max_seqlen - 1

    forward_step = ForwardStep(model, batch_size, max_seqlen, inference_params)
    # Warmup before capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            outputs = forward_step(input_ids, position_ids)
        s.synchronize()
        # This might be needed for correctness if we run with NCCL_GRAPH_MIXING_SUPPORT=0,
        # which requires that graph launch and non-captured launch to not overlap (I think,
        # that's how I interpret the documentation). I'm not sure if this is required.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    torch.cuda.current_stream().wait_stream(s)
    # Captures the graph
    # To allow capture, automatically sets a side stream as the current stream in the context
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        outputs = forward_step(input_ids, position_ids)

    def run(
        new_input_ids: torch.Tensor, new_position_ids: torch.Tensor, seqlen: int
    ) -> Optional[Dict[str, Any]]:
        inference_params.lengths_per_sample[:] = seqlen
        input_ids.copy_(new_input_ids)
        position_ids.copy_(new_position_ids)
        graph.replay()
        return outputs

    inference_params.sequence_len_offset = sequence_len_offset_og
    return run


def allocate_kv_cache(
    batch_size: int,
    seqlen: int,
    num_layers: int,
    num_heads_per_partition: int,
    head_dim: int,
    fused_ft_kernel: bool = False,
    dtype: torch.dtype = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    kv_cache = {}
    for i in range(1, num_layers + 1):
        if not fused_ft_kernel:
            k_cache = torch.empty(
                seqlen,
                batch_size,
                num_heads_per_partition,
                head_dim,
                dtype=dtype,
                device=device,
            )
            v_cache = torch.empty(
                seqlen,
                batch_size,
                num_heads_per_partition,
                head_dim,
                dtype=dtype,
                device=device,
            )
        else:
            dtype = torch.bfloat16 if dtype is None else dtype
            assert dtype in [torch.float16, torch.bfloat16, torch.float32]
            packsize = 4 if dtype == torch.float32 else 8
            assert head_dim % packsize == 0
            k_cache = torch.empty(
                batch_size,
                num_heads_per_partition,
                head_dim // packsize,
                seqlen,
                packsize,
                dtype=dtype,
                device=device,
            )
            v_cache = torch.empty(
                batch_size,
                num_heads_per_partition,
                seqlen,
                head_dim,
                dtype=dtype,
                device=device,
            )
        kv_cache[i] = (k_cache, v_cache)
    return kv_cache


def setup_model(
    model: MegatronModule,
    model_architecture: str,
    use_inference_kv_cache: bool,
    use_fused_ft_kernel: bool,
    use_cuda_graph: bool,
    max_batch_size: int,
    max_seq_len: int,
) -> Optional[InferenceParams]:
    if use_inference_kv_cache:
        print(f"Initializing KV cache with {max_batch_size=}, {max_seq_len=}")
        inference_params = InferenceParams(
            max_batch_size=max_batch_size,
            max_sequence_len=max_seq_len,
            fused_ft_kernel=use_fused_ft_kernel,
        )
    else:
        print("Disabling KV cache")
        inference_params = None

    example_param = next(iter(model.parameters()))
    layers = (
        model.module.language_model.encoder.layers
        if model_architecture in ["GPTModel"]
        else model.module.decoder.decoder.layers
    )
    attn_layer = layers[0].self_attention

    if inference_params is not None:
        inference_params.key_value_memory_dict = allocate_kv_cache(
            max_batch_size,
            max_seq_len,
            num_layers=len(layers),
            num_heads_per_partition=attn_layer.num_attention_heads_per_partition,
            head_dim=attn_layer.hidden_size_per_attention_head,
            fused_ft_kernel=inference_params.fused_ft_kernel,
            dtype=example_param.dtype,
            device=example_param.device,
        )
        if use_cuda_graph:
            # Capture the graph for the generation step
            model._cg_cache = None
            # Go from large to small batch sizes so we don't need to invalidate the graph cache
            for batch_size in reversed(range(1, max_batch_size + 1)):
                model._cg_cache = update_graph_cache(
                    model, model._cg_cache, inference_params, batch_size, 1, max_seq_len
                )

    return inference_params


def print_request_json() -> None:
    json_to_dump = copy.deepcopy(request.get_json())
    print(json.dumps(json_to_dump), flush=True)


class MegatronGenerate(Resource):  # type: ignore
    def __init__(
        self,
        model: MegatronModule,
        inference_params: InferenceParams,
        params_dtype: torch.dtype,
        max_position_embeddings: int,
        termination_id: int,
        tokenizer: AbstractTokenizer,
    ):
        self.model = model
        if inference_params is not None:
            inference_params.reset()
        self.inference_params = inference_params
        self.params_dtype = params_dtype
        self.max_position_embeddings = max_position_embeddings
        self.termination_id = termination_id
        self.tokenizer = tokenizer

    @staticmethod
    def send_do_generate() -> None:
        choice = torch.cuda.LongTensor([GENERATE_NUM])
        torch.distributed.broadcast(choice, 0)

    @cross_origin()
    def put(self) -> Union[Tuple[str, int], str]:
        print("request IP: " + str(request.remote_addr))
        print_request_json()
        print("current time: ", datetime.datetime.now())

        if not "prompts" in request.get_json():
            return "prompts argument required", 400

        if "max_len" in request.get_json():
            return "max_len is no longer used.  Replace with tokens_to_generate", 400

        if "sentences" in request.get_json():
            return "sentences is no longer used.  Replace with prompts", 400

        prompts = request.get_json()["prompts"]

        if type(prompts) != list:
            prompts = [prompts]

        if len(prompts) > 128:
            return "Maximum number of prompts is 128", 400

        max_tokens_to_generate = (
            64  # Choosing hopefully sane default.  Full sequence is slow
        )
        if "tokens_to_generate" in request.get_json():
            max_tokens_to_generate = request.get_json()["tokens_to_generate"]
            if not isinstance(max_tokens_to_generate, int):
                return "tokens_to_generate must be an integer greater than 0"
            if max_tokens_to_generate < 0:
                return (
                    "tokens_to_generate must be an integer greater than or equal to 0"
                )

        logprobs = False
        if "logprobs" in request.get_json():
            logprobs = request.get_json()["logprobs"]
            if not isinstance(logprobs, bool):
                return "logprobs must be a boolean value"

        if max_tokens_to_generate == 0 and not logprobs:
            return "tokens_to_generate=0 implies logprobs should be True"

        temperature = 1.0
        if "temperature" in request.get_json():
            temperature = request.get_json()["temperature"]
            if not isinstance(temperature, numbers.Number):
                return (
                    "temperature must be a positive number less than or equal to 100.0"
                )
            temperature = float(temperature)
            if not (0.0 < temperature <= 100.0):
                return (
                    "temperature must be a positive number less than or equal to 100.0"
                )

        top_k = 0
        if "top_k" in request.get_json():
            top_k = request.get_json()["top_k"]
            if not isinstance(top_k, int):
                return "top_k must be an integer equal to or greater than 0 and less than or equal to 1000"
            if not (0 <= top_k <= 1000):
                return "top_k must be equal to or greater than 0 and less than or equal to 1000"

        top_p = 0.0
        if "top_p" in request.get_json():
            top_p = request.get_json()["top_p"]
            if not isinstance(top_p, numbers.Number):
                return "top_p must be a positive float less than or equal to 1.0"
            top_p = float(top_p)
            if top_p > 0.0 and top_k > 0.0:
                return "cannot set both top-k and top-p samplings."
            if not (0 <= top_p <= 1.0):
                return "top_p must be less than or equal to 1.0"

        add_BOS = False
        if "add_BOS" in request.get_json():
            add_BOS = request.get_json()["add_BOS"]
            if not isinstance(add_BOS, bool):
                return "add_BOS must be a boolean value"

        random_seed = -1
        if "random_seed" in request.get_json():
            random_seed = request.get_json()["random_seed"]
            if not isinstance(random_seed, int):
                return "random_seed must be integer"
            if random_seed < 0:
                return "random_seed must be a positive integer"

        no_log = False
        if "no_log" in request.get_json():
            no_log = request.get_json()["no_log"]
            if not isinstance(no_log, bool):
                return "no_log must be a boolean value"

        if any(len(prompt) == 0 for prompt in prompts) and not add_BOS:
            return "Empty prompts require add_BOS=true"

        with lock:  # Need to get lock to keep multiple threads from hitting code
            if not no_log:
                print("request IP: " + str(request.remote_addr))
                print_request_json()
                start_time = datetime.datetime.now()
                print("start time: ", start_time, flush=True)

            try:
                MegatronGenerate.send_do_generate()  # Tell other ranks we're doing generate
                retval = generate_and_post_process(
                    self.model,
                    params_dtype=self.params_dtype,
                    max_position_embeddings=self.max_position_embeddings,
                    termination_id=self.termination_id,
                    tokenizer=self.tokenizer,
                    prompts=[[prompt] for prompt in prompts],
                    max_tokens_to_generate=max_tokens_to_generate,
                    inference_params=self.inference_params,
                    return_output_log_probs=logprobs,
                    top_k_sampling=top_k,
                    top_p_sampling=top_p,
                    temperature=temperature,
                    add_BOS=add_BOS,
                    random_seed=random_seed,
                )
                assert retval is not None
                (
                    response,
                    response_logprobs,
                    _,
                    generations,
                    _,
                    human_readable_tokens,
                ) = retval

                end_time = datetime.datetime.now()
                print("Query latency: ", end_time - start_time, flush=True)
                output = {
                    "text": response,
                    "logprobs": response_logprobs,
                    "generations": generations,
                    "human_readable_tokens": human_readable_tokens,
                }
                return jsonify(output)  # type: ignore

            except ValueError as ve:
                traceback.print_exc()
                return str(ve.args[0])


class Healthcheck(Resource):  # type: ignore
    def get(self) -> Tuple[Dict[str, bool], int, Dict[str, str]]:
        return {"success": True}, 200, {"ContentType": "application/json"}


class MegatronServer(object):
    def __init__(
        self,
        model: MegatronModule,
        inference_params: InferenceParams,
        params_dtype: torch.dtype,
        max_position_embeddings: int,
        termination_id: int,
        tokenizer: AbstractTokenizer,
        port: int = 5000,
    ):
        self.port = port
        self.app = Flask(__name__, static_url_path="")
        _ = CORS(self.app)
        self.app.config["CORS_HEADERS"] = "Content-Type"
        api = Api(self.app)
        api.add_resource(
            MegatronGenerate,
            "/api",
            resource_class_args=[
                model,
                inference_params,
                params_dtype,
                max_position_embeddings,
                termination_id,
                tokenizer,
            ],
        )
        api.add_resource(Healthcheck, "/healthz")

    def run(self, url: str) -> None:
        self.app.run(url, port=self.port, threaded=True, debug=False)

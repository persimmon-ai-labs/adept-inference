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

"""Input/output checkpointing."""

# pylint: disable=consider-using-f-string, invalid-name

import argparse
import os
import random
import shutil
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import requests

from megatron import mpu, update_num_microbatches
from megatron.model.module import MegatronModule
from megatron.model.utils import validate_replicated_parameters

from .global_vars import get_args
from .utils import calculate_num_params_and_params_l2_norm, print_rank_0, unwrap_model


MegatronOptimizerType = Any


_CHECKPOINT_VERSION = None

TEXT_EVAL_ENDPOINT = (
    "http://experiments.research.adept.ai/run/fine_tune_and_text_eval_flow"
)


def _get_checkpoint_specific_args() -> List[str]:
    """
    Returns a list of arguments that are specific to the 'trained' checkpoint.

    These mostly are all architecture specific args which cannot be changed
    for an already-trained checkpoint.

    Returns:
        List[str]: List of argument names.
    """

    args_specific_to_checkpoint = [
        "num_layers",
        "hidden_size",
        "ffn_hidden_size",
        "seq_length",
        "num_attention_heads",
        "kv_channels",
        "max_position_embeddings",
        "pos_emb",
        "rotary_pct",
        "rotary_emb_base",
        "untie_embeddings",
        "sq_relu",
        "bf16",
        "fp16",
        "tokenizer_type",
        "padded_vocab_size",
        "vocab_file",
        "merge_file",
        "patch_dim_h",
        "patch_dim_w",
        "encoder_num_layers",
        "bias_gelu_fusion",
        "bias_dropout_fusion",
        "sample_temp_type",
        "sample_temp_max",
        "sample_temp_min",
        "use_quantizer_after",
        "qk_layernorm",
    ]
    return args_specific_to_checkpoint


def _get_sentence_piece_tokenizer_file(
    checkpoint_args: argparse.Namespace,
) -> Optional[str]:
    """Get the sentence piece tokenizer file from the checkpoint args.

    This is a special case for the 16B-2K flagship model because
    we want to allow the checkpoints trained on AWS to be loaded
    on OCI. All other checkpoints should have correct sp_model_file.
    """

    sp_model_file = getattr(checkpoint_args, "sp_model_file", None)
    fsx_prefix = "/fsx/sentence_piece_vocabularies/vocab_new/"
    if sp_model_file is not None and sp_model_file.startswith(fsx_prefix):
        sp_model_file = sp_model_file.replace(fsx_prefix, "/mnt/weka/vocabularies/")
    # For backwards compatibility
    old_container_prefix = "/vocabularies/"
    if sp_model_file is not None and sp_model_file.startswith(old_container_prefix):
        sp_model_file = sp_model_file.replace(
            old_container_prefix, "/mnt/weka/vocabularies/"
        )
    return sp_model_file


def compare_or_set_checkpoint_args(
    checkpoint_args: argparse.Namespace,
    to_compare: bool,
    to_set: bool,
    args: argparse.Namespace,
) -> None:
    """Ensure fixed arguments for a model are the same for the input
    arguments and the one retrieved from checkpoint.

    """

    def _compare_or_set(arg_name: str, old_arg_name: Optional[str] = None) -> None:
        checkpoint_value = getattr(checkpoint_args, arg_name, None)
        if checkpoint_value is None and old_arg_name is not None:
            checkpoint_value = getattr(checkpoint_args, old_arg_name, None)

        if arg_name == "sp_model_file":
            checkpoint_value = _get_sentence_piece_tokenizer_file(checkpoint_args)

        if checkpoint_value is None:
            # If the checkpoint does not have the argument, then we assume
            # it is the same as the current argument.
            # Note: We end up skipping if the checkpoint value was actually None.
            return

        if to_compare:
            args_value = getattr(args, arg_name)
            error_message = (
                "{} value from checkpoint ({}) is not equal to the "
                "input argument value ({}).".format(
                    arg_name, checkpoint_value, args_value
                )
            )
            assert checkpoint_value == args_value, error_message
        if to_set:
            print_rank_0(f"Setting {arg_name} to {checkpoint_value} from checkpoint")
            setattr(args, arg_name, checkpoint_value)

    # Check arguments that are specific to the checkpoint.
    for arg_name in _get_checkpoint_specific_args():
        _compare_or_set(arg_name)

    if args and args.data_parallel_random_init:
        _compare_or_set("data_parallel_random_init")
    _compare_or_set("tensor_model_parallel_size")
    _compare_or_set("pipeline_model_parallel_size")


def ensure_directory_exists(filename: str) -> None:
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_checkpoint_names(
    checkpoints_path: str,
    iteration: int,
    use_distributed_optimizer: bool,
    pipeline_parallel: Optional[bool] = None,
    tensor_rank: Optional[int] = None,
    pipeline_rank: Optional[int] = None,
) -> Tuple[str, str, str]:
    """Determine the directory name for this rank's checkpoint."""
    directory = "iter_{:07d}".format(iteration)

    # Use both the tensor and pipeline MP rank.
    if pipeline_parallel is None:
        pipeline_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
    if tensor_rank is None:
        tensor_rank = mpu.get_tensor_model_parallel_rank()
    if pipeline_rank is None:
        pipeline_rank = mpu.get_pipeline_model_parallel_rank()

    # Use both the tensor and pipeline MP rank. If using the distributed
    # optimizer, then the optimizer's path must additionally include the
    # data parallel rank.
    base_path = os.path.join(checkpoints_path, directory)
    if not pipeline_parallel:
        common_path = os.path.join(base_path, f"mp_rank_{tensor_rank:02d}")
    else:
        common_path = os.path.join(
            base_path, f"mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}"
        )

    if use_distributed_optimizer:
        model_name = os.path.join(common_path, "model_rng.pt")
        optim_name = os.path.join(
            common_path + "_%03d" % mpu.get_data_parallel_rank(), "optim.pt"
        )
    else:
        model_name = optim_name = os.path.join(common_path, "model_optim_rng.pt")
    return model_name, optim_name, base_path


def find_checkpoint_rank_0(
    checkpoints_path: str, iteration: int, use_distributed_optimizer: bool
) -> Tuple[str, str]:
    """Finds the checkpoint for rank 0 without knowing if we are using
    pipeline parallelism or not.

    Since the checkpoint naming scheme changes if pipeline parallelism
    is present, we need to look for both naming schemes if we don't
    know if the checkpoint has pipeline parallelism.

    """

    # Look for checkpoint with no pipelining
    model_name, optim_name, _ = get_checkpoint_names(
        checkpoints_path,
        iteration,
        use_distributed_optimizer,
        pipeline_parallel=False,
        tensor_rank=0,
        pipeline_rank=0,
    )
    if os.path.isfile(model_name):
        return model_name, optim_name

    # Look for checkpoint with pipelining
    model_name, optim_name, _ = get_checkpoint_names(
        checkpoints_path,
        iteration,
        use_distributed_optimizer,
        pipeline_parallel=True,
        tensor_rank=0,
        pipeline_rank=0,
    )
    if os.path.isfile(model_name):
        return model_name, optim_name

    raise FileNotFoundError(
        f"Could not find checkpoint for rank 0 at {checkpoints_path}"
    )


def get_checkpoint_tracker_filename(checkpoints_path: str) -> str:
    """Tracker file records the latest checkpoint reached during training.
    When continuing a training job, we usually want to start from here.
    """
    return os.path.join(checkpoints_path, "latest_checkpointed_iteration.txt")


def read_metadata(tracker_filename: str) -> int:
    # Read the tracker file and set the iteration
    iteration: int = 0
    max_iter: int = -1
    with open(tracker_filename, "r") as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            print_rank_0(
                "ERROR: Invalid metadata file {}. Exiting".format(tracker_filename)
            )
            sys.exit()
    # NOTE: it isn't entirely clear why a metadata file with a iteration of 0 can't exist
    # cleanup later
    assert iteration > 0, "error parsing metadata file {}".format(tracker_filename)

    # NOTE: also very unclear how the different ranks would ever load different values either!!
    # Get the max iteration retrieved across the ranks.
    if torch.distributed.is_initialized():
        iters_cuda = torch.cuda.LongTensor([iteration])
        torch.distributed.all_reduce(iters_cuda, op=torch.distributed.ReduceOp.MAX)
        max_iter = iters_cuda[0].item()

        # We should now have all the same iteration.
        # If not, print a warning and chose the maximum
        # iteration across all ranks.
        if iteration != max_iter:
            print(
                "WARNING: on rank {} found iteration {} in the "
                "metadata while max iteration across the ranks "
                "is {}, replacing it with max iteration.".format(
                    torch.distributed.get_rank(), iteration, max_iter
                ),
                flush=True,
            )
    else:
        # When loading a checkpoint outside of training (for example,
        # when editing it), we might not have torch distributed
        # initialized, in this case, just assume we have the latest
        max_iter = iteration

    assert max_iter >= 0
    return max_iter


def get_rng_state() -> List[Optional[Dict[str, Any]]]:
    """collect rng state across data parallel ranks"""
    args = get_args()
    rng_state = {
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "rng_tracker_states": mpu.get_cuda_rng_tracker().get_states(),
    }

    if (
        torch.distributed.is_initialized()
        and mpu.get_data_parallel_world_size() > 1
        and args.data_parallel_random_init
    ):
        rng_state_list: List[Optional[Dict[str, Any]]] = [
            None for i in range(mpu.get_data_parallel_world_size())
        ]
        torch.distributed.all_gather_object(
            rng_state_list, rng_state, group=mpu.get_data_parallel_group()
        )
    else:
        rng_state_list = [rng_state]

    return rng_state_list


def load_checkpoint_from_path(
    model_checkpoint_name: str,
    optim_checkpoint_name: str,
    use_distributed_optimizer: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Load the checkpoint.
    try:
        # checkpoint_size = Path(model_checkpoint_name).stat().st_size
        # if torch.distributed.is_initialized():
        #     checkpoint_sizes = [
        #         None,
        #     ] * mpu.get_tensor_model_parallel_world_size()
        #     torch.distributed.all_gather_object(
        #         checkpoint_sizes, checkpoint_size, group=mpu.get_tensor_model_parallel_group()
        #     )
        #     valid = all(c == checkpoint_sizes[0] for c in checkpoint_sizes[1:])
        #     if not valid and torch.distributed.get_rank() == mpu.get_tensor_model_parallel_src_rank():
        #         print(f"Not all Tensor Parallel checkpoints are the same size!")
        #     valid_tensor = torch.cuda.LongTensor([valid])
        #     torch.distributed.all_reduce(valid_tensor, op=torch.distributed.ReduceOp.MIN)
        #     all_valid = bool(valid_tensor[0].item())
        #     if not all_valid:
        #         print_rank_0("Invalid checkpoint! Aborting training.")
        #         sys.exit(1)

        # else:
        #     print_rank_0("WARNING: Distributed is not initialized, cannot verify checkpoint validity.")
        model_state_dict = torch.load(model_checkpoint_name, map_location="cpu")
        if use_distributed_optimizer:
            optim_state_dict = torch.load(optim_checkpoint_name, map_location="cpu")
        else:
            optim_state_dict = model_state_dict
    except BaseException as e:
        print_rank_0("could not load the checkpoint")
        print_rank_0(e)
        sys.exit()

    return model_state_dict, optim_state_dict


def _load_base_checkpoint(
    load_dir: str,
    use_distributed_optimizer: bool,
    rank0: bool = False,
    load_iteration: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load the base state_dict from the given directory

    If rank0 is true, just loads rank 0 checkpoint, ignoring arguments.
    """

    # If iteration is specified, set that
    if load_iteration:
        iteration = load_iteration
    else:
        # Read the tracker file
        tracker_filename = get_checkpoint_tracker_filename(load_dir)

        # If no tracker file, return nothing
        if not os.path.isfile(tracker_filename):
            raise FileNotFoundError(f"Could not find metadata file: {tracker_filename}")

        # set the iteration
        iteration = read_metadata(tracker_filename)

    # Checkpoint.
    if rank0:
        checkpoint_names = find_checkpoint_rank_0(
            load_dir, iteration, use_distributed_optimizer
        )
    else:
        model_name, optim_name, _ = get_checkpoint_names(
            load_dir, iteration, use_distributed_optimizer
        )
        checkpoint_names = (model_name, optim_name)
        print_rank_0(f" loading checkpoint from {load_dir} at iteration {iteration}")

    model_checkpoint_name, optim_checkpoint_name = checkpoint_names

    model_state_dict, optim_state_dict = load_checkpoint_from_path(
        model_checkpoint_name, optim_checkpoint_name, use_distributed_optimizer
    )

    return model_state_dict, optim_state_dict


def load_args_from_checkpoint(
    args: argparse.Namespace, load_arg: str = "load"
) -> argparse.Namespace:
    """Set required arguments from the checkpoint specified in the arguments.

    Will overwrite arguments that have a non-None default value, but
    will leave any arguments that default to None as set.

    Returns the same args NameSpace with the new values added/updated.

    If no checkpoint is specified in args, or if the checkpoint is
    there but invalid, the arguments will not be modified

    """
    load_dir = getattr(args, load_arg)

    if load_dir is None:
        print_rank_0("No load directory specified, using provided arguments.")
        return args

    model_state_dict, _ = _load_base_checkpoint(
        load_dir, use_distributed_optimizer=args.use_distributed_optimizer, rank0=True
    )

    # For args we only care about model state dict
    state_dict = model_state_dict

    if not state_dict:
        print_rank_0(
            "Checkpoint not found to provide arguments, using provided arguments."
        )
        return args

    if "args" not in state_dict:
        print_rank_0(
            "Checkpoint provided does not have arguments saved, using provided arguments."
        )
        return args

    checkpoint_args = state_dict["args"]
    args.iteration = state_dict["iteration"]

    compare_or_set_checkpoint_args(
        checkpoint_args, to_compare=False, to_set=True, args=args
    )

    return args


def load_state_dicts_and_update_args(
    load_arg: str = "load",
) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
    """Loads raw state dicts from checkpoint and updates starting iteration/samples in args.

    Returns:
        - iteration: iteration index to continue training from
        - model_state_dict: dictionary with raw model weights loaded from checkpoint
        - optim_state_dict: dictionary with raw optimizer state weights loaded from checkpoint
    """
    args = get_args()
    load_dir = getattr(args, load_arg)

    model_state_dict, optim_state_dict = _load_base_checkpoint(
        load_dir,
        use_distributed_optimizer=args.use_distributed_optimizer,
        rank0=False,
        load_iteration=args.load_iteration,
    )

    # Check arguments.
    if "args" in model_state_dict:
        checkpoint_args = model_state_dict["args"]
        compare_or_set_checkpoint_args(
            checkpoint_args, to_compare=True, to_set=False, args=args
        )
        if load_arg == "load":
            args.consumed_train_samples = getattr(
                checkpoint_args, "consumed_train_samples", 0
            )
            update_num_microbatches(consumed_samples=args.consumed_train_samples)
            args.consumed_valid_samples = getattr(
                checkpoint_args, "consumed_valid_samples", 0
            )

    # Don't load the VQ-VAE's rng state.
    # NOTE: unclear if anything is actually implementing the logic of the above comment
    if load_arg == "load":
        rng_states(model_state_dict)

    # Set iteration.
    if args.finetune:
        iteration = 0
        args.consumed_train_samples = 0
        args.consumed_valid_samples = 0
    else:
        try:
            iteration = model_state_dict["iteration"]
        except KeyError:
            print_rank_0(
                "A metadata file exists but unable to load iteration from checkpoint {}, exiting".format(
                    load_dir
                )
            )
            sys.exit()
    return iteration, model_state_dict, optim_state_dict


def update_model_from_state_dict(
    model: List[MegatronModule], model_state_dict: Dict[str, Any], strict: bool = True
) -> None:
    """Takes in raw model weights loaded from the last checkpoint and loads into model."""
    print_rank_0("starting update_model_from_state_dict")
    if len(model) == 1:
        model[0].load_state_dict(model_state_dict["model"], strict=strict)
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            model[i].load_state_dict(model_state_dict["model%d" % i], strict=strict)


def rng_states(model_state_dict: Dict[str, Any]) -> None:
    args = get_args()
    if not args.finetune and not args.no_load_rng:
        try:
            if "rng_state" in model_state_dict:
                # access rng_state for data parallel rank
                if args.data_parallel_random_init:
                    rng_state = model_state_dict["rng_state"][
                        mpu.get_data_parallel_rank()
                    ]
                else:
                    rng_state = model_state_dict["rng_state"][0]
                random.setstate(rng_state["random_rng_state"])
                np.random.set_state(rng_state["np_rng_state"])
                torch.set_rng_state(rng_state["torch_rng_state"])
                torch.cuda.set_rng_state(rng_state["cuda_rng_state"])
                # Check for empty states array
                if not rng_state["rng_tracker_states"]:
                    raise KeyError
                mpu.get_cuda_rng_tracker().set_states(rng_state["rng_tracker_states"])
            else:  # backward compatability
                random.setstate(model_state_dict["random_rng_state"])
                np.random.set_state(model_state_dict["np_rng_state"])
                torch.set_rng_state(model_state_dict["torch_rng_state"])
                torch.cuda.set_rng_state(model_state_dict["cuda_rng_state"])
                # Check for empty states array
                if not model_state_dict["rng_tracker_states"]:
                    raise KeyError
                mpu.get_cuda_rng_tracker().set_states(
                    model_state_dict["rng_tracker_states"]
                )
        except KeyError:
            print_rank_0(
                "Unable to load rng state from checkpoint. "
                "Specify --no-load-rng or --finetune to prevent "
                "attempting to load the rng state, "
                "exiting ..."
            )
            sys.exit()


def update_model_and_optim_from_loaded_data(
    iteration: int,
    model_state_dict: Dict[str, Any],
    optim_state_dict: Dict[str, Any],
    model: List[MegatronModule],
    optimizer: MegatronOptimizerType,
    opt_param_scheduler: None,  # OptimizerParamScheduler,
    strict: bool = True,
    log_norms: bool = True,
) -> None:
    """Take in raw state dicts for model and optimizer and load them into actual model/optimizer/scheduler.

    Args:
        strict: whether to strictly enforce that the keys in the state dict of the checkpoint match the names
            of parameters and buffers in model.
    """
    print_rank_0("starting update_model_and_optim_from_loaded_data")
    args = get_args()
    update_model_from_state_dict(model, model_state_dict, strict)
    # unused in inference mode:
    # update_optimizer_and_scheduler_from_state_dict(optimizer, optim_state_dict, opt_param_scheduler)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(
        f"  successfully loaded checkpoint from {args.load} "
        f"at iteration {iteration}"
    )


def load_checkpoint(
    model: List[MegatronModule],
    optimizer: MegatronOptimizerType,
    opt_param_scheduler: None,  # OptimizerParamScheduler,
    load_arg: str = "load",
    strict: bool = True,
    log_norms: bool = True,
) -> int:
    """
    This function loads a model checkpoint, returning the iteration number.
    Args:
    - model: model object being loaded from checkpoint
    - optimizer: optimizer object being loaded from checkpoint
    - opt_param_scheduler: scheduler being loaded from checkpoint
    - load_arg
    - strict: Whether to strictly enforce name matching between checkpoint's
    state_dict and model's parameters and buffers. Default true.

    Returns:
    Iteration index of the latest checkpoint loaded.
    """
    iteration, model_state_dict, optim_state_dict = load_state_dicts_and_update_args(
        load_arg
    )
    update_model_and_optim_from_loaded_data(
        iteration,
        model_state_dict,
        optim_state_dict,
        model,
        optimizer,
        opt_param_scheduler,
        strict,
        log_norms,
    )
    return iteration

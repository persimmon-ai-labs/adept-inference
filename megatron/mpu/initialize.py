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


"""Model and data parallel groups."""

import os
import sys
import torch
import warnings

from .utils import ensure_divisibility


# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Embedding group.
_EMBEDDING_GROUP = None
# Position embedding group.
_POSITION_EMBEDDING_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None


def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None


def initialize_model_parallel(
    tensor_model_parallel_size_=1,
    pipeline_model_parallel_size_=1,
    virtual_pipeline_model_parallel_size_=None,
    pipeline_model_parallel_split_rank_=None,
    default_backend=None,
    p2p_backend=None,
    cross_block_networking=False,
    share_word_embeddings=False,
    force_socket_networking=False,
):
    """
    Initialize model data parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model parallelism.
        virtual_pipeline_model_parallel_size: number of virtual stages (interleaved
                                              pipeline).
        pipeline_model_parallel_split_rank: for models with both encoder and decoder,
                                            rank in pipeline with split point.
        default_backend: Backend of process groups except for pipeline parallel ones.
            If :obj:`None`, the backend specified in `torch.distributed.init_process_group` will be used.
        p2p_backend: Backend of process groups for pipeline model parallel.
            If :obj:`None`, the backend specified in `torch.distributed.init_process_group` will be used.
        cross_block_networking: Use Socket networking where necessary to ensure cross block communication
            doesn't use IB networking which doesn't span blocks.
        share_word_embeddings: True if the input and output embeddings are to be kept in sync.
        force_socket_networking: Use Socket networking for all communication.


    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    if torch.distributed.get_rank() == 0:
        print("> initializing tensor model parallel with size {}".format(tensor_model_parallel_size_))
        print("> initializing pipeline model parallel with size {}".format(pipeline_model_parallel_size_))
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    assert default_backend is None or default_backend in ("nccl", "ucc")
    assert p2p_backend is None or p2p_backend in ("nccl", "ucc")
    if "ucc" in (default_backend, p2p_backend):
        check_torch_ucc_availability()
    if default_backend == "ucc":
        warnings.warn("The UCC's functionality as `default_backend` is not well verified", ExperimentalWarning)

    world_size = torch.distributed.get_world_size()
    tensor_model_parallel_size = min(tensor_model_parallel_size_, world_size)
    pipeline_model_parallel_size = min(pipeline_model_parallel_size_, world_size)
    ensure_divisibility(world_size, tensor_model_parallel_size * pipeline_model_parallel_size)
    data_parallel_size = world_size // (tensor_model_parallel_size * pipeline_model_parallel_size)

    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size
    num_data_parallel_groups = world_size // data_parallel_size

    if virtual_pipeline_model_parallel_size_ is not None:
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size_

    if pipeline_model_parallel_split_rank_ is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank_

    rank = torch.distributed.get_rank()

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GLOBAL_RANKS
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
    all_data_parallel_group_ranks = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_model_parallel_size):
            ranks = range(start_rank + j, end_rank, tensor_model_parallel_size)
            all_data_parallel_group_ranks.append(list(ranks))
            group = torch.distributed.new_group(ranks, backend=default_backend)
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group
                _DATA_PARALLEL_GLOBAL_RANKS = ranks

    if force_socket_networking:
        os.environ["NCCL_NET"] = "Socket"
    else:
        os.environ["NCCL_NET"] = "IB"
    force_communicator_creation(
        rank=get_data_parallel_rank(),
        world_size=get_data_parallel_world_size(),
        all_reduce=True,
        all_gather=True,
        group=get_data_parallel_group(),
    )

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, "model parallel group is already initialized"
    for i in range(data_parallel_size):
        ranks = [data_parallel_group_ranks[i] for data_parallel_group_ranks in all_data_parallel_group_ranks]
        group = torch.distributed.new_group(ranks, backend=default_backend)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group

    if cross_block_networking or force_socket_networking:
        os.environ["NCCL_NET"] = "Socket"
    else:
        os.environ["NCCL_NET"] = "IB"
    force_communicator_creation(all_reduce=True, group=get_model_parallel_group())

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, "tensor model parallel group is already initialized"
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        group = torch.distributed.new_group(ranks, backend=default_backend)
        if rank in ranks:
            # torch.distributed.barrier(group)
            _TENSOR_MODEL_PARALLEL_GROUP = group

    if force_socket_networking:
        os.environ["NCCL_NET"] = "Socket"
    else:
        os.environ["NCCL_NET"] = "IB"
    force_communicator_creation(
        rank=get_tensor_model_parallel_rank(),
        src_rank=get_tensor_model_parallel_src_rank(),
        world_size=get_tensor_model_parallel_world_size(),
        all_reduce=True,
        all_gather=True,
        broadcast=True,
        group=get_tensor_model_parallel_group(),
    )

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert _PIPELINE_MODEL_PARALLEL_GROUP is None, "pipeline model parallel group is already initialized"
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, "embedding group is already initialized"
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, "position embedding group is already initialized"
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(ranks, backend=p2p_backend)
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
            if pipeline_model_parallel_split_rank_ is not None:
                if ranks[pipeline_model_parallel_split_rank_] not in embedding_ranks:
                    embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank_], ranks[-1]]
                if ranks[pipeline_model_parallel_split_rank_] not in position_embedding_ranks:
                    position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank_]]
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks

        group = torch.distributed.new_group(embedding_ranks, backend=default_backend)
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group
        if rank in ranks:
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks

        group = torch.distributed.new_group(position_embedding_ranks, backend=default_backend)
        if rank in position_embedding_ranks:
            _POSITION_EMBEDDING_GROUP = group
        if rank in ranks:
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

    if cross_block_networking or force_socket_networking:
        os.environ["NCCL_NET"] = "Socket"
    else:
        os.environ["NCCL_NET"] = "IB"
    force_pipeline_communicator_creation(virtual_pipeline_model_parallel_size_ is not None)

    if share_word_embeddings and is_rank_in_embedding_group(ignore_virtual=True):
        force_communicator_creation(all_reduce=True, group=get_embedding_group())

    if is_rank_in_position_embedding_group() and pipeline_model_parallel_split_rank_ is not None:
        force_communicator_creation(all_reduce=True, group=get_position_embedding_group())
    if force_socket_networking:
        os.environ["NCCL_NET"] = "Socket"
    else:
        os.environ["NCCL_NET"] = "IB"


# Used to warn when the UCC is specified.
class ExperimentalWarning(Warning):
    pass


def check_torch_ucc_availability():
    try:
        import torch_ucc  # NOQA
    except ImportError:
        raise ImportError(
            "UCC backend requires [torch_ucc](https://github.com/facebookresearch/torch_ucc) but not found"
        )


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None or _PIPELINE_MODEL_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, "model parallel group is not initialized"
    return _MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, "intra_layer_model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, "pipeline_model parallel group is not initialized"
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP


def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _EMBEDDING_GROUP is not None, "embedding group is not initialized"
    return _EMBEDDING_GROUP


def get_position_embedding_group():
    """Get the position embedding group the caller rank belongs to."""
    assert _POSITION_EMBEDDING_GROUP is not None, "position embedding group is not initialized"
    return _POSITION_EMBEDDING_GROUP


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor model parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())


def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_rank(rank):
    """Set pipeline model parallel rank."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())


def get_num_layers(
    pipeline_model_parallel_split_rank: int,
    standalone_embedding_stage: bool,
    transformer_pipeline_model_parallel_size: int,
    encoder_num_layers: int,
    decoder_num_layers: int,
    num_layers: int,
    is_encoder_and_decoder_model: bool,
    is_decoder: bool = False,
):
    """Compute the number of transformer layers resident on the current rank."""
    if get_pipeline_model_parallel_world_size() > 1:
        if is_encoder_and_decoder_model:
            assert pipeline_model_parallel_split_rank is not None

            # When a standalone embedding stage is used, a rank is taken from
            # the encoder's ranks, to be used for the encoder's embedding
            # layer. This way, the rank referenced by the 'split rank' remains
            # the same whether or not a standalone embedding stage is used.
            num_ranks_in_encoder = (
                pipeline_model_parallel_split_rank - 1
                if standalone_embedding_stage
                else pipeline_model_parallel_split_rank
            )
            num_ranks_in_decoder = transformer_pipeline_model_parallel_size - num_ranks_in_encoder
            assert (
                encoder_num_layers % num_ranks_in_encoder == 0
            ), "encoder_num_layers (%d) must be divisible by number of ranks given to encoder (%d)" % (
                encoder_num_layers,
                num_ranks_in_encoder,
            )
            assert (
                decoder_num_layers % num_ranks_in_decoder == 0
            ), "decoder_num_layers (%d) must be divisible by number of ranks given to decoder (%d)" % (
                decoder_num_layers,
                num_ranks_in_decoder,
            )
            if is_pipeline_stage_before_split():
                num_layers = (
                    0
                    if standalone_embedding_stage and get_pipeline_model_parallel_rank() == 0
                    else encoder_num_layers // num_ranks_in_encoder
                )
            else:
                num_layers = decoder_num_layers // num_ranks_in_decoder
        else:
            assert num_layers == encoder_num_layers
            assert (
                num_layers % transformer_pipeline_model_parallel_size == 0
            ), "num_layers must be divisible by transformer_pipeline_model_parallel_size"

            # When a standalone embedding stage is used, all transformer layers
            # are divided among pipeline rank >= 1, while on pipeline rank 0,
            # ranks either contain the input embedding layer (virtual pp rank 0),
            # or no layers at all (virtual pp rank >= 1).
            num_layers = (
                0
                if standalone_embedding_stage and get_pipeline_model_parallel_rank() == 0
                else num_layers // transformer_pipeline_model_parallel_size
            )
    else:
        if not is_decoder:
            num_layers = encoder_num_layers
        else:
            num_layers = decoder_num_layers
    return num_layers


def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        if (
            get_virtual_pipeline_model_parallel_world_size() is not None
            and get_virtual_pipeline_model_parallel_rank() != 0
        ):
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        virtual_pipeline_model_parallel_world_size = get_virtual_pipeline_model_parallel_world_size()
        if virtual_pipeline_model_parallel_world_size is not None and get_virtual_pipeline_model_parallel_rank() != (
            virtual_pipeline_model_parallel_world_size - 1
        ):
            return False
    return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)


def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _EMBEDDING_GLOBAL_RANKS
    if ignore_virtual:
        return rank in _EMBEDDING_GLOBAL_RANKS
    if rank in _EMBEDDING_GLOBAL_RANKS:
        if rank == _EMBEDDING_GLOBAL_RANKS[0]:
            return is_pipeline_first_stage(ignore_virtual=False)
        elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
            return is_pipeline_last_stage(ignore_virtual=False)
        else:
            return True
    return False


def is_rank_in_position_embedding_group():
    """Return true if current rank is in position embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    return rank in _POSITION_EMBEDDING_GLOBAL_RANKS


def is_pipeline_stage_before_split(rank=None):
    """Return True if pipeline stage executes encoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank < _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_after_split(rank=None):
    """Return True if pipeline stage executes decoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank >= _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_at_split():
    """Return true if pipeline stage executes decoder block and next
    stage executes encoder block for a model with both encoder and
    decoder."""
    rank = get_pipeline_model_parallel_rank()
    return is_pipeline_stage_before_split(rank) and is_pipeline_stage_after_split(rank + 1)


def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the data parallel group."""
    assert _DATA_PARALLEL_GLOBAL_RANKS is not None, "Data parallel group is not initialized"
    return _DATA_PARALLEL_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_first_rank():
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_next_rank():
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


def force_communicator_creation(
    rank=None,
    src_rank=None,
    world_size=None,
    group=None,
    all_reduce=False,
    all_gather=False,
    barrier=False,
    broadcast=False,
):
    """Will do trivial ops of the requested type so that pytorch is forced to create
    a communicator.  This is useful because we control NCCL parameters related
    to each communicator through changing environment variables.  It is impossible
    to make sure the right env variables are set whereever the first use happens.

    By using this function, we can make sure to set the right env variables right before
    calling it."""
    if barrier:
        torch.distributed.barrier(group=group)
    if all_reduce:
        one_tensor = torch.cuda.FloatTensor([1.0])
        torch.distributed.all_reduce(one_tensor, op=torch.distributed.ReduceOp.SUM, group=group)
    if all_gather:
        assert (
            rank is not None and world_size is not None
        ), "Must supply rank and world_size for all_gather initialization"
        tensor_list = [torch.empty_like(one_tensor) for _ in range(world_size)]
        tensor_list[rank] = one_tensor
        torch.distributed.all_gather(tensor_list, one_tensor, group=group)
    if broadcast:
        one_tensor = torch.cuda.FloatTensor([1.0])
        torch.distributed.broadcast(one_tensor, src_rank, group=group)


def force_pipeline_communicator_creation(ignore_virtual=False):
    """Will do trivial ops for each of the pipeline communications to make sure that
    pytorch is forced to create the require communicators. See docstring for
    force_communicator_creation to understand how this is useful."""
    if get_pipeline_model_parallel_world_size() == 1:
        return
    recv_prev_tensor = torch.cuda.FloatTensor([1.0])
    recv_next_tensor = torch.cuda.FloatTensor([1.0])
    send_prev_tensor = torch.cuda.FloatTensor([1.0])
    send_next_tensor = torch.cuda.FloatTensor([1.0])
    ops = []
    if not is_pipeline_first_stage(ignore_virtual):
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            recv_prev_tensor,
            get_pipeline_model_parallel_prev_rank(),
            group=get_pipeline_model_parallel_group(),
        )
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend,
            send_prev_tensor,
            get_pipeline_model_parallel_prev_rank(),
            group=get_pipeline_model_parallel_group(),
        )
        ops.append(recv_prev_op)
        ops.append(send_prev_op)
    if not is_pipeline_last_stage(ignore_virtual):
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend,
            send_next_tensor,
            get_pipeline_model_parallel_next_rank(),
            group=get_pipeline_model_parallel_group(),
        )
        recv_next_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            recv_next_tensor,
            get_pipeline_model_parallel_next_rank(),
            group=get_pipeline_model_parallel_group(),
        )
        ops.append(send_next_op)
        ops.append(recv_next_op)

    reqs = torch.distributed.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _EMBEDDING_GROUP
    _EMBEDDING_GROUP = None
    global _POSITION_EMBEDDING_GROUP
    _POSITION_EMBEDDING_GROUP = None

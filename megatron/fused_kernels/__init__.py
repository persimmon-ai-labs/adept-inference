import os
import pathlib
import signal
import multiprocessing
import time
import importlib
import sys
from torch.utils import cpp_extension
from megatron.fused_kernels.megatron_fused_kernels.load_kernel import (
    KernelBuildInfo,
    ALL_BUILD_KERNELS,
    JIT_BUILD_KERNELS,
)


class CompilationTimeoutError(Exception):
    pass


def _load_kernel(kernel_build_info):
    kernel_build_info.load()


def load(force_build_fused_kernels=False):
    """Load fused kernels."""
    if force_build_fused_kernels:
        for kernel_build_info in ALL_BUILD_KERNELS.values():
            _load_kernel(kernel_build_info)
    else:
        # Just comile the kernels that we need to JIT compile.
        for kernel_name in JIT_BUILD_KERNELS:
            kernel_build_info = ALL_BUILD_KERNELS[kernel_name]
            _load_kernel(kernel_build_info)

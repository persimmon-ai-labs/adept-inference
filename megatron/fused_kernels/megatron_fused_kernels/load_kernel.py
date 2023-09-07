import os
import pathlib
from dataclasses import dataclass
from typing import List
from torch.utils import cpp_extension
from pathlib import Path


def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")


@dataclass
class KernelBuildInfo:
    name: str
    sources: List[Path]
    build_directory: str
    extra_cflags: List[str]
    extra_cuda_cflags: List[str]
    verbose: bool

    def __init__(self, name, sources, build_directory, extra_cflags, extra_cuda_cflags, verbose):
        self.name = name
        self.extra_cflags = extra_cflags
        self.extra_cuda_cflags = extra_cuda_cflags
        self.verbose = verbose
        if not isinstance(build_directory, Path):
            build_directory = Path(build_directory)
        self.build_directory = build_directory
        for i, source in enumerate(sources):
            if not isinstance(source, Path):
                sources[i] = Path(source)
        self.sources = sources

    def __repr__(self):
        return f"KernelBuildInfo(name={self.name}, sources={self.sources}, build_directory={self.build_directory}, extra_cflags={self.extra_cflags}, extra_cuda_cflags={self.extra_cuda_cflags}, verbose={self.verbose})\n"

    def to_setup_cpp_extension(self) -> cpp_extension.CUDAExtension:
        sources = [str(source) for source in self.sources]
        return cpp_extension.CUDAExtension(
            name=self.name,
            sources=sources,
            extra_compile_args={
                "cxx": self.extra_cflags,
                "nvcc": self.extra_cuda_cflags,
            },
            is_python_module=True,
        )

    def load(self):
        os.environ["TORCH_CUDA_ARCH_LIST"] = ""
        _create_build_dir(self.build_directory)
        _ = cpp_extension.load(
            name=self.name,
            sources=self.sources,
            build_directory=Path(self.build_directory),
            extra_cflags=self.extra_cflags,
            extra_cuda_cflags=self.extra_cuda_cflags,
            verbose=self.verbose,
        )


srcpath = pathlib.Path(__file__).parent.absolute()

BUILD_DIR = srcpath / "build"

BASE_CFLAGS = [
    "-O3",
    "-llibtorch_python",
]
BASE_CUDA_CFLAGS = [
    "-O3",
    "--use_fast_math",
    "-gencode",
    "arch=compute_80,code=sm_80",
]

BASE_MASKED_SOFTMAX_FUSION_CUDA_CFLAGS = BASE_CUDA_CFLAGS + [
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]

# These are the kernels that we need to build JIT as they have
# some issues in installing with pip. Note: They are defined in the
# ALL_BUILD_KERNELS dictionary below.
JIT_BUILD_KERNELS = []

ALL_BUILD_KERNELS = {
    "scaled_upper_triang_masked_softmax": KernelBuildInfo(
        name="scaled_upper_triang_masked_softmax_cuda",
        sources=[
            srcpath / "scaled_upper_triang_masked_softmax.cpp",
            srcpath / "scaled_upper_triang_masked_softmax_cuda.cu",
        ],
        build_directory=BUILD_DIR,
        extra_cflags=BASE_CFLAGS,
        extra_cuda_cflags=BASE_MASKED_SOFTMAX_FUSION_CUDA_CFLAGS,
        verbose=True,
    ),
    "scaled_masked_softmax_cuda": KernelBuildInfo(
        name="scaled_masked_softmax_cuda",
        sources=[srcpath / "scaled_masked_softmax.cpp", srcpath / "scaled_masked_softmax_cuda.cu"],
        build_directory=BUILD_DIR,
        extra_cflags=BASE_CFLAGS,
        extra_cuda_cflags=BASE_MASKED_SOFTMAX_FUSION_CUDA_CFLAGS,
        verbose=True,
    ),
    "scaled_softmax_cuda": KernelBuildInfo(
        name="scaled_softmax_cuda",
        sources=[srcpath / "scaled_softmax.cpp", srcpath / "scaled_softmax_cuda.cu"],
        build_directory=BUILD_DIR,
        extra_cflags=BASE_CFLAGS,
        extra_cuda_cflags=BASE_MASKED_SOFTMAX_FUSION_CUDA_CFLAGS,
        verbose=True,
    ),
    "fused_mix_prec_layer_norm_cuda": KernelBuildInfo(
        name="fused_mix_prec_layer_norm_cuda",
        sources=[srcpath / "layer_norm_cuda.cpp", srcpath / "layer_norm_cuda_kernel.cu"],
        build_directory=BUILD_DIR,
        extra_cflags=BASE_CFLAGS,
        extra_cuda_cflags=BASE_CUDA_CFLAGS + ["-maxrregcount=50"],
        verbose=True,
    ),
    "fused_dense_cuda": KernelBuildInfo(
        name="fused_dense_cuda",
        sources=[srcpath / "fused_weight_gradient_dense.cpp", srcpath / "fused_weight_gradient_dense_cuda.cu"],
        build_directory=BUILD_DIR,
        extra_cflags=BASE_CFLAGS,
        extra_cuda_cflags=BASE_CUDA_CFLAGS,
        verbose=True,
    ),
}

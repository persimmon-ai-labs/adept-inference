from setuptools import setup, find_packages
from torch.utils import cpp_extension
from megatron_fused_kernels.load_kernel import ALL_BUILD_KERNELS, JIT_BUILD_KERNELS
from megatron_fused_kernels.load_cpp_extensions import get_helpers_extension


def get_kernel_extensions():
    # remove 'fused_dense_cuda' from the list of kernels to build
    # as it is not used in the current version of Megatron-LM
    # and it causes compilation errors when doing ninja installs.
    # We'll need to JIT compile it instead.
    kernels_to_build = [
        build_info.to_setup_cpp_extension() for k, build_info in ALL_BUILD_KERNELS.items() if k not in JIT_BUILD_KERNELS
    ]
    return kernels_to_build


setup(
    name="megatron_fused_kernels",
    packages=find_packages(exclude=("tests",)),
    packages_dir={"megatron_fused_kernels": "megatron_fused_kernels"},
    ext_modules=get_kernel_extensions() + get_helpers_extension(),
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    zip_safe=False,
    author="Adept AI",
    author_email="",
)

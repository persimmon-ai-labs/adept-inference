# Copyright (c) 2023, ADEPT AI LABS INC.  All rights reserved.
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

FROM nvcr.io/nvidia/pytorch:23.04-py3

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata build-essential devscripts debhelper fakeroot


WORKDIR /opt/hpcx
RUN rm -rf ucx && \
    git clone --recursive https://github.com/openucx/ucx.git && \
    pushd ucx && \
    git fetch --all --tags && \
    git checkout tags/v1.14.1 && \
    ./autogen.sh && \
    mkdir UCX_BUILD && \
    ./contrib/configure-release-mt --prefix=/opt/hpcx/ucx/UCX_BUILD/ --with-cuda=/usr/local/cuda/ && \
    make -j && \
    make install && \
    popd

RUN rm -rf ucc && \
    git clone --recursive https://github.com/openucx/ucc.git && \
    pushd ucc && \
    git fetch --all --tags && \
    git checkout tags/v1.2.0 && \
    ./autogen.sh && \
    mkdir UCC_BUILD && \
    ./configure --prefix=/opt/hpcx/ucc/UCC_BUILD --with-ucx=/opt/hpcx/ucx/UCX_BUILD/ --with-nccl=/usr --with-cuda=/usr/local/cuda/ --with-nvcc-gencode="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_90,code=compute_90"&& \
    make -j && \
    make install && \
    popd

ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/UCX_BUILD/lib:/opt/hpcx/ucc/UCC_BUILD/lib:$LD_LIBRARY_PATH
ENV UCX_HOME=/opt/hpcx/ucx/UCX_BUILD/
ENV UCC_HOME=/opt/hpcx/ucc/UCC_BUILD/
ENV WITH_CUDA=/usr/local/cuda

WORKDIR /workspace

# Install FlashAttention
RUN pip install flash-attn==2.0.0.post1

# Install rotary embedding, cross entropy, and FT't attention kernel
# [2022-11-08] TD: Check out a specific version to make build reproducible
RUN git clone https://github.com/HazyResearch/flash-attention \
    && cd flash-attention && git checkout b8020d73c9e068665586989883083a4a5429a443 \
    && cd csrc/rotary && pip install . && cd ../../ \
    && cd csrc/xentropy && pip install . && cd ../../ \
    && cd csrc/ft_attention && pip install . && cd ../../ \
    && cd .. && rm -rf flash-attention


COPY megatron/fused_kernels/ megatron/fused_kernels/
ENV PATH="${PATH}:/opt/hpcx/ompi/bin" LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"
RUN ldconfig /opt/hpcx/ompi/bin
RUN TORCH_CUDA_ARCH_LIST="" cd megatron/fused_kernels/ && python setup.py install sdist && cd ../..
RUN cd /usr/local/lib/python3.8/dist-packages/megatron_fused_kernels-0.0.0-py3.8-linux-x86_64.egg/; mv *.so megatron_fused_kernels/;
RUN rm -rf megatron/fused_kernels

# Install apt-get dependencies for pip requirements.
ENV DEBIAN_FRONTEND=noninteractive
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get update -y && apt-get install -y nodejs iputils-ping \
    wget ca-certificates tzdata zip locales \
    && locale-gen en_US en_US.UTF-8 en_US.utf8 && dpkg-reconfigure --frontend noninteractive locales \
    && npm install pm2 -g \
    && pip install --upgrade pip setuptools \
    && rm -rf /var/lib/apt/lists/*

# Change locale for click
ENV LANG=C.UTF-8 LANGUAGE=en_US.en LC_ALL=C.UTF-8

# Install requirements & cleanup.
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools && \
    pip install -r requirements.txt && rm requirements.txt

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel AS base

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# use tsinghua source
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list \
    && sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y vim tmux zip unzip wget git build-essential libibverbs-dev ca-certificates \
    curl iproute2 ffmpeg libsm6 libxext6 kmod ccache libnuma-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

RUN pip install --no-cache-dir packaging ninja cmake scikit-build-core uv ruff pre-commit -U

RUN git clone https://github.com/vllm-project/vllm.git && cd vllm \
    && python use_existing_torch.py && pip install -r requirements/build.txt \
    && pip install --no-cache-dir --no-build-isolation -v -e .

RUN git clone https://github.com/sgl-project/sglang.git && cd sglang/sgl-kernel \
    && make build && make clean

RUN pip install --no-cache-dir diffusers transformers tokenizers accelerate safetensors opencv-python numpy imageio \
    imageio-ffmpeg einops loguru qtorch ftfy easydict

RUN git clone https://github.com/Dao-AILab/flash-attention.git --recursive

RUN cd flash-attention && python setup.py install && rm -rf build

RUN cd flash-attention/hopper && python setup.py install && rm -rf build

# RUN git clone https://github.com/thu-ml/SageAttention.git

# # install sageattention with hopper gpu sm9.0
# RUN cd SageAttention && sed -i 's/set()/{"9.0"}/' setup.py && EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 pip install --no-cache-dir -v -e .

WORKDIR /workspace

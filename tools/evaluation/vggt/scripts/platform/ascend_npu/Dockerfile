# From tools/evaluation/vggt/:
# docker build -f scripts/platform/ascend_npu/Dockerfile -t lightx2v-eval-vggt:ascend-910b .
# This image contains dependencies only; frozen sources and inputs are mounted at run time.

FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.10
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /eval
ENV PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8


# Install the complete Torch/NPU stack from public package sources.
RUN python3 -m pip install --no-cache-dir --no-deps torch==2.12.0+cpu torchvision==0.27.0+cpu \
      --index-url https://download.pytorch.org/whl/cpu && \
    python3 -m pip install --no-cache-dir \
      torch-npu==2.12.0 \
      filelock==3.32.3 typing-extensions==4.16.0 sympy==1.14.0 \
      networkx==3.4.2 Jinja2==3.1.6 fsspec==2026.7.0 packaging==26.3 \
      numpy==2.2.6 Pillow==11.3.0 scipy==1.15.3 \
      transformers==4.57.1 tokenizers==0.22.2 huggingface-hub==0.36.2 \
      diffusers==0.35.2 accelerate==1.10.1 safetensors==0.6.2 \
      einops==0.8.1 loguru==0.7.3 imageio==2.37.0 imageio-ffmpeg==0.6.0 \
      gguf==0.17.1 ftfy==6.3.1 aiohttp==3.12.15 peft==0.17.0 \
      timm==1.0.20 kornia==0.8.1 kornia-rs==0.1.9 trimesh==4.7.4

COPY configs/ascend_npu/runtime-requirements.txt /opt/eval-runtime-requirements.txt
RUN python3 -m pip install --no-cache-dir --no-deps -r /opt/eval-runtime-requirements.txt && \
    python3 -m pip install --no-cache-dir matplotlib==3.10.6

# The evaluator overrides ENTRYPOINT for both NPU and CPU workers. Restore CANN
# setup here, then preserve /eval/runtime in PYTHONPATH for --install-deps users.
RUN printf '%s\n' '#!/bin/bash' 'set -eo pipefail' \
      'source /usr/local/Ascend/ascend-toolkit/set_env.sh' \
      'source /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh' \
      'source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1' \
      'exec python3 "$@"' > /usr/local/bin/eval-python && \
    chmod 755 /usr/local/bin/eval-python
ENTRYPOINT ["/usr/local/bin/eval-python"]
CMD ["--version"]

# lightx2v_kernel

### Preparation
```
# Install torch, at least version 2.7

pip install scikit_build_core uv
```

### Build whl

```
git clone https://github.com/NVIDIA/cutlass.git

git clone https://github.com/ModelTC/LightX2V.git

cd LightX2V/lightx2v_kernel

# Set the /path/to/cutlass below to the absolute path of cutlass you download.

MAX_JOBS=$(nproc) && CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
uv build --wheel \
    -Cbuild-dir=build . \
    -Ccmake.define.CUTLASS_PATH=/path/to/cutlass \
    --verbose \
    --color=always \
    --no-build-isolation
```

### NVIDIA Thor NVFP4 + CuTe DSL ViT FMHA

The Thor-only ViT FMHA bridge links the AArch64 SM110 CUDA 13 artifact shipped
by TensorRT-Edge-LLM. It implements dense, bidirectional self-attention and
converts LightX2V BF16 Q/K/V tensors to the FP16 AOT kernel interface.

```bash
MAX_JOBS=$(nproc) && CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
uv build --wheel \
    -Cbuild-dir=build-thor-cutedsl . \
    -Ccmake.define.CUTLASS_PATH=/path/to/cutlass \
    -Ccmake.define.LIGHTX2V_THOR_NVFP4_ONLY=ON \
    -Ccmake.define.LIGHTX2V_ENABLE_CUTEDSL_VIT_FMHA=ON \
    -Ccmake.define.TENSORRT_EDGE_LLM_ROOT=/path/to/TensorRT-Edge-LLM \
    --verbose \
    --color=always \
    --no-build-isolation
```

The TensorRT-Edge-LLM checkout must contain
`kernelSrcs/cuteDSLPrebuilt/cutedsl_aarch64_sm_110_cuda13.tar.gz` or an already
extracted `cpp/kernels/cuteDSLArtifact/aarch64/sm_110` directory.


### Install whl
```
pip install dist/*whl --force-reinstall --no-deps
```

### Test

##### cos and speed test, mm without bias
```
python test/nvfp4_nvfp4/test_bench2.py
```

##### cos and speed test, mm with bias
```
python test/nvfp4_nvfp4/test_bench3_bias.py
```

##### Bandwidth utilization test for quant
```
python test/nvfp4_nvfp4/test_quant_mem_utils.py
```

##### tflops test for mm
```
python test/nvfp4_nvfp4/test_mm_tflops.py
```

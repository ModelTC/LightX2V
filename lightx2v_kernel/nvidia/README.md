# LightX2V NVIDIA Kernels

The wheel distribution is `lightx2v-kernel-nvidia`, imported as
`lightx2v_kernel_nvidia`. LightX2V uses the new package when installed and falls
back to the legacy `lightx2v_kernel` package when it is absent, so existing wheels
can still be used without rebuilding. The legacy wheel must include the operators
required by the selected model.

The C++ operator namespace remains `torch.ops.lightx2v_kernel` for binary
compatibility.

### Preparation
```
# Install torch, at least version 2.7

pip install scikit_build_core uv
```

### Build whl

```
git clone https://github.com/NVIDIA/cutlass.git

git clone https://github.com/ModelTC/LightX2V.git

cd LightX2V/lightx2v_kernel/nvidia

# Set the /path/to/cutlass below to the absolute path of cutlass you download.

MAX_JOBS=$(nproc) && CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
uv build --wheel \
    -Cbuild-dir=build . \
    -Ccmake.define.CUTLASS_PATH=/path/to/cutlass \
    --verbose \
    --color=always \
    --no-build-isolation
```


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

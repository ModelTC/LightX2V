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

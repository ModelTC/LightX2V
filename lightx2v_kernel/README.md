# lightx2v_kernel

### Preparation
```
# Install torch, at least version 2.7

pip install scikit_build_core uv
```

### Build whl
```
MAX_JOBS=$(nproc) && CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
uv build --wheel \
    -Cbuild-dir=build . \
    --verbose \
    --color=always \
    --no-build-isolation
```

During the above build process, the cutlass source code will be downloaded automatically. If you have already downloaded the source code, you can specify the local cutlass path:
```
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

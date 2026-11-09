# LightX2V Apple Kernels

Currently supports **INT8 GEMM**. Requires **Apple M5 or newer**, macOS 26.4+, and PyTorch 2.14+.
Run the following commands from the **LightX2V repository root**.

## Install

```bash
pip install ./lightx2v_kernel/apple
```

## Prepare the Metal libraries

If your model already includes `apple_silicon_metal/`, use that folder and skip compilation.

To compile your own libraries, install the Metal Toolchain and compile the required shapes.

Specify each shape as `KxN`, corresponding to:

```text
X[M,K] @ W[K,N] -> Y[M,N]
```

K is the input feature dimension, N is the output feature dimension, and M is the number of input rows (e.g. tokens). K and N are fixed at compilation; M remains dynamic.

K and N must be positive multiples of 128. Weights use their original INT8
`[N,K]` checkpoint storage, passed to GEMM as a `[K,N]` transpose with strides
`(1,K)`. The kernel handles widths such as K=5376 and K=2688 directly, without
padding or dequantizing weights. Partial activation rows are padded along M.

For Qwen-Image-2.1:

```bash
python lightx2v_kernel/apple/python/lightx2v_kernel_apple/int8_gemm/compile_metallib.py \
    4096x4096 4096x12288 12288x4096 \
    --output-dir /path/to/metallib
```

Precompiled libraries for this example are also available in [Qwen-Image-2.1-Turbo-INT8 / apple_silicon_metal](https://huggingface.co/lightx2v/Qwen-Image-2.1-Turbo-INT8/tree/main/apple_silicon_metal). Download the `.metallib` files to any local directory to skip compilation.

For MiniMax-H3's six main-block projections (FC1, FC2, Q, K, V and
attention output):

```bash
python lightx2v_kernel/apple/python/lightx2v_kernel_apple/int8_gemm/compile_metallib.py \
    5376x28672 14336x5376 5376x7168 7168x5376 \
    --output-dir /path/to/h3-metallib
```

### MiniMax-H3 checkpoint

Convert an already merged Turbo BF16 transformer into standalone INT8 shards:

```bash
python tools/convert/convert_h3_int8_metal.py \
    --source /path/to/MiniMax-H3-Turbo-Merged/transformer \
    --output /path/to/MiniMax-H3-Turbo-INT8/transformer \
    --device mps
```

The output directory must be new. The converter uses FP32 per-output-channel
scales and quantizes 300 Linear weights across the 50 main blocks. AdaLN,
pre/post projections, text refiner, text encoder and VAEs keep their original
precision. Do not reapply the merged Turbo LoRA at runtime.

In `configs/platforms/mps/minimax_h3_t2av_4step_int8.json`, set
`dit_quantized_ckpt` to the converted `transformer` directory. Update
`lightx2v_path`, `model_path` and `LIGHTX2V_METALLIB_DIR` in
`scripts/platforms/mps/run_minimax_h3_t2av_int8.sh`, then run:

```bash
bash scripts/platforms/mps/run_minimax_h3_t2av_int8.sh
```

This uses the standard `python -m lightx2v.infer` entry point for resident
T2AV: four steps, 544x960, 124 frames at 24 fps. `model_path` supplies the
original tokenizer, text encoder and VAEs. Q/K/V remain separate.

The opt-in `mps_resident_weights=true` loader reads DiT and BF16 text weights
directly into MPS storage without retaining host weight copies. It requires
single-device resident execution without offload, shared CPU arenas, runtime
LoRA or lazy loading. The dedicated MPS text encoder loads the native text
prefix for T2AV. Mixed INT8 disk streaming, tensor
parallelism and other task types have not been validated.

Compilation generates `.metal`, `.air`, and `.metallib` files. Only the `.metallib` files need to be kept for inference.

The output directory can be anywhere. Set `LIGHTX2V_METALLIB_DIR` to the same directory when running inference.

## Run inference

Set the directory containing the `.metallib` files:

```bash
export LIGHTX2V_METALLIB_DIR="/path/to/metallib"
```

For a model with bundled libraries, point this variable to `<model_path>/apple_silicon_metal/` instead.

Use `dit_quantized=true` and `dit_quant_scheme="int8-metal"` in the model configuration. Update `lightx2v_path`, `model_path`, and `LIGHTX2V_METALLIB_DIR` in the launch script before running:

```bash
bash scripts/platforms/mps/qwen_image_21_turbo_t2i_int8.sh
bash scripts/platforms/mps/qwen_image_21_turbo_i2i_int8.sh
```

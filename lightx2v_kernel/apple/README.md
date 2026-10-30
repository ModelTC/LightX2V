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

For Qwen-Image-2.1:

```bash
python lightx2v_kernel/apple/python/lightx2v_kernel_apple/int8_gemm/compile_metallib.py \
    4096x4096 4096x12288 12288x4096 \
    --output-dir /path/to/metallib
```

Precompiled libraries for this example are also available in [Qwen-Image-2.1-Turbo-INT8 / apple_silicon_metal](https://huggingface.co/lightx2v/Qwen-Image-2.1-Turbo-INT8/tree/main/apple_silicon_metal). Download the `.metallib` files to any local directory to skip compilation.

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

For model-bound Qwen INT8 checkpoints, keep `apple_silicon_metal/manifest.json` beside the libraries. The Qwen runner validates the checkpoint's INT8 K/N coverage, model revision, library sizes, and SHA-256 values before loading the transformer. Set `LIGHTX2V_ALLOW_UNVERIFIED_METALLIB=1` only for an explicitly unverified custom library directory.

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

For the standard MiniMax-H3 transformer Linear shapes:

```bash
python lightx2v_kernel/apple/python/lightx2v_kernel_apple/int8_gemm/compile_metallib.py \
    5376x28672 14336x5376 5376x7168 7168x5376 2688x96768 \
    --output-dir /path/to/h3-metallib
```

These cover FC1, FC2, individual Q/K/V, attention output, and uncached AdaLN.
Pre/post projections and the text refiner keep their existing floating-point
backends. This is single-Linear support; a full H3 INT8 inference setup also
needs a compatible per-channel INT8 checkpoint and model-level validation.

### MiniMax-H3 MLP checkpoint

Convert an already merged Turbo BF16 transformer into an independent sharded
checkpoint. This profile quantizes only the main transformer's FC1 and FC2;
attention, AdaLN, pre/post projections and the text refiner retain their
original tensor bytes and dtypes.

```bash
PLATFORM=mps python -m tools.convert.convert_h3_mlp_int8_metal \
    --source /path/to/MiniMax-H3-Turbo-Merged/transformer \
    --output /path/to/MiniMax-H3-Turbo-MLP-INT8/transformer \
    --device mps
```

The output directory must not already exist. Conversion writes FP32
per-output-channel scales, an indexed mixed-precision checkpoint and a
provenance manifest. Do not apply the Turbo LoRA again at runtime.

Select this checkpoint with:

```json
{
  "dit_quantized": true,
  "dit_quant_scheme": "int8-metal",
  "dit_quant_scope": "mlp",
  "dit_quantized_ckpt": "/path/to/MiniMax-H3-Turbo-MLP-INT8/transformer",
  "use_fused_qkv": false
}
```

The model checks the checkpoint's declared scope before loading. This profile
needs the `5376x28672` and `14336x5376` libraries. For resident `t2av`, use
the opt-in resident loader described below. Mixed INT8 disk streaming, tensor
parallelism and fused QKV have not been validated.

### MiniMax-H3 full projection checkpoint and resident inference

For all six main-transformer projections per block (FC1, FC2, Q, K, V and
attention output), create a standalone checkpoint directly from merged Turbo
BF16 weights. This copies untouched tensors into the output shards; it does
not depend on the original weight directory after conversion.

```bash
PLATFORM=mps python -m tools.convert.convert_h3_int8_metal \
    --source /path/to/MiniMax-H3-Turbo-Merged/transformer \
    --output /path/to/MiniMax-H3-Turbo-INT8/transformer \
    --scope dit_proj --device mps
```

The output directory must be new. Do not reapply the Turbo LoRA. AdaLN,
pre/post projections, text refiner, text encoder and VAEs retain their existing
precision. The checkpoint records its scope, quantized projection count,
source hashes and output shard hashes. The H3 loader validates INT8 weight
shapes and FP32 per-channel scale shapes against this contract before loading.

Compile the four required Linear shapes, then run the repository entry point:

```bash
python lightx2v_kernel/apple/python/lightx2v_kernel_apple/int8_gemm/compile_metallib.py \
    5376x28672 14336x5376 5376x7168 7168x5376 --output-dir /path/to/h3-metallib

bash scripts/platforms/mps/run_minimax_h3_t2av_int8.sh \
    --model-path /path/to/MiniMax-H3 \
    --checkpoint /path/to/MiniMax-H3-Turbo-INT8/transformer \
    --metallib-dir /path/to/h3-metallib \
    --prompt "A cinematic fox walking through a snowy forest" \
    --output save_results/h3_int8.mp4
```

`--model-path` supplies the tokenizer, text encoder and VAEs. The default
`configs/platforms/mps/minimax_h3_t2av_4step_int8.json` uses 4 steps, 544x960,
124 frames at 24 fps, with `dit_quant_scope="dit_proj"`. Override `--config`
to use another compatible resident configuration.

The opt-in `mps_resident_weights=true` loader reads DiT and BF16 text weights
directly into CPU-visible MPS allocations and adopts their storage without
retaining a CPU/pinned copy. It requires resident single-device execution;
offload, shared CPU arenas, runtime LoRA and lazy loading are incompatible.
`text_encoder_load_vision_on_init=false` defers the unused vision tower and
processor for T2AV; image requests can still load them on demand. Without that
flag, the original eager-loading behavior is retained. This entry point uses
the standard H3 runner and model, without task-specific loader replacements.

The H3 scope selector also accepts experimental projection profiles:

| `dit_quant_scope` | Main-transformer INT8 projections |
| --- | --- |
| `fc1` | FC1 only |
| `fc2` | FC2 only |
| `mlp` | FC1 and FC2 |
| `mlp_out` | FC1, FC2 and attention output |
| `mlp_qkv` | FC1, FC2 and separate Q/K/V |
| `dit_proj` | FC1, FC2, separate Q/K/V and attention output |

These profiles leave AdaLN, pre/post projections, text refiner, text encoder
and VAEs at their existing precision. Each profile requires an indexed
checkpoint declaring the same scope. The MLP converter above produces only
`mlp`; changing the configuration alone does not convert other projections.
For ablations, the loader honors the index's exact tensor-to-file mapping,
including nested paths, so a checkpoint view can select tensors from shared
weight banks. Such views depend on their referenced directories and are not
self-contained model exports. Profile quality and performance must be checked
on the intended workload before use.

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

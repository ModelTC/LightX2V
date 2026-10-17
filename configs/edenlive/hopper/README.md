# EdenLive single-GPU H100/H200 profile

`wan22_5b_ar_fp8_sgl_lighttae.json` uses DiT FP8-SGL, SM90 fused dense
SageAttention, LightTAE, 1280 x 704, 24 FPS, four denoising steps and the
existing 18-frame latent attention window / six-frame sink / three-frame chunk.
KV/decode overlap stays disabled. This profile has not been benchmarked on H100/H200.

Enabled DiT optimizations: shared activation quantization, packed QKV,
GELU + FP8 activation quantization fusion, fixed pre-infer projection caches,
Proxy interpolation caches and skipping the unused cache-commit suffix.

`cache_cross_attn_quantized_kv` and the whole DiT `use_cuda_graph` are disabled:
the current prepared cross-KV implementation only supports
`sage_attn2_sm120_cached`, and the DiT graph requires that prepared cache.
SM90 still reuses projected text K/V and the transposed self-attention V buffer.
Proxy CUDA Graph remains enabled independently.

ProxyDINO loads the original `vision-encoder/projection.safetensors`, with BF16
linear weights, BF16 residuals, Sage2 attention and fused RoPE/normalization.
No additional Proxy checkpoint conversion is needed. These Proxy precision and
attention choices can change outputs relative to the original FP32 residual /
Torch SDPA encoder. The SM120-only ConvRot shared-QKV path is not enabled.

## Convert DiT

Run from the LightX2V repository root. Convert the native BF16 checkpoint using
`--quantize fp8`; the runtime name for this checkpoint is `fp8-sgl`.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/convert/convert_edenlive.py \
  --source ../EdenLive-x2v/dit-5b/diffusion_pytorch_model.safetensors \
  --output ../EdenLive-x2v/quantized \
  --output_name dit-5b-fp8 \
  --quantize fp8 \
  --device cuda:0
```

The converter keeps an existing native destination unless `--overwrite` is
passed. The existing `dit-5b-fp8.safetensors` from the 5090 machine can also be
copied to the H-card machine; no H-specific weight conversion is needed.
Keep the complete native `EdenLive-x2v` model directory for encoders/VAE and
copy `lighttaew2_2.safetensors` separately. Use this modified LightX2V checkout,
which contains the fused attention and projection implementations.

## Run

With the same sibling directories as this workspace:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/edenlive/run_hopper.sh
```

For different destination paths:

```bash
MODEL_PATH=/path/to/EdenLive-x2v \
TAE_PATH=/path/to/lighttaew2_2.safetensors \
SAMPLE_DIR=/path/to/EdenLive-Official-wp-demo/examples/samples/cryo-crossing \
CUDA_VISIBLE_DEVICES=0 bash scripts/edenlive/run_hopper.sh
```

`DIT_QUANTIZED_CKPT` and `SAVE_RESULT_PATH` optionally override the checkpoint
and output path. The launcher writes path overrides to a temporary config and
removes it on exit. Extra arguments are forwarded to `lightx2v.infer`.
The H-card environment needs SM90 builds of PyTorch, SGL Kernel, SageAttention
(including `_fused` and `_qattn_sm90`), FlashInfer and Triton.

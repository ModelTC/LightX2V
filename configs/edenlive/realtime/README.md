# EdenLive realtime profiles

Both profiles use 1280×704 output, 24 FPS, four denoising steps, a three-latent-frame chunk, local attention 18, sink 6, LightTAE streaming decode, and resident FP8-SGL T5.

| Profile | Hardware | DiT | ProxyDINO | Launcher |
| --- | --- | --- | --- | --- |
| `wan22_5b_ar_int8_convrot_5090.json` | RTX 5090 / SM120 | INT8 ConvRot | INT8 ConvRot | `scripts/edenlive/run_5090.sh` |
| `wan22_5b_ar_fp8_sgl_hopper.json` | H100 / H200 / SM90 | FP8-SGL | Original BF16 weights | `scripts/edenlive/run_hopper.sh` |

Each profile explicitly sets `self_attn_1_type`, `cross_attn_1_type`, and `cross_attn_2_type`:

| Profile | Self attention | Cross attention 1 | Cross attention 2 |
| --- | --- | --- | --- |
| RTX 5090 | `sage_attn2_sm120_cached` | `sage_attn2_sm120_cached` | `sage_attn2` |
| H100 / H200 | `sage_attn2_sm90_fused` | `sage_attn2_sm90_fused` | `sage_attn2_sm90_fused` |

Use the profile matching the GPU architecture. ProxyDINO uses batched Sage2 for these cached/fused DiT profiles; original top-level profiles retain its SDPA default. Shared and packed projections, FFN fusion, fixed-input caches, and quantized cross-attention KV reuse are enabled when their operators and execution mode support them. The shared linear operators support `fp8-sgl` and `int8-convrot`; unsupported paths retain the regular operators. `use_cuda_graph` controls graph use for both DiT and ProxyDINO, with capability checks for each.

| Attention backend | Persistent transposed self-attention V | Prepared quantized cross-attention KV | DiT stack CUDA Graph |
| --- | --- | --- | --- |
| SM120 cached Sage2 | Supported | Supported | Eligible in the resident single-GPU profile |
| SM90 fused Sage2 | Supported | Unavailable | Uses compiled blocks; ProxyDINO can still use CUDA Graph |

See [`common/ops/mm/README.md`](../../../lightx2v/common/ops/mm/README.md) for projection/fusion shape limits, precision requirements, and fallback behavior. Existing top-level profiles retain their original DiT attention settings.

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/edenlive/run_5090.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/edenlive/run_hopper.sh
```

The realtime launchers default to `PROFILING_DEBUG_LEVEL=0`. Set it to `1` or `2`
for stage diagnostics. These diagnostic timers insert GPU synchronizations and
logs, so use `0` when measuring streaming latency or comparing performance.

Set `MODEL_PATH`, `SAMPLE_DIR`, `TAE_PATH`, `DIT_QUANTIZED_CKPT`, `T5_QUANTIZED_CKPT`, or `SAVE_RESULT_PATH` for another host. The 5090 launcher also accepts `PROXY_QUANTIZED_CKPT`. `T5_CPU_OFFLOAD=true` enables a T5 offload comparison. Launchers create a temporary configuration and keep the saved profiles unchanged.

Convert the Hopper DiT checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/convert/convert_edenlive.py \
  --source ../EdenLive-x2v/dit-5b/diffusion_pytorch_model.safetensors \
  --output ../EdenLive-x2v/quantized --output_name dit-5b-fp8 \
  --quantize fp8 --device cuda:0
```

The existing `dit-5b-fp8.safetensors` and `t5-fp8.safetensors` checkpoints can be copied to another host. Checkpoint conversion uses `fp8`; inference uses `fp8-sgl`.

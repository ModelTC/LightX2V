# Qwen-Image-2.1 optional configuration reference

This file collects optional capabilities and alternative values that are not all enabled in the recommended presets. Each JSON block is an independent configuration fragment: copy the fields you need into a complete inference JSON. Merge fields inside `parallel` with care; the fragments are not one combined configuration.

Runnable presets remain [1K single GPU](qwen_image_21_5090_1k.json), [2K single GPU](qwen_image_21_5090_2k.json), and [2K SP2](qwen_image_21_5090_2k_sp2.json). They already enable DiT FP8-F16 accumulation, dense Sage2, QwenVL language FP8, and VAE compile. The SP2 preset also enables language TP, VAE encoder/decoder parallelism, FP8 communication, and grouped head parallelism.

## DiT tensor parallelism

TP2 without SP:

```json
{
    "parallel": {
        "tensor_p_size": 2,
        "seq_p_size": 1
    }
}
```

`tensor_p_size` defaults to 1. DiT TP supports unquantized, `fp8-sgl`, and `fp8-f16-accum` safetensors weights. Heads, hidden size, and MLP intermediate size must be divisible by the TP size.

For **TP2 + SP2**, set `seq_p_size` to 2 and `seq_p_attn_type` to `"ulysses"`. This requires **4 processes/GPUs**: the distributed world size is `tensor_p_size * seq_p_size`. TP-local heads and target-image token count must be divisible by the SP size. Adjust the launcher as well; the supplied SP2 scripts launch exactly two processes.

DiT TP uses the `tensor_p` mesh group. It is independent of QwenVL language TP and cannot be combined with `shared_cpu_weights=true`.

## DiT torch.compile

```json
{
    "use_compile": true,
    "compile_backend": "default"
}
```

`use_compile` defaults to false. The default backend compiles each DiT block with PyTorch; `"default"` is the configuration value, not `"inductor"`. Other registered platform backends need separate validation.

This is independent of `vae_use_compile`. DiT compile has not improved the measured RTX 5090 workload, so it remains optional. Keep normal warmup enabled and allow compilation for newly encountered shapes before measuring steady-state latency.

## DiT CPU offload

```json
{
    "cpu_offload": true,
    "offload_granularity": "model"
}
```

| Field | Values | Meaning |
| --- | --- | --- |
| `cpu_offload` | `false` (default), `true` | Keep DiT resident, or enable CPU offload |
| `offload_granularity` | `"model"` (default), `"block"` | Load/unload the entire DiT around its stage, or prefetch blocks through two GPU buffers |

Model offload reduces overlap with other pipeline components; it still needs the complete DiT weights on GPU during denoising. Block offload also reduces active DiT weight memory, at the cost of repeated transfers. Both keep condition KV on GPU and do not solve an oversized KV cache.

### Shared CPU block weights

```json
{
    "cpu_offload": true,
    "offload_granularity": "block",
    "shared_cpu_weights": true,
    "shared_cpu_weight_scope": "auto"
}
```

This shares the source block weights within one distributed job and IPC namespace. Pre/post weights, QwenVL, VAE, GPU buffers, and KV caches remain private. It reduces duplicated CPU weight storage, not GPU memory or transfer volume.

| Field | Values / default |
| --- | --- |
| `shared_cpu_weights` | `false` (default), `true` |
| `shared_cpu_weight_scope` | `"auto"` (default), `"numa"`, `"host"` |
| `shared_cpu_weight_strict_numa` | `true` (default), `false`; controls NUMA binding failure handling |
| `shared_cpu_weight_register_chunk_mb` | Positive integer; default `128` |
| `shared_cpu_weight_backend` | Only `"sysv"` |

`auto` selects NUMA-local groups when topology is known, otherwise host groups. `numa` requires known NUMA placement; `host` uses one copy per host/cohort and may incur remote NUMA accesses.

Requires CUDA, BF16 inference, and **block** offload. Source checkpoints may be unquantized BF16, FP8-SGL, or FP8-F16-accum. DiT TP, lazy loading, automatic weight quantization, and LoRA are unsupported combinations. Independent service processes do not automatically join the same shared arena.

## VAE alternatives

### Decoder mixed FP8

```json
{
    "vae_decoder_conv_mode": "cutlass_fp8_f16_accum",
    "vae_quantized_ckpt": "/path/to/Qwen-Image-2.1-vae-fp8/qwen_image_21_vae_decoder_fp8.safetensors",
    "vae_use_compile": true
}
```

`vae_decoder_conv_mode` accepts `"torch"` (default) or `"cutlass_fp8_f16_accum"`. FP8 mode requires its matching mixed qmax21 checkpoint and the SM120 kernel. The profile quantizes selected decoder convolutions; the encoder, attention, and high-resolution sensitive convolutions retain ordinary precision.

This is a memory-oriented option; the measured benefit used VAE compile, and decoding can be slower. FP8 together with VAE parallelism has not received the same complete quality/performance validation.

The conversion command is in the [VAE converter docstring](../../tools/convert/qwen_image_21_vae_decoder.py). To restore ordinary precision, set the mode to `"torch"` and remove `vae_quantized_ckpt` together.

### Decoder cut point

```json
{
    "vae_decode_parallel": true,
    "vae_decode_parallel_mode": "post_mid"
}
```

| Value | Cut point |
| --- | --- |
| `"full"` (default) | Split before the decoder; mid-block attention operates on each overlapping spatial shard |
| `"post_mid"` | Execute global mid-block attention first, then split the upsampling path |

Both modes use a finite halo, so neither guarantees bitwise equivalence to serial decode. VAE parallelism uses the distributed WORLD group, is not limited to SP2, and requires an evenly partitionable latent grid.

`vae_encode_parallel` independently shards encoder convolutions and reconstructs the full feature map before global mid-block attention. It defaults to false and is already enabled in the SP2 preset. `vae_use_compile` also defaults to false in code; the recommended presets enable it for both encoder and decoder.

## Alternative DiT linear and attention backends

Ordinary FP8 linear:

```json
{
    "dit_quantized": true,
    "dit_quant_scheme": "fp8-sgl",
    "dit_quantized_ckpt": "/path/to/Qwen-Image-2.1-fp8/qwen_image_21_fp8.safetensors"
}
```

| Linear mode | Required settings |
| --- | --- |
| Unquantized | `dit_quantized=false`; remove quant scheme/checkpoint fields |
| FP8-SGL | `dit_quantized=true`, `dit_quant_scheme="fp8-sgl"`, matching converted checkpoint |
| FP8-F16 accumulation | `dit_quantized=true`, `dit_quant_scheme="fp8-f16-accum"`, matching W14 profile checkpoint; `dit_fp8_activation_qmax=7.0` is recommended |

The FP16-accumulation path requires SM120 kernel support. Its activation qmax accepts finite values in `(0, 448]`, but changing it needs numerical validation; the checkpoint weight qmax remains fixed at 14. A regular FP8 checkpoint does not satisfy this profile by changing the runtime scheme alone.

The [conversion documentation](../../tools/convert/readme.md) describes the DiT conversion modes. For an ordinary-precision Torch baseline, see [qwen_image_21_torch.json](qwen_image_21_torch.json).

| `attn_type` | Use |
| --- | --- |
| `"sage_attn2"` | Recommended dense backend |
| `"sage_attn3"` | Optional lower-precision alternative; inspect output quality before adoption |
| `"flash_attn2"` | Alternative requiring a compatible FlashAttention installation |
| `"torch_sdpa"` | PyTorch backend |
| `"flash_attn3"` | Original general preset; requires supported hardware, not a validated RTX 5090 option |

This field selects DiT image/target attention. The triangular text-prefix mask, QwenVL attention, and VAE attention retain their SDPA paths.

## Ulysses tuning values

These fields belong **inside `parallel`**. The SP2 preset shows the recommended combination; this table records alternatives without adding one preset per combination.

| Field | Values / default |
| --- | --- |
| `seq_p_size` | Positive integer; `1` disables SP |
| `seq_p_attn_type` | Only `"ulysses"` for this model |
| `seq_p_prepost_backend` | `"torch"` (default), `"triton"` |
| `seq_p_a2a_backend` | `"torch"` (default); public `"round_robin"` path needs separate Qwen validation |
| `seq_p_tensor_fusion` | `false` (default), `true` |
| `seq_p_head_parallel` | `false` (default), `true` |
| `seq_p_head_parallel_group_size` | Integer from 1 to the local head count; default `1`; the preset uses `4` |
| `seq_p_quant_scheme` | `null`/absent: ordinary precision; `"fp8"`: FP8 communication |

Keep `seq_p_head_parallel` and `seq_p_head_parallel_group_size` together. Group size need not divide the local head count; the final group handles the remainder. When disabling head parallel, remove the group size or reset it to 1.

Triton pre/post requires tensor fusion and supports ordinary/FP8 communication. The public Torch path also accepts `"fp4"`, but Qwen quality/performance have not been validated for it; Triton pre/post rejects it. `round_robin` cannot be combined with head parallel.

## QwenVL switches already covered by the presets

These switches are included here to distinguish language TP from DiT TP:

| Field | Values / scope |
| --- | --- |
| `text_encoder_tensor_parallel` | `false` (default), `true`; language embedding, attention and MLP over WORLD |
| `text_encoder_cpu_offload` | `false` (default), `true`; stage offload of the whole QwenVL conditioner |
| `text_encoder_quantized` | `false` (default), `true`; language linear quantization |
| `text_encoder_quant_scheme` | `"Default"` when unquantized; only `"fp8-sgl"` when quantized |
| `text_encoder_quantized_ckpt` | Required matching checkpoint when quantized |

Language TP is enabled in the SP2 preset independently of DiT TP. Its size is WORLD size, with no separate text TP-size field; attention heads, KV heads, intermediate size, and vocabulary size must be divisible by it. For TP2×SP2, enabling this flag gives language TP4.

The vision tower remains unquantized and replicated. Language quantization, TP, and stage offload can be selected independently. For unquantized operation, also remove the quant scheme/checkpoint fields. The conversion example is in `convert_qwen_image_21_text_encoder_fp8` in [converter.py](../../tools/convert/converter.py).

## Support boundaries

SLA sparse attention and Ring are not supported Qwen presets. VAE encoder FP8, VAE Sage2, and QwenVL vision FP8/TP were not retained as supported options.

The runner rejects VAE CPU offload/tiling, lazy loading, module unloading, CFG parallel, PipeFusion, disaggregated mode, LoRA, and feature caching other than `"NoCaching"`. Exact per-request condition KV caching is built in; it does not provide cross-request prefix caching.

Serial CFG is separate from CFG parallel: `enable_cfg=true` requires `sample_guide_scale>1`; the recommended presets disable it.

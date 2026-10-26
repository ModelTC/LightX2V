# Self-Forcing (Wan2.1-T2V-1.3B)

This page shows how to run the Self-Forcing Wan2.1 1.3B text-to-video model in LightX2V.

## Prepare the model

Follow the LightX2V [Quick Start](../../docs/EN/source/getting_started/quickstart.md). Download the original Wan2.1 1.3B weights and the Self-Forcing checkpoint:

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./wan_models/Wan2.1-T2V-1.3B
hf download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

The current RTX 5090 profiles use the same 81-frame, 480×832, 16 FPS request geometry and LightTAE decoder:

- [BF16 configuration](../../configs/self_forcing/wan_t2v_sf_bf16_lighttae_5090.json)
- [FP8-FP16 configuration](../../configs/self_forcing/wan_t2v_sf_fp8_f16_accum_lighttae_5090.json)

The FP8 configuration uses converted DiT and T5 checkpoints. Adjust the absolute weight, tokenizer and decoder paths in a copied JSON before running it on another machine. Both profiles enable Sage, warmup, DiT compile/CUDA Graph, streaming LightTAE and top-level asynchronous decode. KV offload is disabled; `vae_use_compile` controls the decoder independently of `use_compile`.

## Run

Run from the repository root and point `MODEL_PATH` at the downloaded Wan weights. The launchers find the repository automatically and select the respective configurations above:

```bash
export MODEL_PATH="$PWD/wan_models/Wan2.1-T2V-1.3B"
bash scripts/self_forcing/run_wan_t2v_sf.sh
bash scripts/self_forcing/run_wan_t2v_sf_fp8.sh
```

To select a copied JSON, set `CONFIG_PATH` when invoking either launcher. The source checkpoint path is the `dit_original_ckpt` field. Self-attention writes RoPE-rotated keys to a rolling ring; Sage stores encoded historical K/V and incrementally updates the newly written slots. See the [AR KV implementation map](../../lightx2v/common/kvcache/AR_IMPLEMENTATIONS.md) for the storage and FlashAttention paths.

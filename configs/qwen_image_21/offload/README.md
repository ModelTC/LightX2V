# Shared CPU block weights

Set the original model directory in the script and `dit_quantized_ckpt` in the config, then run from the repository root:

```bash
bash scripts/qwen_image_21/offload/qwen_image_21_t2i_fp8_f16_accum_5090_block_shared.sh
bash scripts/qwen_image_21/offload/qwen_image_21_i2i_fp8_f16_accum_5090_block_shared.sh
```

These examples use two GPUs, 2048×2048 output, FP8-FP16-accumulation and block offload. Select idle GPUs with `CUDA_VISIBLE_DEVICES`; keep the GPU count, `--nproc_per_node` and `parallel.seq_p_size` consistent. The ordinary converted DiT checkpoint is used without another conversion.

To enable shared weights in another block-offload config:

```json
{
    "cpu_offload": true,
    "offload_granularity": "block",
    "shared_cpu_weights": true,
    "shared_cpu_weight_scope": "auto"
}
```

Support covers CUDA with `DTYPE=BF16`, original BF16 or converted `fp8-sgl` / `fp8-f16-accum` DiT weights, T2I/I2I, and sequence parallelism. TP, LoRA, automatic quantization and disk lazy loading are unsupported with shared weights.

Each replica group holds one copy of the DiT block weights in registered shared CPU memory. `auto` keeps replicas NUMA-local; `host` uses one copy per host/IPC namespace, which may incur remote-NUMA traffic. Processes must belong to the same distributed job and IPC namespace. Independently launched services do not automatically share an arena.

Pre/post weights, text encoder, VAE, condition KV and the two GPU block buffers remain private. This option reduces host physical/pinned memory used by repeated DiT weights; it does not offload the condition KV cache. Compare host PSS or shared-segment bytes rather than summing process RSS, which counts shared pages repeatedly. FP8 scale casting follows the ordinary loader to preserve its numerical results.

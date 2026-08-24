# Attention Mechanisms

## Attention Mechanisms Supported by LightX2V

| Name               | Type Name        | GitHub Link |
|--------------------|------------------|-------------|
| Flash Attention 2  | `flash_attn2`    | [flash-attention v2](https://github.com/Dao-AILab/flash-attention) |
| Flash Attention 3  | `flash_attn3`    | [flash-attention v3](https://github.com/Dao-AILab/flash-attention) |
| Sage Attention 2   | `sage_attn2`     | [SageAttention](https://github.com/thu-ml/SageAttention) |
| Radial Attention   | `radial_attn`    | [Radial Attention](https://github.com/mit-han-lab/radial-attention) |
| Sol-Attn           | `sol_attn`       | [Sol-Attn](https://github.com/NVlabs/Sana/tree/sol-engine/techniques/sparse_backends) |
| Sparge Attention   | `sparge_ckpt`     | [Sparge Attention](https://github.com/thu-ml/SpargeAttn) |

---

## Configuration Examples

The configuration files for attention mechanisms are located [here](https://github.com/ModelTC/lightx2v/tree/main/configs/attentions)

By specifying --config_json to a specific config file, you can test different attention mechanisms.

For example, for radial_attn, the configuration is as follows:

```json
{
  "self_attn_1_type": "radial_attn",
  "cross_attn_1_type": "flash_attn3",
  "cross_attn_2_type": "flash_attn3"
}
```

To switch to other types, simply replace the corresponding values with the type names from the table above.

Tips: radial_attn can only be used in self attention due to the limitations of its sparse algorithm principle.

For further customization of attention mechanism behavior, please refer to the official documentation or implementation code of each attention library.

### Sol-Attn on Wan2.1

Sol-Attn is a forward-only, non-causal self-attention backend. The released kernels require contiguous BF16 tensors with head dimension 128, which matches Wan2.1 self-attention. On H100/H200 (SM90) or RTX 5090 (SM120), install the pinned, architecture-specific Sol-Attn and CUTLASS DSL versions with `scripts/install_sol_attn.sh`, then run:

```bash
MODEL_PATH=/path/to/Wan2.1-I2V-14B-480P \
    bash scripts/wan/run_wan_i2v_sol_attn.sh
```

The example config is `configs/attentions/wan_i2v_sol_attn.json`. It enables Morton3D token ordering and strict mode so an invalid installation fails instead of silently using dense SDPA. Wan applies Morton ordering once around the full transformer block stack and keeps RoPE aligned, instead of moving Q/K/V/output in every attention layer. Following the paper's quality guard, the first 8 denoising steps in the 40-step I2V schedule and transformer layer 0 use FlashAttention 3; all remaining calls use Sol-Attn. Because the paper does not directly evaluate portrait I2V, the example uses a conservative `tau=0.5`; increasing `tau` improves sparsity and speed but may reduce visual quality. `sample_shift=3` matches the standard Wan2.1 I2V configuration so color or exposure changes from the sampling noise schedule are not misattributed to attention. `dense_steps` and `dense_layers` are configurable under `sol_attn_setting`; `dense_layers` accepts both `[0, 1]` and `"0-1"`, while `dense_backend` accepts `flash_attn3`, `sage_attn2`, or `torch_sdpa`. On RTX 5090, set `dense_backend` to `sage_attn2` or `torch_sdpa`, and replace Wan cross-attention `flash_attn3` selections with a supported implementation such as `flash_attn2`. SM120 uses `kv_splits=1` (`"auto"` resolves to 1). The first Sol-Attn call compiles the shape-specific kernel and should be excluded from timing.

### Sol-Attn on MiniMax-H3

The MiniMax-H3 example retains the 15-second, 768p, block CPU-offload setup:

```bash
MODEL_PATH=/path/to/MiniMax-H3 \
    bash scripts/minimax_h3/run_minimax_h3_t2av_sol_attn_offload.sh
```

Its config is `configs/minimax_h3/minimax_h3_t2av_sol_attn_block_offload.json`. Sol-Attn is used only by the 50-layer main transformer; the short text refiner uses dense Torch SDPA. The first six denoising steps and transformer layer 0 use SageAttention2 through `dense_backend=sage_attn2`. H3 attention uses a mixed `[text | audio | video]` packed sequence rather than one 3D video grid, so this config uses `reorder=none`; Wan's Morton3D reorder must not be enabled directly.

# Attention Mechanisms

## Attention Mechanisms Supported by LightX2V

| Name               | Type Name        | GitHub Link |
|--------------------|------------------|-------------|
| Flash Attention 2  | `flash_attn2`    | [flash-attention v2](https://github.com/Dao-AILab/flash-attention) |
| Flash Attention 3  | `flash_attn3`    | [flash-attention v3](https://github.com/Dao-AILab/flash-attention) |
| Sage Attention 2   | `sage_attn2`     | [SageAttention](https://github.com/thu-ml/SageAttention) |
| Radial Attention   | `radial_attn`    | [Radial Attention](https://github.com/mit-han-lab/radial-attention) |
| Sol-Attn           | `sol_attn`       | [Sol-Attn](https://github.com/NVlabs/Sana/tree/sol-engine/techniques/sparse_backends) |
| Sparge Attention   | `sparge_attn`     | [Sparge Attention](https://github.com/thu-ml/SpargeAttn) |

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

For Wan, `cross_attn_2_type` selects image cross-attention. Keep it in Wan2.1 I2V/FLF2V and Wan2.2 Animate/Animate2 configs; T2V/VACE and Wan2.2 MoE, TI2V, and S2V configs with the image encoder disabled do not need it.

Tips: radial_attn can only be used in self attention due to the limitations of its sparse algorithm principle.

For further customization of attention mechanism behavior, please refer to the official documentation or implementation code of each attention library.

### Sol-Attn on Wan2.1

Sol-Attn is a forward-only, non-causal self-attention backend requiring contiguous BF16 tensors and head dimension 128. For RTX 5090 (SM120), use the repository's pinned environment:

```bash
docker build -f dockerfiles/Dockerfile_5090_cu130_sol_attn -t lightx2v-sol-attn:sm120 .
```

Follow [Quick Start](../getting_started/quickstart.md) to run a container, replacing the image with `lightx2v-sol-attn:sm120`. For H100/H200 (SM90), install the backend using the [upstream Sol-Attn instructions](https://github.com/NVlabs/Sana/tree/sol-engine/techniques/sparse_backends).

Before running, set `lightx2v_path`, `model_path`, and `CUDA_VISIBLE_DEVICES` in `scripts/wan/inference/run_wan_i2v_sol_attn.sh`. Then edit `configs/attentions/wan_i2v_sol_attn.json` to match your GPU:

| GPU | `sol_attn_setting.dense_backend` | `cross_attn_1_type` / `cross_attn_2_type` |
| --- | --- | --- |
| H100/H200 | `flash_attn3` (default) | `flash_attn3` (default) |
| RTX 5090 | Change to `sage_attn2` or `torch_sdpa` | Change to an installed, supported backend such as `flash_attn2` |

```bash
bash scripts/wan/inference/run_wan_i2v_sol_attn.sh
```

The first call compiles the shape-specific kernel; exclude it from performance timing.

#### Tuning and implementation

- In the default 40-step config, the first 8 steps and Transformer layer 0 use the dense backend; other calls use Sol-Attn. Adjust `dense_steps` and `dense_layers`; the latter accepts `[0, 1]` or `"0-1"`.
- The example uses `tau=0.8`. Lower it for more conservative sparsity; increasing it may reduce visual quality. `sample_shift=3` matches the standard Wan2.1 I2V config.
- Morton3D ordering is applied once around the entire Transformer block stack, with RoPE kept aligned. Strict mode fails on an invalid installation.
- SM120 uses `kv_splits=1`; `"auto"` also resolves to 1.

### Sol-Attn on MiniMax-H3

The MiniMax-H3 example retains the 15-second, 768p, block CPU-offload setup:

Set `lightx2v_path` and `model_path` in `scripts/minimax_h3/run_minimax_h3_t2av.sh` to your local directories, and change its `--config_json` argument to `"${lightx2v_path}/configs/minimax_h3/minimax_h3_sol_block_offload.json"` before running.

```bash
bash scripts/minimax_h3/run_minimax_h3_t2av.sh
```

Its config is `configs/minimax_h3/minimax_h3_sol_block_offload.json`. Sol-Attn is used only by the 50-layer main transformer; the short text refiner uses dense Torch SDPA. The first six denoising steps and transformer layer 0 use SageAttention2 through `dense_backend=sage_attn2`. H3 attention uses a mixed `[text | audio | video]` packed sequence rather than one 3D video grid, so this config uses `reorder=none`; Wan's Morton3D reorder must not be enabled directly.

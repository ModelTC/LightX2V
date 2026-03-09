
# Self-Forcing (Wan2.1-T2V-1.3B)

This document shows how to run the Self-Forcing acceleration for `Wan2.1-T2V-1.3B` in LightX2V.

## Prepare the Environment

Please refer to [01.PrepareEnv](01.PrepareEnv.md)

## Download

### 1. BF16 original model (Wan2.1-T2V-1.3B)

Download the original BF16 model from the Self-Forcing HuggingFace repo:

```
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
```

### 2. Self-Forcing checkpoint

Download the Self-Forcing checkpoint (`self_forcing_dmd.pt`):

```
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

### 3. Quantized models (FP8 / NVFP4)

Download the quantized models from the LightX2V HuggingFace collection:

https://huggingface.co/collections/lightx2v/ar-lightx2v

## Start Running

We provide three scripts:

1. `scripts/self_forcing/run_wan_t2v_sf.sh`: BF16 original model + Self-Forcing checkpoint
2. `scripts/self_forcing/run_wan_t2v_sf_fp8.sh`: FP8 quantized model
3. `scripts/self_forcing/run_wan_t2v_sf_nvfp4.sh`: NVFP4 quantized model

Before running any script below, you need to fill in the path variables in the script.

### 1) BF16 + Self-Forcing checkpoint

Edit and run:

```
bash scripts/self_forcing/run_wan_t2v_sf.sh
```

Key fields in `run_wan_t2v_sf.sh`:

- `lightx2v_path`: your LightX2V repo path
- `model_path`: path to the original BF16 `Wan2.1-T2V-1.3B` directory
- `sf_model_path`: path to `self_forcing_dmd.pt` (downloaded from `gdhe17/Self-Forcing`)

The script runs:

- `python -m lightx2v.infer`
- `--model_cls wan2.1_sf`: use the Wan2.1 Self-Forcing pipeline
- `--task t2v`: text-to-video
- `--sf_model_path`: the Self-Forcing checkpoint
- `--config_json configs/self_forcing/wan_t2v_sf.json`: Self-Forcing runtime config

Note: In `configs/self_forcing/wan_t2v_sf.json`, `enable_cfg` is set to `false` by default.

### 2) FP8 quantized model

Edit and run:

```
bash scripts/self_forcing/run_wan_t2v_sf_fp8.sh
```

Additional notes for FP8:

- The script uses `configs/self_forcing/wan_t2v_sf_fp8.json`.
- You must change `dit_quantized_ckpt` in that config file to the **absolute path** of the downloaded FP8 weights (from the HuggingFace collection).
- `dit_quant_scheme` is `fp8-vllm`.

### 3) NVFP4 quantized model

Edit and run:

```
bash scripts/self_forcing/run_wan_t2v_sf_nvfp4.sh
```

Additional notes for NVFP4:

- The script uses `configs/self_forcing/wan_t2v_sf_nvfp4.json`.
- You must change `dit_quantized_ckpt` in that config file to the **absolute path** of the downloaded NVFP4 weights (from the HuggingFace collection).
- `dit_quant_scheme` is `nvfp4`.

# Wan2.1 and VACE

[English](README.md) | [简体中文](README_zh.md)

This directory provides inference, server, and distillation examples for Wan2.1 and VACE.

Set `lightx2v_path`, `model_path`, and `CUDA_VISIBLE_DEVICES` in the script. Command paths below are relative to the repository root; absolute script paths also work.

## Environment and weights

Complete the [environment setup](../../docs/EN/source/getting_started/quickstart.md).
Download only the models and additional weights needed for the selected task:

```bash
# Download from HuggingFace
hf download Wan-AI/Wan2.1-T2V-14B --local-dir Wan-AI/Wan2.1-T2V-14B
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir Wan-AI/Wan2.1-I2V-14B-480P
hf download Wan-AI/Wan2.1-I2V-14B-720P --local-dir Wan-AI/Wan2.1-I2V-14B-720P

# Download distilled models
hf download lightx2v/Wan2.1-Distill-Models --local-dir lightx2v/Wan2.1-Distill-Models
hf download lightx2v/Wan2.1-Distill-Loras --local-dir lightx2v/Wan2.1-Distill-Loras
```

Keep `model_path` pointed at the complete base model directory. Set the checkpoint
and LoRA paths in the selected JSON to the downloaded local files.

## Basic tasks

| Task | Script | Example weights |
| --- | --- | --- |
| Text to video | `run_wan_t2v.sh` | Wan2.1-T2V-1.3B |
| Image to video | `run_wan_i2v.sh` | Wan2.1-I2V-14B-480P |
| First and last frames to video | `run_wan_flf2v.sh` | Wan2.1-FLF2V-14B-720P |
| Reference image/video conditioning | `run_wan_vace.sh` | Wan2.1-VACE-14B |

```bash
bash scripts/wan/run_wan_t2v.sh
```

Edit inputs, prompts, seed, and output paths in the shell script, and sampling parameters in the JSON selected by `--config_json`. The JSON fields `num_frames` and `size` supply defaults when omitted from the command or request. `size` is `[height, width]`.

## Server

Scripts and clients are in `server/`, using the weights listed above. The default listening address is `0.0.0.0:8000`.

| Task | Start server | Submit request |
| --- | --- | --- |
| T2V | `start_server_t2v.sh` | `post_t2v.py` |
| I2V | `start_server_i2v.sh` | `post_i2v.py` |
| FLF2V | `start_server_flf2v.sh` | `post_flf2v.py` |
| VACE | `start_server_vace.sh` | `post_vace.py` |
| T2V Reuse | `start_server_t2v_reuse.sh` | `post_t2v_reuse.py` |

T2V example:

```bash
# Terminal 1: start the server
bash scripts/wan/server/start_server_t2v.sh
```

```bash
# Terminal 2: submit a request after the model loads
python scripts/wan/server/post_t2v.py
```

Edit `url` and `message` in the client. Images accept Base64 or absolute paths; VACE videos use absolute paths in `video_path`. Files referenced by absolute paths must be accessible to the server. Set the output location with `save_result_path`.

### Reuse input encodings

T2V uses the Reuse entry above. For I2V, add `"enable_reuse": true` and `"reuse_cache_path": "save_results/reuse_cache/wan_i2v"` to the base JSON, start the regular I2V server, and add `reuse` to the client's `message`. Use a separate cache directory for each model or server.

Use `reuse=False` for the first request and `reuse=True` to reuse input encodings. Keep the prompt, negative prompt, frame count, size, and I2V image paths and contents unchanged. The seed may change; use `False` when changing inputs.

## Multiple GPUs

The two-GPU Ulysses example uses the same weights and inputs as basic T2V, with [wan_t2v_parallel.json](../../configs/wan/wan_t2v_parallel.json):

```bash
bash scripts/wan/run_wan_t2v_parallel.sh
```

For other tasks, use matching weights and base JSON, add `parallel`, and switch to `torchrun` as in the example. The visible GPU count, `--nproc_per_node`, and `parallel.seq_p_size` must match. The Ulysses GPU count must divide the model's attention head count.

## Advanced features

These entries are in `inference/`. Set additional checkpoint paths for distillation and lazy loading in the corresponding JSON.

| Feature | Entry |
| --- | --- |
| Compile | `run_wan_t2v_compile.sh` |
| Lazy loading | `run_wan_i2v_lazy_load.sh` |
| Sol-Attn | `run_wan_i2v_sol_attn.sh` |
| Shared CPU offload | [Usage](inference/shared_offload/README.md) |
| Contiguous memory layout vs. baseline | [Usage](inference/offload_layout/README.md) |

For T2V/I2V, add `"use_compile": true` to enable compilation and `"warmup": true` to enable warmup in the base JSON. I2V uses the regular entry; restart the server after changing its configuration.

### Distillation

`inference/distill/` provides single-GPU 4-step examples. Choose the matching weight type below; model, LoRA, and FP8 examples follow the GPU setup in the previous section.

| Weight type | T2V | I2V |
| --- | --- | --- |
| Full distilled DiT | `run_wan_t2v_distill_model.sh` | `run_wan_i2v_distill_model.sh` |
| Base model + distilled LoRA | `run_wan_t2v_distill_lora.sh` | `run_wan_i2v_distill_lora.sh` |
| FP8 distilled DiT | `run_wan_t2v_distill_fp8.sh` | `run_wan_i2v_distill_fp8.sh` |
| INT8 distilled DiT | — | `run_wan_i2v_distill_int8.sh` |

- `run_wan_i2v_distill_single_file.sh`: a single distilled DiT file, its config, encoders, and VAE in the same model directory.
- `run_wan_t2v_distill_dynamic_cfg.sh`: a dynamic CFG model; `cfg_scale` in the JSON controls guidance strength.
- `run_wan_t2v_distill_model_ulysses.sh`: an eight-GPU Ulysses example for the full distilled T2V model.

## Python API examples

See the [Python API guide](../../docs/EN/source/getting_started/python_api.md) for
generator setup, configuration precedence, offload and repeated generation.
Edit paths in the selected file, then run it from the repository root after installation.
These examples retain their own model, sampling and attention settings; using a
Shell launcher's JSON is an explicit change of configuration.

| Use case | Python entry |
| --- | --- |
| Wan2.1 T2V 14B | [wan_t2v.py](inference/wan_t2v.py) |
| Wan2.1 FLF2V | [wan_flf2v.py](inference/wan_flf2v.py) |
| Wan2.1 VACE | [wan_vace.py](inference/wan_vace.py) |
| Wan2.1 T2V 1.3B NVFP4, 4-step | [wan_t2v_nvfp4.py](inference/distill/wan_t2v_nvfp4.py) |
| Wan2.1 I2V 14B NVFP4, 4-step | [wan_i2v_nvfp4.py](inference/distill/wan_i2v_nvfp4.py) |

The T2V Python example uses the 14B model, `sage_attn2`, guidance 5 and shift 5;
the basic Shell launcher uses the 1.3B model with its JSON settings. Keep these
choices explicit when comparing outputs. For NVFP4, use the named 4-step weights
from [Wan-NVFP4](https://huggingface.co/lightx2v/Wan-NVFP4), keep `distill_method="dmd2"`,
and install the [quantization kernels](../../lightx2v_kernel/README.md).

For CPU offload, set `cpu_offload` and a supported `offload_granularity` in the JSON;
see the [offload guide](../../docs/EN/source/deploy_guides/for_low_resource.md).

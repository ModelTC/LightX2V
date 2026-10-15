# Wan2.2

[English](README.md) | [简体中文](README_zh.md)

Set `lightx2v_path`, `model_path`, and `CUDA_VISIBLE_DEVICES` in the selected script. Commands below use repository-relative paths; absolute script paths also work.

## Basic tasks

| Task | Script | Model weights |
| --- | --- | --- |
| MoE text-to-video | `run_wan22_moe_t2v.sh` | Wan2.2-T2V-A14B |
| MoE image-to-video | `run_wan22_moe_i2v.sh` | Wan2.2-I2V-A14B |
| MoE first/last-frame-to-video | `run_wan22_moe_flf2v.sh` | Wan2.2-I2V-A14B |
| TI2V text-to-video | `run_wan22_t2v.sh` | Wan2.2-TI2V-5B |
| TI2V image-to-video | `run_wan22_i2v.sh` | Wan2.2-TI2V-5B |
| Image + audio-to-video | `run_wan22_s2v.sh` | Wan2.2-S2V-14B |
| Character animation | `run_wan22_animate.sh` | Wan2.2-Animate-14B |
| Character replacement | `run_wan22_animate_replace.sh` | Wan2.2-Animate-14B with matching quantized weights |
| Animate2 | `run_wan22_animate2_distill.sh` | Wan2.2-Animate-2-14B distilled weights |

```bash
bash scripts/wan22/run_wan22_moe_t2v.sh
```

Edit inputs, prompts, seed, and output paths in the script; edit inference settings in the JSON selected by `--config_json`. `size` uses `[height, width]`.

When switching to a preset with `enable_cfg: false`, remove `--negative_prompt` from the script. Also omit `negative_prompt` from service requests, including empty strings.

MoE T2V and I2V use different weights. Both TI2V tasks share [wan_ti2v.json](../../configs/wan22/wan_ti2v.json), or [wan_ti2v_4090.json](../../configs/wan22/wan_ti2v_4090.json) for CPU offload with FlashAttention2.

Animate scripts preprocess the driving video and reference image before inference. The replacement script defaults to [wan_animate_replace_4090.json](../../configs/wan22/wan_animate_replace_4090.json), which needs quantized DiT/T5/CLIP weights; use [wan_animate_replace.json](../../configs/wan22/wan_animate_replace.json) for original weights. Animate2 takes a reference image and driving video directly; set its component paths in the JSON.

## Server

Choose a MoE T2V service; both use port 8000. Single-GPU and TP4 services share `server/post_t2v.py`:

| GPUs | Startup script |
| --- | --- |
| Single GPU | `server/start_server_t2v.sh` |
| TP4 | `server/start_server_t2v_tp.sh` |

```bash
# Terminal 1: start the service
bash scripts/wan22/server/start_server_t2v.sh
```

```bash
# Terminal 2: send a request after the model loads
python scripts/wan22/server/post_t2v.py
```

For I2V or FLF2V, update the single-GPU service's `model_path`, `--task`, and `--config_json` to match the corresponding offline script, then use `scripts/wan/server/post_i2v.py` or `post_flf2v.py`. The task is selected at service startup. For TI2V, also set `--model_cls wan2.2` and use its weights and shared JSON.

Edit `url` and `message` in the client. Images accept Base64 or server-accessible absolute paths; set `save_result_path` for output.

### Reuse

For MoE T2V, select [wan_moe_t2v_reuse.json](../../configs/wan22/wan_moe_t2v_reuse.json) in the single-GPU service and add `reuse` to `message` in `server/post_t2v.py`. For MoE I2V, add `"enable_reuse": true` and `"reuse_cache_path": "save_results/reuse_cache/wan22_moe_i2v"` to its base JSON, then add `reuse` to the normal client's `message`. Use separate cache directories for different models or services.

Use `reuse=False` for the first request and `reuse=True` to reuse input encodings. Keep prompts, frame count, dimensions, and I2V image path/content unchanged; seed may change. Use `False` when changing inputs.

## Multiple GPUs

The four-GPU TP example uses [wan_moe_t2v_tp.json](../../configs/wan22/wan_moe_t2v_tp.json):

```bash
bash scripts/wan22/run_wan22_moe_t2v_tp.sh
```

Match `CUDA_VISIBLE_DEVICES` and `torchrun --nproc_per_node` to the selected JSON. For TP/CFG/SP combinations, the process count is `tensor_p_size × cfg_p_size × seq_p_size`; omitted dimensions count as 1. Ulysses group size must divide the model's attention head count.

Animate has an eight-GPU preset, [wan_animate_8gpu.json](../../configs/wan22/wan_animate_8gpu.json): select it in `run_wan22_animate.sh`, set eight GPUs, and replace the inference command with `torchrun --nproc_per_node=8 -m lightx2v.infer`. Keep preprocessing on `python`. For two-GPU LoRA, use the same script with `wan_animate_lora.json`, add `"parallel": {"seq_p_size": 2, "seq_p_attn_type": "ulysses"}`, and select two GPUs with two inference processes.

## Advanced features

Advanced scripts are in `inference/`; select other presets with `--config_json`. Configure matching local distilled, LoRA, or quantized weights in the selected JSON.

### Compile

Select [wan_moe_t2v_compile.json](../../configs/wan22/wan_moe_t2v_compile.json) in `run_wan22_moe_t2v.sh`. For MoE I2V, add `"use_compile": true` to its base JSON. Add `"warmup": true` for startup warmup; restart services after changing startup settings.

### Distillation

Single-GPU scripts are in `inference/distill/`: use `run_wan22_moe_t2v_distill.sh` for T2V (LoRA by default) or `run_wan22_moe_i2v_distill.sh` for I2V (full distilled weights by default). Set `--config_json` in the script to select a preset:

| Weights | T2V preset | I2V preset |
| --- | --- | --- |
| Full distilled weights | [wan_moe_t2v_distill.json](../../configs/wan22/wan_moe_t2v_distill.json) | [wan_moe_i2v_distill_model.json](../../configs/distill/wan22/wan_moe_i2v_distill_model.json) |
| LoRA | [wan_moe_t2v_distill_lora.json](../../configs/distill/wan22/wan_moe_t2v_distill_lora.json) | [wan_moe_i2v_distill_with_lora.json](../../configs/distill/wan22/wan_moe_i2v_distill_with_lora.json) |
| FP8 | — | [wan_moe_i2v_distill_quant.json](../../configs/distill/wan22/wan_moe_i2v_distill_quant.json) |
| NVFP4 + sparse attention | [wan_moe_t2v_distill_nvfp4_sparse_attn.json](../../configs/wan22/extreme/wan_moe_t2v_distill_nvfp4_sparse_attn.json) | [wan_moe_i2v_distill_nvfp4_sparse_attn.json](../../configs/wan22/extreme/wan_moe_i2v_distill_nvfp4_sparse_attn.json) |

Other distillation scripts are also in `inference/distill/`:

| Task / weights | Script |
| --- | --- |
| FLF2V full distilled weights | `run_wan22_distill_moe_flf2v.sh` |
| T2V LoRA, CFG2 × SP4 | `run_wan22_moe_t2v_distill_lora_4step_cfg_ulysses.sh` |

FLF2V presets are in [configs/wan22](../../configs/wan22), including FP8, INT8, and LoRA variants. Keep their task-specific sampling settings and weight paths.

To run I2V full weights, LoRA, or FP8 with CFG2 × SP4, update the corresponding single-GPU JSON with:

```json
{
  "enable_cfg": true,
  "parallel": {
    "seq_p_size": 4,
    "seq_p_attn_type": "ulysses",
    "cfg_p_size": 2
  }
}
```

Set eight GPUs and replace `python -m lightx2v.infer` with `torchrun --nproc_per_node=8 -m lightx2v.infer`. This enables CFG as well as parallel inference; keep the selected model's weights and sampling settings.

### Animate LoRA

Select the corresponding complete JSON with `--config_json` in the base script, then set the local weight paths in `lora_configs`:

| Script | Preset |
| --- | --- |
| `run_wan22_animate.sh` | [wan_animate_lora.json](../../configs/wan22/wan_animate_lora.json) |
| `run_wan22_animate_replace.sh` | [wan_animate_replace_lora.json](../../configs/wan22/wan_animate_replace_lora.json) |

### Other examples

Paths below are relative to `inference/`:

| Feature | Entry |
| --- | --- |
| S2V with pose video | `run_wan22_s2v_pose_audio.sh` |
| Animate preprocessing | `run_wan22_animate_preprocess.sh` |
| Animate2, eight GPUs | `run_wan22_animate2_distill_8gpu.sh` |
| CUDA RainFusion, four GPUs | `distill/run_wan22_moe_i2v_distill_4step_ulysses_rainfusion.sh` |
| Ascend INT8 + RainFusion, two NPUs | `distill/run_wan22_moe_i2v_distill_int8_4step_ulysses_npu.sh` |
| NVFP4 + sparse attention, eight GPUs | `extreme/`: T2V/I2V examples |

RainFusion, NPU, and Extreme examples retain their own operator and communication settings. Use matching hardware and dependencies; NPU scripts select devices with `ASCEND_RT_VISIBLE_DEVICES`.

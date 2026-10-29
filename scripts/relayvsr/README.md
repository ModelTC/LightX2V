# RelayVSR video super-resolution

[English](README.md) | [简体中文](README_zh.md)

## Model downloads

[Official RelayVSR GitHub repository](https://github.com/kopperx/RelayVSR)

[relayvsr_x4 on Hugging Face](https://huggingface.co/kopper/RelayVSR)

## Model variants

| `model_cls` | Checkpoints | Output contract |
| --- | --- | --- |
| `relayvsr_x4` | RelayVSR HF `wan.safetensors`, `flash.safetensors` | Compute 4x; select 2x or 4x output with `sr_ratio=2/4` |
| `relayvsr_x2` | LightX2V HF `wan_x2.safetensors`, `flash_x2.safetensors` | Compute 2x; require `sr_ratio=2` |

Both entries validate checkpoint formats, scale and architecture, and reject mixed 2x/4x weights.
Both entries reject `size`; output dimensions are determined by the input dimensions and `sr_ratio`.

Configuration filenames distinguish `_x4` from `_x2`; choose the one matching `model_cls`.

## 4× usage

Set the paths in the script, then run from the repository root:

```bash
bash scripts/relayvsr/run_relayvsr_x4_sr.sh
```

The x4 model also supports 2× output: change `--sr_ratio 4` to `--sr_ratio 2` in the script. It computes at 4× internally, then downsizes the result to 2×.

## 2× usage

Set the paths in the script, then run from the repository root:

```bash
bash scripts/relayvsr/run_relayvsr_x2_sr.sh
```

## Execution modes

| Config | Behavior |
| --- | --- |
| `relayvsr_x4.json` | Weights remain on GPU |
| `relayvsr_x4_block_offload.json` | Wan uses block weight offload; Flash remains on GPU |
| `relayvsr_x4_sp.json` | Two-GPU sequence parallelism |

The 2x model has corresponding `_x2` configurations. Select a mode through `--config_json` in the script.

All configurations enable compilation and warmup by default. Set `use_compile` and `warmup` to `false` in the selected configuration to disable them.

With `PROFILING_DEBUG_LEVEL=1` or `2`, RelayVSR logs the full `run_pipeline` duration, excluding model loading and startup warmup.

## Serving and requests

Set the repository and model paths, then start a single-GPU server:

```bash
lightx2v_path=/path/to/LightX2V
model_path=/models/RelayVSR

export CUDA_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.server \
  --model_cls relayvsr_x4 --task sr \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/relayvsr/relayvsr_x4.json" \
  --host 0.0.0.0 --port 8000
```

The startup configuration enables compilation and warmup. After model loading and warmup finish, check the service status and submit a 4× request from another terminal. Both input and output paths must be absolute paths accessible on the server:

```bash
curl -fsS http://127.0.0.1:8000/v1/service/status

curl -fsS -X POST http://127.0.0.1:8000/v1/tasks/video/ \
  -H 'Content-Type: application/json' \
  -d '{
    "video_path": "/videos/input.mp4",
    "sr_ratio": 4,
    "save_result_path": "/videos/output_x4.mp4",
    "seed": 42
  }'
```

Submit further requests to the resident server to reuse the loaded model, setting the input path, output path and `seed` for each request. Set `sr_ratio` to `2` for 2× output.

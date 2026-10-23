# RelayVSR video super-resolution

[中文说明](README_zh.md)

Native integration of the official [RelayVSR](https://github.com/kopperx/RelayVSR) big model: Wan, with approximately 5.03B parameters, generates sparse reference frames, while Flash-S, with approximately 31.9M parameters, decodes all video frames.

**The model always computes at 4× internally.** `--sr_ratio 4` outputs the result directly; `--sr_ratio 2` removes padding from the 4× result and downsizes it using FP32 antialiased bicubic interpolation before encoding. 2× still requires the full 4× model computation, but reduces output transfer, encoding costs and file size.

## Setup and running

Use the LightX2V CUDA environment with PyTorch, Triton, PyAV and FFmpeg. BF16 is the default. Download the two official checkpoints; no weight conversion is required:

```bash
hf download kopper/RelayVSR wan.safetensors flash.safetensors --local-dir /models/RelayVSR
```

Edit `lightx2v_path`, `model_path` and `video_path` at the top of `scripts/relayvsr/run_relayvsr_sr.sh`, then run:

```bash
bash scripts/relayvsr/run_relayvsr_sr.sh
```

GPU selection and inference arguments are set directly in the script, which initializes the environment through `scripts/base/base.sh`. Change `--sr_ratio` to `4` for 4× output and set the output path with `--save_result_path`. You can also use the unified CLI directly:

```bash
python -m lightx2v.infer \
  --model_cls relayvsr --task sr \
  --model_path /models/RelayVSR \
  --config_json configs/relayvsr/relayvsr.json \
  --video_path /videos/input.mp4 --sr_ratio 2 --seed 42 \
  --save_result_path /videos/output_x2.mp4
```

For images, replace `--video_path` with `--image_path input.png` and save the output as PNG.

## Execution modes

| Config | Behavior |
| --- | --- |
| `relayvsr.json` | Weights remain on GPU, with eager execution; compilation and warmup are disabled |
| `relayvsr_compile.json` | Enables `torch.compile` for both Wan and Flash Transformer blocks, with startup warmup |
| `relayvsr_block_offload.json` | Wan weights stay on CPU and are loaded onto the GPU block by block using LightX2V double buffering and prefetch; Flash remains on GPU |
| `relayvsr_sp.json` | Two-GPU sequence parallelism; both Wan and Flash reuse LightX2V Ulysses all-to-all |

Select a configuration by changing `--config_json` in the script. For example, set it to `"${lightx2v_path}/configs/relayvsr/relayvsr_block_offload.json"` to enable block weight offload.

The example script uses `relayvsr_compile.json` by default, enabling compilation and startup warmup. `use_compile` and `warmup` are startup JSON settings: setting both to `true` triggers compilation during warmup after model loading; enabling only `warmup` warms up eager execution; enabling only `use_compile` triggers compilation during the first inference. To disable both, change the script's `--config_json` back to `"${lightx2v_path}/configs/relayvsr/relayvsr.json"`. Compilation covers all 30 Wan blocks and all 10 Flash blocks, including Flash condition and video paths. Pre/postprocessing, weight transfers and video I/O stay outside the compiled blocks.

Two-GPU example:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m lightx2v.infer \
  --model_cls relayvsr --task sr --model_path /models/RelayVSR \
  --config_json configs/relayvsr/relayvsr_sp.json \
  --video_path /videos/input.mp4 --sr_ratio 2 --seed 42 \
  --save_result_path /videos/output_x2.mp4
```

Flash has only two KV groups, so sequence parallelism supports at most two GPUs. Partitioned linear operations may introduce BF16 numerical differences; use the default single-GPU eager mode when elementwise parity is required.

Each configuration explicitly lists the cross-attention backend, offload switch, clip length and encoding parameters for easy editing. Defaults are `torch_sdpa`, 16-frame clips and source FPS. CPU encoding uses `libx265`, `ultrafast` and CRF 18, with the `hvc1` tag; `encoder_threads` controls the x265 worker pool.

## Serving and requests

Set the repository and model paths, then start a single-GPU server:

```bash
lightx2v_path=/path/to/LightX2V
model_path=/models/RelayVSR

export CUDA_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.server \
  --model_cls relayvsr --task sr \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/relayvsr/relayvsr_compile.json" \
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

The submission endpoint returns a `task_id` asynchronously. Set the variable below to that value to query the task status:

```bash
task_id=REPLACE_WITH_RETURNED_TASK_ID
curl -fsS "http://127.0.0.1:8000/v1/tasks/${task_id}/status"
```

When the status endpoint reports `status` as `completed`, the video has been saved to the server-side `save_result_path`. If it reports `failed`, check the returned `error`. Submit further requests to the resident server to reuse the loaded model, setting the input path, output path and `seed` for each request. Set `sr_ratio` to `2` for 2× output. Compilation, warmup and encoding options are startup JSON settings; restart the server after changing them.

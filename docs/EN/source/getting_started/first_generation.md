# Generate your first video

This walkthrough uses Wan2.1-T2V-1.3B. Complete the [environment setup](quickstart.md) first, then run the commands below from the LightX2V repository root.

## Download the model

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir /path/to/models/Wan2.1-T2V-1.3B
```

Alternatively, download `Wan-AI/Wan2.1-T2V-1.3B` through ModelScope and use its local directory as `model_path`.

## Configure and run

Open [run_wan_t2v.sh](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan/run_wan_t2v.sh) and set:

| Setting | Value |
| --- | --- |
| `lightx2v_path` | Absolute path to this repository |
| `model_path` | Absolute path to the downloaded model |
| `CUDA_VISIBLE_DEVICES` | GPU to use, such as `0` |
| `--prompt` / `--negative_prompt` | Text describing the scene and unwanted content |
| `--size` | Output size as `HEIGHT WIDTH`; the launcher uses `480 832` |
| `--num_frames` | Output frame count; the launcher uses `81` |
| `--save_result_path` | Output video path |

The launcher selects [wan_t2v.json](https://github.com/ModelTC/LightX2V/blob/main/configs/wan/wan_t2v.json). Edit startup settings such as sampling steps and guidance scale in JSON, and request parameters in the launcher.

The launcher's explicit `--size` and `--num_frames` override the corresponding JSON defaults, so changing only JSON does not change this launcher's output size or frame count. Edit those arguments in the launcher. To control both through JSON instead, remove the two arguments from the launcher, then set `size` (`[height, width]`) and `num_frames` in JSON.

The supplied JSON uses `flash_attn3`. Select attention operators supported by your GPU and installation; see the [attention guide](../method_tutorials/attention.md). For lower GPU memory usage, see the [offload guide](../deploy_guides/for_low_resource.md).

```bash
bash scripts/wan/run_wan_t2v.sh
```

Wait for inference to finish and open the file at `--save_result_path`. The supplied launcher writes to `save_results/output_lightx2v_wan_t2v.mp4` under the repository.

## Continue

- [Wan2.1 and VACE](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan/README.md): model variants, distilled weights, services and multiple GPUs.
- [Wan2.2](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan22/README.md) and [Qwen Image](https://github.com/ModelTC/LightX2V/blob/main/scripts/qwen_image/README.md): model-specific launchers and configurations.
- [Python API](python_api.md): embed generation in an application and reuse a pipeline.
- [Service deployment](../deploy_guides/deploy_service.md): serve generation requests. The Wan-specific server and client are listed in the Wan README.

Use the same model, JSON and request parameters, including seed, size and frame count, when comparing CLI, service and Python execution. The manually configured Python examples may intentionally use different sampling settings.

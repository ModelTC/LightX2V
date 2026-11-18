# Z-Image-Turbo

[English](README.md) | [简体中文](README_zh.md)

## Prepare

Complete the [environment setup](../../docs/EN/source/getting_started/quickstart.md) and download the complete [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) model directory.

| Task | Shell entry |
| --- | --- |
| Text to image | [z_image_turbo_t2i.sh](z_image_turbo_t2i.sh) |
| Text to image, two GPUs | [z_image_turbo_t2i_dist.sh](z_image_turbo_t2i_dist.sh) |
| Image to image | [z_image_turbo_i2i.sh](z_image_turbo_i2i.sh) |

Set `lightx2v_path`, `model_path`, and `CUDA_VISIBLE_DEVICES` in the selected script. Edit the prompt and, for image-to-image generation, `image_path` and `--i2i_denoise_strength`.

## Run

Run from the repository root:

```bash
bash scripts/z_image/z_image_turbo_t2i.sh
```

The basic text-to-image example saves to `save_results/z_image_turbo.png`; change `--save_result_path` to choose another location. Sampling settings are in the JSON selected by `--config_json` under [configs/z_image](../../configs/z_image).

## Python API

[z_image_turbo.py](inference/z_image_turbo.py) reads the [FP8 configuration](../../configs/z_image/z_image_turbo_t2i_fp8.json). Download FP8 weights from [Z-Image-Turbo-Quantized](https://huggingface.co/lightx2v/Z-Image-Turbo-Quantized), then set `dit_quantized_ckpt` in that JSON. Set `model_path` and `save_result_path` in the script; configure offload in the JSON if needed.

```bash
python scripts/z_image/inference/z_image_turbo.py
```

See the [Python API guide](../../docs/EN/source/getting_started/python_api.md) for shared usage.

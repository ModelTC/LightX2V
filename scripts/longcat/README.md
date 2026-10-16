# LongCat Image

[English](README.md) | [简体中文](README_zh.md)

## Prepare

Complete the [environment setup](../../docs/EN/source/getting_started/quickstart.md), then download the model for your task:

| Task | Weights | Shell entry | Python entry |
| --- | --- | --- | --- |
| Text to image | [LongCat-Image](https://huggingface.co/meituan-longcat/LongCat-Image) | [longcat_image_t2i.sh](longcat_image_t2i.sh) | [longcat_t2i.py](inference/longcat_t2i.py) |
| Image editing | [LongCat-Image-Edit](https://huggingface.co/meituan-longcat/LongCat-Image-Edit) | [longcat_image_i2i.sh](longcat_image_i2i.sh) | [longcat_i2i.py](inference/longcat_i2i.py) |

Set `lightx2v_path`, `model_path`, and `CUDA_VISIBLE_DEVICES` in the selected Shell script. Point `model_path` to the complete local model directory. Edit the prompt and, for image editing, `--image_path`.

## Run

Run from the repository root:

```bash
bash scripts/longcat/longcat_image_t2i.sh
```

The basic scripts save images to `save_results/longcat_image_t2i.png` or `save_results/longcat_image_i2i.png`. Change `--save_result_path` to choose another location. Sampling settings are in the JSON selected by `--config_json` under [configs/longcat_image](../../configs/longcat_image).

For CPU offload, use [longcat_image_t2i_offload.sh](longcat_image_t2i_offload.sh). For two-GPU CFG parallelism, use [longcat_image_t2i_cfg_parallel.sh](longcat_image_t2i_cfg_parallel.sh) or [longcat_image_i2i_cfg_parallel.sh](longcat_image_i2i_cfg_parallel.sh).

## Python API

Edit the model, input, and output paths in the selected Python example, then run:

```bash
python scripts/longcat/inference/longcat_t2i.py
```

See the [Python API guide](../../docs/EN/source/getting_started/python_api.md) for shared usage.

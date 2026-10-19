# SeedVR2 Super-Resolution

[English](README.md) | [简体中文](README_zh.md)

## Prepare

Complete the [environment setup](../../docs/EN/source/getting_started/quickstart.md) and download [SeedVR2-3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B). The local model directory must contain `seedvr2_ema_3b.pth`, `ema_vae.pth`, and `pos_emb.pt`.

| Task | Shell entry | Config |
| --- | --- | --- |
| 3B video super-resolution | [run_seedvr2_3b_sr.sh](run_seedvr2_3b_sr.sh) | [seedvr2_3b.json](../../configs/seedvr/seedvr2_3b.json) |
| 3B image super-resolution | [run_seedvr2_3b_image_sr.sh](run_seedvr2_3b_image_sr.sh) | [seedvr2_3b.json](../../configs/seedvr/seedvr2_3b.json) |
| 7B video super-resolution | [run_seedvr2_7b_sr.sh](run_seedvr2_7b_sr.sh) | [seedvr2_7b.json](../../configs/seedvr/seedvr2_7b.json) |
| 7B video super-resolution, eight GPUs | [run_seedvr2_7b_sp.sh](run_seedvr2_7b_sp.sh) | [seedvr2_7b_sp.json](../../configs/seedvr/seedvr2_7b_sp.json) |

For 7B, also download [SeedVR2-7B](https://huggingface.co/ByteDance-Seed/SeedVR2-7B) and set `dit_original_ckpt` in the selected JSON to `seedvr2_ema_7b.pth`. Keep `model_path` pointing to the 3B directory for `ema_vae.pth` and `pos_emb.pt`.

## Run

Set `lightx2v_path`, `model_path`, `video_path` (or `image_path`), and `CUDA_VISIBLE_DEVICES` in the selected Shell script. Run from the repository root:

```bash
bash scripts/seedvr2/run_seedvr2_3b_sr.sh
```

The 3B video example uses `--sr_ratio 2.0` and saves to `save_results/output_lightx2v_seedvr2_sr.mp4`. Edit `--sr_ratio` and `--save_result_path` to change the scale and output location. Image inputs use `--image_path` and a `.png` output.

## Python API

Edit `model_path`, `input_video_path`, and `save_result_path` in [seedvr_sr_inference.py](inference/seedvr_sr_inference.py), then run:

```bash
python scripts/seedvr2/inference/seedvr_sr_inference.py
```

The Python example uses `configs/seedvr/seedvr2_3b.json`; its relative input and output paths are resolved from the current working directory. See the [Python API guide](../../docs/EN/source/getting_started/python_api.md) for shared usage.

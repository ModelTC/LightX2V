# SeedVR2 超分辨率

[English](README.md) | [简体中文](README_zh.md)

## 准备

完成[环境安装](../../docs/ZH_CN/source/getting_started/quickstart.md)，下载 [SeedVR2-3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B)。本地模型目录需包含 `seedvr2_ema_3b.pth`、`ema_vae.pth` 和 `pos_emb.pt`。

| 任务 | Shell 入口 | 配置 |
| --- | --- | --- |
| 3B 视频超分 | [run_seedvr2_3b_sr.sh](run_seedvr2_3b_sr.sh) | [seedvr2_3b.json](../../configs/seedvr/seedvr2_3b.json) |
| 3B 图像超分 | [run_seedvr2_3b_image_sr.sh](run_seedvr2_3b_image_sr.sh) | [seedvr2_3b.json](../../configs/seedvr/seedvr2_3b.json) |
| 7B 视频超分 | [run_seedvr2_7b_sr.sh](run_seedvr2_7b_sr.sh) | [seedvr2_7b.json](../../configs/seedvr/seedvr2_7b.json) |
| 7B 视频超分，八卡 | [run_seedvr2_7b_sp.sh](run_seedvr2_7b_sp.sh) | [seedvr2_7b_sp.json](../../configs/seedvr/seedvr2_7b_sp.json) |

使用 7B 时，额外下载 [SeedVR2-7B](https://huggingface.co/ByteDance-Seed/SeedVR2-7B)，并在所选 JSON 中将 `dit_original_ckpt` 指向 `seedvr2_ema_7b.pth`。`model_path` 仍指向 3B 目录，用于读取 `ema_vae.pth` 和 `pos_emb.pt`。

## 运行

在所选 Shell 脚本中设置 `lightx2v_path`、`model_path`、`video_path`（或 `image_path`）和 `CUDA_VISIBLE_DEVICES`，然后在仓库根目录运行：

```bash
bash scripts/seedvr2/run_seedvr2_3b_sr.sh
```

3B 视频示例使用 `--sr_ratio 2.0`，结果保存到 `save_results/output_lightx2v_seedvr2_sr.mp4`。修改 `--sr_ratio` 和 `--save_result_path` 可调整放大倍率及输出位置。图像输入使用 `--image_path`，输出路径以 `.png` 结尾。

## Python API

修改 [seedvr_sr_inference.py](inference/seedvr_sr_inference.py) 中的 `model_path`、`input_video_path` 和 `save_result_path`，然后运行：

```bash
python scripts/seedvr2/inference/seedvr_sr_inference.py
```

Python 示例使用 `configs/seedvr/seedvr2_3b.json`；其中的相对输入、输出路径以当前工作目录为准。通用用法见 [Python API 指南](../../docs/ZH_CN/source/getting_started/python_api.md)。

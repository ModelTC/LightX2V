# Z-Image-Turbo

[English](README.md) | [简体中文](README_zh.md)

## 准备

完成[环境安装](../../docs/ZH_CN/source/getting_started/quickstart.md)，下载完整的 [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) 模型目录。

| 任务 | Shell 入口 |
| --- | --- |
| 文生图 | [z_image_turbo_t2i.sh](z_image_turbo_t2i.sh) |
| 文生图，双卡 | [z_image_turbo_t2i_dist.sh](z_image_turbo_t2i_dist.sh) |
| 图生图 | [z_image_turbo_i2i.sh](z_image_turbo_i2i.sh) |

在所选脚本中设置 `lightx2v_path`、`model_path` 和 `CUDA_VISIBLE_DEVICES`。修改提示词；图生图还需设置 `image_path` 和 `--i2i_denoise_strength`。

## 运行

在仓库根目录运行：

```bash
bash scripts/z_image/z_image_turbo_t2i.sh
```

基础文生图示例将结果保存到 `save_results/z_image_turbo.png`，可通过 `--save_result_path` 修改。采样参数在 `--config_json` 指定的 JSON 中，配置目录为 [configs/z_image](../../configs/z_image)。

## Python API

[z_image_turbo.py](inference/z_image_turbo.py) 使用 FP8 DiT 权重和 CPU 卸载。额外从 [Z-Image-Turbo-Quantized](https://huggingface.co/lightx2v/Z-Image-Turbo-Quantized) 下载 `z_image_turbo_scaled_fp8_e4m3fn.safetensors`，并修改示例中的 `model_path`、`dit_quantized_ckpt` 和 `save_result_path`。

```bash
python scripts/z_image/inference/z_image_turbo.py
```

通用用法见 [Python API 指南](../../docs/ZH_CN/source/getting_started/python_api.md)。

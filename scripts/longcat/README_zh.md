# LongCat Image

[English](README.md) | [简体中文](README_zh.md)

## 准备

完成[环境安装](../../docs/ZH_CN/source/getting_started/quickstart.md)，按任务下载模型：

| 任务 | 权重 | Shell 入口 | Python 入口 |
| --- | --- | --- | --- |
| 文生图 | [LongCat-Image](https://huggingface.co/meituan-longcat/LongCat-Image) | [longcat_image_t2i.sh](longcat_image_t2i.sh) | [longcat_t2i.py](inference/longcat_t2i.py) |
| 图像编辑 | [LongCat-Image-Edit](https://huggingface.co/meituan-longcat/LongCat-Image-Edit) | [longcat_image_i2i.sh](longcat_image_i2i.sh) | [longcat_i2i.py](inference/longcat_i2i.py) |

在所选 Shell 脚本中设置 `lightx2v_path`、`model_path` 和 `CUDA_VISIBLE_DEVICES`。`model_path` 指向完整的本地模型目录。修改提示词；图像编辑还需设置 `--image_path`。

## 运行

在仓库根目录运行：

```bash
bash scripts/longcat/longcat_image_t2i.sh
```

基础脚本将图片保存到 `save_results/longcat_image_t2i.png` 或 `save_results/longcat_image_i2i.png`，可通过 `--save_result_path` 修改。采样参数在 `--config_json` 指定的 JSON 中，配置目录为 [configs/longcat_image](../../configs/longcat_image)。

CPU 卸载使用 [longcat_image_t2i_offload.sh](longcat_image_t2i_offload.sh)；双卡 CFG 并行使用 [longcat_image_t2i_cfg_parallel.sh](longcat_image_t2i_cfg_parallel.sh) 或 [longcat_image_i2i_cfg_parallel.sh](longcat_image_i2i_cfg_parallel.sh)。

## Python API

修改所选 Python 示例中的模型、输入和输出路径，然后运行：

```bash
python scripts/longcat/inference/longcat_t2i.py
```

通用用法见 [Python API 指南](../../docs/ZH_CN/source/getting_started/python_api.md)。

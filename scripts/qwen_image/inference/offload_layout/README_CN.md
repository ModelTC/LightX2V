# Qwen Image CPU Offload

[English](README.md) | 简体中文

单卡运行 Qwen-Image-2512 文生图，对比两种 CPU block offload：`baseline` 逐 tensor 搬运权重，`contiguous` 使用连续 pinned 内存并整 block 搬运。

安装对应平台的项目依赖，准备完整 BF16 权重（包含文本编码器和 VAE）。以下命令使用仓库相对路径，也可使用脚本绝对路径。

## NVIDIA GPU

先修改两个脚本开头的 `lightx2v_path`、`model_path` 和 GPU 编号。

```bash
bash scripts/qwen_image/inference/offload_layout/run_baseline.sh
bash scripts/qwen_image/inference/offload_layout/run_contiguous.sh
```

## Ascend NPU

在脚本中设置 `lightx2v_path`、`model_path` 和 `ASCEND_RT_VISIBLE_DEVICES`：

```bash
bash scripts/qwen_image/inference/offload_layout/run_baseline_npu.sh
bash scripts/qwen_image/inference/offload_layout/run_contiguous_npu.sh
```

配置位于 `configs/qwen_image/layout/`，提示词在脚本的 `--prompt` 中设置。图片保存在 `save_results/qwen_layout/`，重复运行会覆盖同名结果。

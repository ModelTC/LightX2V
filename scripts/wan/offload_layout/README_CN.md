# Wan CPU Offload

[English](README.md) | 简体中文

单卡运行 Wan 图生视频，对比两种 CPU block offload：`baseline` 逐 tensor 搬运权重，`contiguous` 使用连续 pinned 内存并整 block 搬运。

安装对应平台的项目依赖后，在仓库根目录执行以下命令。

## NVIDIA GPU

先修改两个脚本开头的 `lightx2v_path`、`model_path` 和 GPU 编号。默认使用 `Wan2.1-I2V-14B-720P-Lightx2v` 的 FP8 权重。

```bash
bash scripts/wan/offload_layout/run_baseline.sh
bash scripts/wan/offload_layout/run_contiguous.sh
```

## Ascend NPU

使用原始 `Wan2.1-I2V-14B-720P` BF16 权重，包含文本编码器和 VAE。脚本默认使用第 0 张卡。

```bash
MODEL_PATH="$PWD/models/Wan2.1-I2V-14B-720P" \
  bash scripts/wan/offload_layout/run_baseline_npu.sh
MODEL_PATH="$PWD/models/Wan2.1-I2V-14B-720P" \
  bash scripts/wan/offload_layout/run_contiguous_npu.sh
```

配置位于 `configs/wan/layout/`，输入图片在脚本的 `--image_path` 中设置。视频保存在 `save_results/wan_layout/`，重复运行会覆盖同名结果。

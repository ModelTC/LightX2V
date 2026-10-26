# Wan CPU Offload

[English](README.md) | 简体中文

单卡运行 Wan 图生视频，对比两种 CPU block offload：`baseline` 逐 tensor 搬运权重，`contiguous` 使用连续 pinned 内存并整 block 搬运。

安装对应平台的项目依赖后，在仓库根目录执行以下命令。

## NVIDIA GPU

使用 `Wan2.1-I2V-14B-720P-Lightx2v` 的 FP8 权重，先设置两个脚本开头的 `lightx2v_path`、`model_path` 和 `CUDA_VISIBLE_DEVICES`。

在 `configs/wan/layout/baseline.json` 和 `contiguous.json` 中，将 `dit_quantized_ckpt` 设置为 FP8 DiT 目录的绝对路径，将 `clip_quantized_ckpt` 设置为实际量化 CLIP 文件的路径。示例文件名为 `clip-fp8.pth`，请按实际文件名填写。这些路径独立于 `model_path`。

```bash
bash scripts/wan/inference/offload_layout/run_baseline.sh
bash scripts/wan/inference/offload_layout/run_contiguous.sh
```

## Ascend NPU

使用原始 `Wan2.1-I2V-14B-720P` BF16 权重，包含文本编码器和 VAE。在两个 NPU 脚本开头设置 `lightx2v_path`、`model_path` 和 `ASCEND_RT_VISIBLE_DEVICES`，默认使用第 0 张卡。

```bash
bash scripts/wan/inference/offload_layout/run_baseline_npu.sh
bash scripts/wan/inference/offload_layout/run_contiguous_npu.sh
```

配置位于 `configs/wan/layout/`，输入图片在脚本的 `--image_path` 中设置。视频保存在 `save_results/wan_layout/`，重复运行会覆盖同名结果。

`cpu_offload_block_contiguous` 在 baseline 配置中为 `false`，在 contiguous 配置中为 `true`。

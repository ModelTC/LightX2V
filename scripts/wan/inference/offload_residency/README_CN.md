# Wan 常驻权重

[English](README.md) | 简体中文

单卡运行 Wan2.1 图生视频，对比零常驻 baseline 和部分 block 常驻策略。baseline 按需从 CPU 搬运权重；常驻策略将估测出的 block 数量均匀分布在模型各层，减少重复搬运。

在脚本顶部设置 `lightx2v_path`、`model_path`。估测脚本的 `--image-path` 和推理脚本的 `--image_path` 使用同一图片，两份推理脚本的提示词和 seed 保持一致。

下面命令在仓库根目录执行：先估测并生成常驻配置，再运行对比。baseline 也可以单独运行。

## NVIDIA GPU

使用 `Wan2.1-I2V-14B-720P-Lightx2v` FP8 权重，设置三个脚本中的 `CUDA_VISIBLE_DEVICES`。在 `configs/wan/residency/baseline.json` 中分别填写 `dit_quantized_ckpt`、`t5_quantized_ckpt`、`clip_quantized_ckpt`，建议使用绝对路径；这些路径不会随 `model_path` 自动改变。

```bash
bash scripts/wan/inference/offload_residency/estimate.sh
bash scripts/wan/inference/offload_residency/run_baseline.sh
bash scripts/wan/inference/offload_residency/run.sh
```

## Ascend NPU

使用完整的 `Wan2.1-I2V-14B-720P` BF16 权重，包含文本编码器、CLIP 和 VAE。设置三个 NPU 脚本中的 `ASCEND_RT_VISIBLE_DEVICES`。

```bash
bash scripts/wan/inference/offload_residency/estimate_npu.sh
bash scripts/wan/inference/offload_residency/run_baseline_npu.sh
bash scripts/wan/inference/offload_residency/run_npu.sh
```

基础配置位于 `configs/wan/residency/`，CUDA 使用 `baseline.json`，NPU 使用 `baseline_npu.json`。结果保存在 `save_results/wan_residency/`：`resident.json` 为常驻配置，`resident.profile.json` 为测量报告，`baseline.mp4` 和 `resident.mp4` 为推理结果；NPU 文件带 `_npu` 后缀。重复运行会覆盖同名文件。

生成配置中的 `offload_resident_blocks` 是常驻数量，`0` 表示零常驻。修改设备、输入或基础配置后重新估测。profiling 统一沿用 [base.sh](../../../base/base.sh)。

`cpu_offload_block_contiguous` 控制连续布局，默认 `false`；设为 `true` 后重新估测，baseline 与常驻使用相同值。

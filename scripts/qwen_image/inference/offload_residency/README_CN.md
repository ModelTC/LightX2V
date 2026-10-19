# Qwen Image 常驻权重

[English](README.md) | 简体中文

单卡运行 Qwen Image 文生图，对比零常驻 baseline 和部分 block 常驻策略。baseline 按需从 CPU 搬运权重；常驻策略将估测出的 block 数量均匀分布在模型各层，减少重复搬运。

准备完整的 `Qwen-Image-2512` BF16 模型，在脚本顶部设置 `lightx2v_path`、`model_path`。三个脚本的正负提示词保持一致，两份推理脚本使用相同 seed；默认 16:9、50 步。

下面命令在仓库根目录执行：先估测并生成常驻配置，再运行对比。baseline 也可以单独运行。

## NVIDIA GPU

在三个 CUDA 脚本中设置 `CUDA_VISIBLE_DEVICES`。

```bash
bash scripts/qwen_image/inference/offload_residency/estimate.sh
bash scripts/qwen_image/inference/offload_residency/run_baseline.sh
bash scripts/qwen_image/inference/offload_residency/run.sh
```

## Ascend NPU

在三个 NPU 脚本中设置 `ASCEND_RT_VISIBLE_DEVICES`。

```bash
bash scripts/qwen_image/inference/offload_residency/estimate_npu.sh
bash scripts/qwen_image/inference/offload_residency/run_baseline_npu.sh
bash scripts/qwen_image/inference/offload_residency/run_npu.sh
```

基础配置位于 `configs/qwen_image/residency/`，CUDA 使用 `baseline.json`，NPU 使用 `baseline_npu.json`。结果保存在 `save_results/qwen_residency/`：`resident.json` 为常驻配置，`resident.profile.json` 为测量报告，`baseline.png` 和 `resident.png` 为推理结果；NPU 文件带 `_npu` 后缀。重复运行会覆盖同名文件。

生成配置中的 `offload_resident_blocks` 是常驻数量，`0` 表示零常驻。修改设备、输入或基础配置后重新估测。profiling 统一沿用 [base.sh](../../../base/base.sh)。

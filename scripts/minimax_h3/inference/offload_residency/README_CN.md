# MiniMax-H3 常驻权重

[English](README.md) | 简体中文

单卡运行 MiniMax-H3 文生音视频，对比零常驻 baseline 和部分 block 常驻策略。baseline 按需从 CPU 搬运权重；常驻策略将估测出的 block 数量均匀分布在模型各层，减少重复搬运。

按[模型说明](../../README_zh.md)准备完整 BF16 Diffusers 组件，设置脚本中的 `lightx2v_path`、`model_path`，保持提示词和推理 seed 一致。先生成匹配的 [AdaLN 缓存](../../README_zh.md#adaln-缓存)；权重和采样设置相同时，估测、baseline 与常驻推理共用缓存。

下面命令在仓库根目录执行：先估测并生成常驻配置，再运行对比。baseline 也可以单独运行。

## NVIDIA GPU

在三个 CUDA 脚本中设置 `CUDA_VISIBLE_DEVICES`。缓存生成脚本选择 `PLATFORM=cuda`、`--model-variant fl2av` 和 `configs/minimax_h3/residency/baseline.json`。

```bash
bash scripts/minimax_h3/inference/offload_residency/estimate.sh
bash scripts/minimax_h3/inference/offload_residency/run_baseline.sh
bash scripts/minimax_h3/inference/offload_residency/run.sh
```

## Ascend NPU

在三个 NPU 脚本中设置 `ASCEND_RT_VISIBLE_DEVICES`。缓存生成脚本选择 `PLATFORM=ascend_npu`、`--model-variant fl2av` 和 `configs/minimax_h3/residency/baseline_npu.json`。

```bash
bash scripts/minimax_h3/inference/offload_residency/estimate_npu.sh
bash scripts/minimax_h3/inference/offload_residency/run_baseline_npu.sh
bash scripts/minimax_h3/inference/offload_residency/run_npu.sh
```

基础配置位于 `configs/minimax_h3/residency/`，CUDA 使用 `baseline.json`，NPU 使用 `baseline_npu.json`。结果保存在 `save_results/minimax_h3_residency/`：`resident.json` 为常驻配置，`resident.profile.json` 为测量报告，`baseline.mp4` 和 `resident.mp4` 为推理结果；NPU 文件带 `_npu` 后缀。重复运行会覆盖同名文件。

生成配置中的 `offload_resident_blocks` 是常驻数量，`0` 表示零常驻。修改设备、输入或基础配置后重新估测。profiling 统一沿用 [base.sh](../../../base/base.sh)。

更改权重、步数或采样偏移时，还需重新生成匹配的 AdaLN 缓存。

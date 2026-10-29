# MiniMax-H3 CPU Offload

[English](README.md) | 简体中文

单卡运行 MiniMax-H3 文生音视频，对比两种 DiT CPU block offload：`baseline` 逐 tensor 搬运权重，`contiguous` 使用连续 pinned 内存并整 block 搬运。

按[模型说明](../../README_zh.md)准备完整 BF16 Diffusers 组件，在两份脚本中设置 `lightx2v_path`、`model_path`。先生成匹配的 [AdaLN 缓存](../../README_zh.md#adaln-缓存)；权重和采样设置相同时，baseline 与 contiguous 共用缓存。

## NVIDIA GPU

在两份脚本中设置 `CUDA_VISIBLE_DEVICES`。缓存生成脚本选择 `PLATFORM=cuda`、`--model-variant fl2av` 和 `configs/minimax_h3/layout/baseline.json`。

```bash
bash scripts/minimax_h3/inference/offload_layout/run_baseline.sh
bash scripts/minimax_h3/inference/offload_layout/run_contiguous.sh
```

## Ascend NPU

在两份 NPU 脚本中设置 `ASCEND_RT_VISIBLE_DEVICES`。缓存生成脚本选择 `PLATFORM=ascend_npu`、`--model-variant fl2av` 和 `configs/minimax_h3/layout/baseline_npu.json`。

```bash
bash scripts/minimax_h3/inference/offload_layout/run_baseline_npu.sh
bash scripts/minimax_h3/inference/offload_layout/run_contiguous_npu.sh
```

配置位于 `configs/minimax_h3/layout/`，提示词在脚本的 `--prompt` 中设置。结果保存在 `save_results/minimax_h3_layout/`，重复运行会覆盖同名结果。

`cpu_offload_block_contiguous` 在 baseline 配置中为 `false`，在 contiguous 配置中为 `true`。

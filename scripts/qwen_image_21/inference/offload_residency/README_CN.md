# Qwen-Image-2.1-Turbo 常驻权重

[English](README.md) | 简体中文

单卡运行 Qwen-Image-2.1-Turbo，对比零常驻 baseline 和部分 block 常驻策略。baseline 按需从 CPU 搬运权重；常驻策略将估测出的 block 数量均匀分布在模型各层，减少重复搬运。

准备完整 BF16 模型，包含 `transformer`、`text_encoder`、`processor` 和 `vae`。设置脚本中的 `lightx2v_path`、`model_path`，保持提示词和推理 seed 一致。默认文生图，2048 × 2048、8 步、关闭 CFG。

下面命令在仓库根目录执行：先估测并生成常驻配置，再运行对比。baseline 也可以单独运行。

## NVIDIA GPU

在三个 CUDA 脚本中设置 `CUDA_VISIBLE_DEVICES`。

```bash
bash scripts/qwen_image_21/inference/offload_residency/estimate.sh
bash scripts/qwen_image_21/inference/offload_residency/run_baseline.sh
bash scripts/qwen_image_21/inference/offload_residency/run.sh
```

## Ascend NPU

在三个 NPU 脚本中设置 `ASCEND_RT_VISIBLE_DEVICES`。

```bash
bash scripts/qwen_image_21/inference/offload_residency/estimate_npu.sh
bash scripts/qwen_image_21/inference/offload_residency/run_baseline_npu.sh
bash scripts/qwen_image_21/inference/offload_residency/run_npu.sh
```

## 图生图

在上述命令末尾追加 `--task i2i --prompt '编辑指令'`：估测命令还需追加 `--image-path /path/to/input.png`，两条推理命令追加 `--image_path /path/to/input.png`。多张参考图用逗号分隔，三条命令使用相同输入。

切换到图生图后先重新估测，再运行两份推理脚本；默认会覆盖同名配置和图片。

基础配置位于 `configs/qwen_image_21/residency/`，CUDA 使用 `baseline.json`，NPU 使用 `baseline_npu.json`。结果保存在 `save_results/qwen_image_21_residency/`：`resident.json` 为常驻配置，`resident.profile.json` 为测量报告，`baseline.png` 和 `resident.png` 为推理结果；NPU 文件带 `_npu` 后缀。重复运行会覆盖同名文件。

生成配置中的 `offload_resident_blocks` 是常驻数量，`0` 表示零常驻。修改设备、输入或基础配置后重新估测。profiling 统一沿用 [base.sh](../../../base/base.sh)。

`cpu_offload_block_contiguous` 控制连续布局，默认 `false`；设为 `true` 后重新估测，baseline 与常驻使用相同值。

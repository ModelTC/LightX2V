# MiniMax-H3 CPU Offload

[English](README.md) | 简体中文

单卡运行 MiniMax-H3 文生音视频，对比两种 DiT CPU block offload：`baseline` 逐 tensor 搬运权重，`contiguous` 使用连续 pinned 内存并整 block 搬运。

安装对应平台的项目依赖，准备完整 BF16 Diffusers 权重（包含文本编码器、视频和音频 VAE，见[模型说明](../README_zh.md)）。在仓库根目录设置权重路径：

```bash
export MODEL_PATH="$PWD/models/MiniMax-H3"
```

首次运行前，需要生成与权重及配置匹配的 AdaLN 缓存。已有匹配缓存时跳过生成命令；baseline 和 contiguous 共用缓存。

## NVIDIA GPU

```bash
# 首次运行：生成缓存
PLATFORM=cuda CUDA_VISIBLE_DEVICES=0 DTYPE=BF16 SENSITIVE_LAYER_DTYPE=None \
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
  --model_path "$MODEL_PATH" \
  --config_json configs/minimax_h3/layout/baseline.json \
  --model-variant fl2av

# 生成音视频
bash scripts/minimax_h3/offload_layout/run_baseline.sh
bash scripts/minimax_h3/offload_layout/run_contiguous.sh
```

## Ascend NPU

```bash
# 首次运行：生成缓存
PLATFORM=ascend_npu ASCEND_RT_VISIBLE_DEVICES=0 DTYPE=BF16 SENSITIVE_LAYER_DTYPE=None \
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
  --model_path "$MODEL_PATH" \
  --config_json configs/minimax_h3/layout/baseline_npu.json \
  --model-variant fl2av

# 生成音视频
bash scripts/minimax_h3/offload_layout/run_baseline_npu.sh
bash scripts/minimax_h3/offload_layout/run_contiguous_npu.sh
```

默认使用第 0 张卡。配置位于 `configs/minimax_h3/layout/`，提示词在脚本的 `--prompt` 中设置。结果保存在 `save_results/minimax_h3_layout/`，重复运行会覆盖同名结果。

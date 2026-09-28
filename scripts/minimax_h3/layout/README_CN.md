# MiniMax-H3 CPU Block Offload 连续权重布局

[English](README.md) | 简体中文

这些入口对比 MiniMax-H3 DiT 的普通逐 tensor CPU block offload 与持久连续 pinned 权重布局。同一平台的两份配置仅相差 `cpu_offload_layout`。连续方案在加载时填充最终 CPU block 存储，推理时将整块字节 buffer 复制到两个设备 slot 之一，不在每一步重新 pack。算子的辅助设备状态单独复制。

## 接入范围

- 单张 NVIDIA GPU 或单张 Ascend NPU，原始 BF16 checkpoint、eager 推理，不使用 CFG。
- 入口使用 `--model-variant fl2av --task t2av`，29 步、124 帧、768 × 1344、seed 42。
- 连续布局仅作用于 DiT transformer blocks。Qwen3-VL 保留原有逐层 offload，视频和音频 VAE 保留 model offload。Pre/post 权重沿用原有加载精度，保留媒体投影所需的 FP32。
- 必须预先生成匹配的 AdaLN 缓存。被缓存替代的 AdaLN 投影权重在 baseline 和 contiguous 中都不进入 block 存储。
- H3 连续模式暂不支持 fused QKV 投影、量化 DiT checkpoint 和 `dit_release_block_offload_buffers`。公共层还会拒绝共享 CPU 权重、磁盘 lazy load、多 rank 并行、LoRA/adapter、特征缓存和 compile/graph 等组合。

## 权重与缓存

`MODEL_PATH` 需要包含原始 Diffusers 组件：`transformer/`、`text_encoder/`、`tokenizer/`、`processor/`、`vae/` 和 `audio_vae/`。仅有原始格式的 `FL2VA/` 子目录不足以运行，详见[模型说明](../README_zh.md)。

在仓库根目录执行，先指定权重：

```bash
export MODEL_PATH=/workspace/code/models/MiniMax-H3
```

在所选平台上先生成 AdaLN 缓存。生成工具拒绝覆盖已有目录；如果已经有对应权重和配置的匹配缓存，可以直接复用。同一平台的两个变体共用缓存。复用时保持权重、模型 variant、`infer_steps` 和 flow shift 一致。配置中的缓存根目录是 `~/.cache/lightx2v/adaln`。

NVIDIA：

```bash
PLATFORM=cuda CUDA_VISIBLE_DEVICES=0 DTYPE=BF16 SENSITIVE_LAYER_DTYPE=None \
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
  --model_path "$MODEL_PATH" \
  --config_json configs/minimax_h3/layout/baseline.json \
  --model-variant fl2av
```

Ascend：

```bash
PLATFORM=ascend_npu ASCEND_RT_VISIBLE_DEVICES=0 DTYPE=BF16 SENSITIVE_LAYER_DTYPE=None \
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
  --model_path "$MODEL_PATH" \
  --config_json configs/minimax_h3/layout/baseline_npu.json \
  --model-variant fl2av
```

## 运行入口

| 平台 | Baseline | Contiguous |
|---|---|---|
| NVIDIA | `bash scripts/minimax_h3/layout/run_baseline.sh` | `bash scripts/minimax_h3/layout/run_contiguous.sh` |
| Ascend | `bash scripts/minimax_h3/layout/run_baseline_npu.sh` | `bash scripts/minimax_h3/layout/run_contiguous_npu.sh` |

脚本自动定位仓库，支持 `MODEL_PATH`。设备默认选择第 0 张卡，也可通过对应的可见设备环境变量指定。结果保存在仓库下的 `save_results/minimax_h3_layout/{baseline,contiguous,baseline_npu,contiguous_npu}.mp4`。额外 CLI 参数会转交给推理入口，例如 `--save_result_path /workspace/code/results/h3.mp4`；请先创建输出父目录。

CUDA 保留 SageAttention2、SGL RMSNorm 和 H3 Triton RoPE。Ascend 使用 `npu_flash_attn`、`npu_rms_norm`、`minimax_h3_npu_rope`；后者在可用时调用 MindIE-SD，否则使用已有的 torch real-RoPE 实现。NPU 两个变体均保持 `qwen3vl_attn_type=torch_sdpa`，以保留文本编码器的 causal/GQA 语义，VAE 使用 `vae_attn_type=torch_sdpa`。DiT attention 配置不控制文本编码器；当前 NPU flash-attention 包装尚不能直接替代需要因果掩码的文本 attention。

## 代码与验证

`MiniMaxH3TransformerWeights` 注册 `transformer_blocks.{i}.` 及原有两个 slot；模型 checkpoint reader 记录原始元数据并保留混合精度。`BaseTransformerModel`、`BlockLoadPlan`、`BlockBuffer`、`ContiguousBlockTransfer` 完成公共加载和传输。`MiniMaxH3OffloadTransformerInfer` 保持原有 block 循环与同步。平台内存处理留在 `lightx2v_platform`，NVIDIA 设备类保持不变。

在实际设备环境执行存储和调度检查：

```bash
PLATFORM=cuda CUDA_VISIBLE_DEVICES=0 DTYPE=BF16 SENSITIVE_LAYER_DTYPE=None PROFILING_DEBUG_LEVEL=0 \
python -m pytest -q test_cases/test_block_buffer.py test_cases/test_block_offload_groups.py

PLATFORM=ascend_npu ASCEND_RT_VISIBLE_DEVICES=0 DTYPE=BF16 SENSITIVE_LAYER_DTYPE=None PROFILING_DEBUG_LEVEL=0 \
python -m pytest -q test_cases/test_block_buffer.py test_cases/test_block_offload_groups.py
```

测试覆盖 H3 元数据与过滤、CPU 存储归属、slot 地址稳定、多轮复制、跨 step/request 的原生 block 计算和不支持组合。小尺寸合成 block 测试不等于完整生成验证；还需在各平台固定输入与配置，对比 baseline/contiguous 的视频和音频 latents，并检查解码结果。CUDA 验证不能替代 910B 实机验证，NPU 入口需要在目标机器确认。

这里的连续指每个 block 内虚拟地址连续。它减少独立分配和 H2D 提交次数，不减少主体权重字节量或 attention 计算。性能收益需要在正确性通过后单独测量。

# Qwen Image CPU block offload 内存布局

[English](README.md) | 简体中文

单张 NVIDIA GPU 或 Ascend NPU 上比较 Qwen-Image-2512 BF16 的两种 DiT 权重布局：baseline 为各 tensor 独立分配 pinned 内存并逐 tensor H2D；连续布局在加载时为每个 block 分配连续 pinned 存储，推理时整 block H2D。同一平台的两个入口使用相同的计算算子和双缓冲调度。

## 运行

NVIDIA GPU：在仓库根目录执行：

```bash
bash scripts/qwen_image/layout/run_baseline.sh
bash scripts/qwen_image/layout/run_contiguous.sh
```

脚本顶部设置项目路径、模型路径和 GPU 编号。模型目录需要包含原始 BF16 的 `transformer/`、`text_encoder/`、`tokenizer/`、`vae/` 等文件；不能使用 FP8 checkpoint 替代 BF16 权重。环境需安装项目 CUDA 依赖，包括 FlashAttention 3、FlashInfer 和 Triton，并使用支持这些算子的 GPU。

默认 T2I、50 步、16:9、CFG 4.0、seed 42，沿用 Qwen 原有 prompt 和计算算子。脚本显式设置 BF16、敏感层跟随主 dtype，并在加载 `base.sh` 后关闭调试计时，避免计时同步影响 offload 调度。

配置为 `configs/qwen_image/layout/baseline.json` 和 `continuous.json`，仅后者增加 `"cpu_offload_layout": "contiguous"`。Qwen2.5-VL 文本编码器和 VAE 均开启原有整体 CPU offload；连续 block 布局只作用于 DiT。

输出分别为 `save_results/qwen_layout/baseline.png` 和 `continuous.png`，重复运行会覆盖对应图片。需要调整输入、步数或算子时，两套配置和命令保持一致。

### Ascend 910B

在可正常使用 `torch_npu` 的环境中执行：

```bash
MODEL_PATH=/workspace/code/models/Qwen-Image-2512 \
  bash /workspace/code/scripts/qwen_image/layout/run_baseline_npu.sh
MODEL_PATH=/workspace/code/models/Qwen-Image-2512 \
  bash /workspace/code/scripts/qwen_image/layout/run_contiguous_npu.sh
```

NPU 脚本自动定位仓库，设置 `PLATFORM=ascend_npu`，默认使用第 0 张设备；可通过 `ASCEND_RT_VISIBLE_DEVICES` 指定其他单卡。`MODEL_PATH` 默认是仓库下的 `models/Qwen-Image-2512`，需要同样的原始 BF16 权重及文本编码器、VAE 文件。运行环境需安装与驱动、CANN 匹配的 PyTorch、torch_npu，以及项目的 Transformers、Diffusers 等依赖。

配置为 `baseline_npu.json`、`continuous_npu.json`，仅连续版本增加 `"cpu_offload_layout": "contiguous"`。两者显式选择 `npu_flash_attn`，沿用项目 Qwen NPU 配置中的 `torch_real_rope`、PyTorch RMSNorm、LayerNorm 和调制路径；这些 PyTorch 运算在 NPU 张量上执行。BF16、50 步、16:9、CFG 4.0、seed 42 及组件 offload 设置与 GPU 入口一致。

输出为仓库下的 `save_results/qwen_layout/baseline_npu.png`、`continuous_npu.png`，不受启动时工作目录影响。当前开发机没有 NPU，入口及本地回归检查不能替代 910B 上的完整生成与结果一致性验证。

## 接入方式

`QwenImageTransformerWeights` 向公用层注册 `transformer_blocks.{i}.` 对应的 block 和两个已有设备 slot。每个 block 内的图像注意力、文本注意力、联合注意力和 FFN 四个 phase 统一规划布局。

```text
register_offload_group
  → BaseTransformerModel._apply_weights
  → prepare_contiguous_groups → 算子存储描述 → BlockLoadPlan.load
  → CPU pinned block 与设备 slot 的 tensor 视图

QwenImageOffloadTransformerInfer
  → init_first_buffer → prefetch_weights
  → ContiguousBlockTransfer.copy → run_block → swap_blocks
```

复用 `lightx2v/common/offload/` 和 `lightx2v_platform/`，没有 Qwen 专用布局加载器。CPU 权重在推理期间只读，不做每步 pack；block 之间不要求地址连续。RMSNorm 的辅助设备标量仍单独复制。

本入口限定单卡 BF16、block offload、eager、NoCaching，不组合共享权重、lazy load、并行、LoRA 或编译。现有 `fp8-sgl` 蒸馏配置不在此次支持范围内。连续布局减少分配和传输提交次数，不减少主体权重字节量，也不保证相同比例的推理加速。

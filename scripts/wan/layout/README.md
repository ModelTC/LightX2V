# Wan CPU block offload layout

baseline 使用逐 tensor CPU 权重和 H2D 复制；solution2 在加载时将权重写入连续 pinned block，推理时整 block H2D。两者使用相同的计算和双缓冲调度。

## CUDA 推理

在仓库根目录运行，每个脚本生成一个视频：

```bash
bash scripts/wan/layout/run_baseline.sh
bash scripts/wan/layout/run_contiguous.sh
```

模型路径和 GPU 编号在脚本开头设置。配置分别为 `configs/wan/layout/baseline.json`、`solution2.json`，后者只增加 `"cpu_offload_layout": "contiguous"`。

默认使用 Wan2.1 I2V 14B LightX2V FP8 DiT/T5/CLIP、BF16 计算和 FlashAttention 3，单卡、seed 42、40 步、81 帧、CFG。`size: [480, 832]` 是面积预算，实际宽高随输入图片确定。输出为 `save_results/wan_layout/baseline.mp4`、`solution2.mp4`，再次运行会覆盖对应视频。

## Ascend 910B 推理

在驱动、CANN、PyTorch、torch_npu 和项目依赖已配置好的机器上，从仓库根目录运行：

```bash
MODEL_PATH=/workspace/code/models/Wan2.1-I2V-14B-720P \
bash scripts/wan/layout/run_baseline_npu.sh

MODEL_PATH=/workspace/code/models/Wan2.1-I2V-14B-720P \
bash scripts/wan/layout/run_contiguous_npu.sh
```

默认选择一张卡（`ASCEND_RT_VISIBLE_DEVICES=0`）、BF16、原始 Wan2.1 I2V 权重，使用 `baseline_npu.json`、`solution2_npu.json`。模型目录需要包含 DiT、T5、CLIP、VAE 及 tokenizer；FP8 checkpoint 不能作为 BF16 原始权重使用。T5、CLIP、VAE 也开启 CPU offload，两个方案配置相同。

两个脚本直接调用 `python -m lightx2v.infer`，各执行一次完整推理，输出为 `baseline_npu.mp4`、`solution2_npu.mp4`。可用 `ASCEND_RT_VISIBLE_DEVICES` 选择卡，`CONFIG_PATH` 指定配置；例如使用现有的 `baseline_npu_int8.json`、`solution2_npu_int8.json` 时，先将配置里的 `dit_quantized_ckpt` 改为匹配 INT8-NPU 的权重路径。

Wan NPU block offload 在模型初始化、权重创建前关闭 private format，使用 ND 存储。普通浮点和 INT8 checkpoint 在 dtype 转换前检查矩阵精度。NPU UniPC 小型系数系统在 CPU 求解，保留原来的采样时间表。

所有 layout 脚本在加载 `base.sh` 后设置 `PROFILING_DEBUG_LEVEL=0`。项目原生 profiler 保留，但这些入口不启用调试计时。

## 连续布局

每个 CPU block 按最终 dtype、shape 和转置方式规划布局，分配一块 pinned `uint8` 存储。各 tensor 使用其中的视图，起始偏移按 256 字节对齐；checkpoint 直接写入最终视图，包含需要的 dtype 转换。这里的连续指虚拟地址连续，不要求物理页连续。

两套设备 block buffer 使用相同布局。预取时一次 `storage.copy_` 将整个 CPU block 传入备用设备 buffer。推理期间 CPU 权重只读，没有 pack、staging buffer 或重新分配。CUDA RMSNorm 的辅助占位标量保留独立 D2D 复制。

```text
WanTransformerAttentionBlock.load
  → load_contiguous_block
    → BlockLayout / BlockBuffer
    → WeightModule.load → BlockLoadContext.take / bind
    → 校验最终视图和 pinned 状态

WanModel._init_offload_manager
  → init_contiguous_blocks
  → init_first_buffer / prefetch_weights
    → ContiguousBlockTransfer.copy
  → run_block
  → swap_blocks：同步加载和计算，交换设备 buffer
```

通用存储和传输逻辑在 `lightx2v/common/offload/block_layout.py`，Wan 加载约束在 `lightx2v/models/networks/wan/weights/block_layout.py`。solution2 支持单卡 Wan2.1 I2V、普通 CPU block offload 和 NoCaching；CUDA 支持原生浮点和 FP8-vLLM，Ascend 支持原生浮点和 INT8-NPU。共享权重、lazy load、并行、LoRA、CUDA Graph 等组合不在当前支持范围。

CUDA 的 Wan 加载路径不导入 Ascend 算子。NPU 算子只在选择 NPU 时接入，连续视图绑定和 dtype 规则位于已有的 `lightx2v_platform/ops/mm/ascend_npu/`、`ops/norm/ascend_npu/` 具体算子中；平台公共模板保持普通加载行为。Wan 的 NPU 初始化、T5 默认算子选择和 scheduler 系数设备选择仍在现有模型文件的明确平台分支中。

baseline 的 tensor 独立分配，不保证同一 block 内的地址连续。CUDA FP8 路径中部分 scale 的后续 dtype 转换可能使其不再 pinned；实际 pinned 状态由加载结果决定。

历史实验结果保存在 `save_results/wan_layout/`，包括 `benchmark_20260923_031506_688642/` 的 910B 数据。测速和诊断工具已移除，已有日志、CSV、JSON 和 latent 文件保留。

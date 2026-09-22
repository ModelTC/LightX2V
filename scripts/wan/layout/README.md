# Wan CPU block offload layout

在仓库根目录运行：

```bash
bash scripts/wan/layout/run_baseline.sh
```

脚本使用 `configs/wan/layout/baseline.json`，单卡、单进程运行 Wan2.1 I2V 14B。模型路径和 GPU 编号在脚本开头设置；配置中的相对路径从仓库根目录解析。

固定条件：FP8 DiT/T5/CLIP、BF16 计算、FlashAttention 3、seed 42、40 步、81 帧、CFG。`size: [480, 832]` 是面积预算，当前输入图片对应输出宽 544、高 720。仅 DiT 启用普通 CPU block offload，T5、CLIP 和 VAE 的 CPU offload 关闭。

CPU 权重沿用原生加载流程，各 tensor 独立分配，不保证同一 block 内的地址连续。预取按 block 调度，通过 `state_dict()` 和 `load_state_dict()` 逐 tensor H2D；GPU 使用原有两套 block buffer，配合加载流、计算流和双缓冲交换。

原始 tensor 创建时使用 pinned memory；部分 FP32 scale 在后续 dtype 转换后不再 pinned，因此不能假设所有 CPU tensor 都处于 pinned 状态。

输出为 `save_results/wan_layout/baseline.mp4`，重复运行会覆盖该视频。脚本沿用 `scripts/base/base.sh` 的 `PROFILING_DEBUG_LEVEL=2`；比较推理性能时应统一 profiling 设置，关闭调试同步的测量需在 `source base.sh` 之后设为 `0`。

## 方案2：加载时直接写入连续 pinned block

```bash
bash scripts/wan/layout/run_solution2.sh
```

使用 `configs/wan/layout/solution2.json`，相对 baseline 只增加 `"cpu_offload_layout": "contiguous"`，输出为 `save_results/wan_layout/solution2.mp4`。当前支持单卡 Wan2.1 I2V、FP8-vllm、普通 CPU block offload 和 NoCaching；共享权重、lazy load、并行、LoRA、CUDA Graph 等组合会明确报错。

每个 CPU block 在加载前根据最终 dtype、shape、转置方式规划布局，分配一块 pinned `uint8` 存储，tensor 使用其中的视图，起始偏移按 256 字节对齐。checkpoint 权重直接复制到最终视图，包含 FP32 scale 的 dtype 转换；不先创建独立 pinned tensor 再重新打包。加载完成后不保留 checkpoint 字典，也没有运行时 staging buffer。这里的连续指 CPU 虚拟地址连续，不要求物理页连续。

GPU 两套 block buffer 各自使用一块连续存储，采用相同布局。加载流每次将整个 CPU block 一次 H2D 到备用 GPU buffer，沿用原来的加载流、计算流同步及双缓冲交换。CPU block 在推理阶段只读，不重新整合、不重新分配。原有 RMSNorm CUDA `weight_diff` 占位标量保留小量 D2D 复制，不计入 H2D。

调用链：

```text
WanTransformerAttentionBlock.load
  → load_contiguous_block
    → 收集算子绑定，规划 BlockLayout，创建 BlockBuffer
    → WeightModule.load
      → create_default_tensors / create_cuda_buffers / DefaultTensor.load
        → BlockLoadContext.take：CPU 写入最终视图；GPU 绑定视图
    → 校验 dtype、shape、stride、地址及 pinned 状态，保存 buffer

WanModel._init_offload_manager
  → WeightAsyncStreamManager.init_contiguous_blocks
  → init_first_buffer / prefetch_weights
    → ContiguousBlockTransfer.copy：一次 storage.copy_ H2D
  → swap_blocks：等待加载和计算完成，交换两套 GPU buffer
```

通用存储及传输逻辑位于 `lightx2v/common/offload/block_layout.py`；Wan 的算子、dtype 和配置约束位于 `lightx2v/models/networks/wan/weights/block_layout.py`。配置在 `WanModel._init_infer_class` 校验一次，block 加载复用同一份算子绑定信息完成布局和加载后检查。省略配置开关时沿用 baseline。

## 验证

在依赖齐全的单卡 CUDA 环境运行：

```bash
CUDA_VISIBLE_DEVICES=0 DTYPE=BF16 SENSITIVE_LAYER_DTYPE=None PROFILING_DEBUG_LEVEL=0 \
  python -m pytest -q test_cases/test_block_layout.py
```

测试覆盖直接加载、逐字节一致性、转置和 FP32 scale 视图、GPU 双缓冲切换、跨 step 回绕、CPU 存储只读、初始化流同步，以及不兼容配置和意外重新分配的拒绝行为。

真实权重传输测试及完整 40 步推理对照的日志、测量脚本和 JSON 结果保存在 `save_results/wan_layout/solution2_20260921_063733/`。性能测试统一关闭 profiling 调试同步，DMA 活动耗时与包含提交/等待的 block 加载墙钟耗时分别统计；独立 H2D 微基准不能直接代表端到端推理收益。

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

使用 `configs/wan/layout/solution2.json`，相对 baseline 只增加 `"cpu_offload_layout": "contiguous"`，输出为 `save_results/wan_layout/solution2.mp4`。支持单卡 Wan2.1 I2V、普通 CPU block offload 和 NoCaching；CUDA 支持原生浮点和 FP8-vLLM，Ascend 支持原生浮点和 INT8-NPU。共享权重、lazy load、并行、LoRA、CUDA Graph 等组合会明确报错。

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

## Ascend 910B 独立计时

不需要 Codex。在有匹配的驱动、CANN、PyTorch、torch_npu 和 LightX2V 依赖的机器上，从仓库根目录运行以下命令。项目提供的容器参考为 `dockerfiles/platforms/Dockerfile_ascend_910b`。本次开发机器是 H200，910B 的设备测试和完整推理需要在目标机器执行。

### 先验证设备及加载路径

```bash
PLATFORM=ascend_npu ASCEND_RT_VISIBLE_DEVICES=0 \
DTYPE=BF16 SENSITIVE_LAYER_DTYPE=None PROFILING_DEBUG_LEVEL=0 \
python -m pytest -q test_cases/test_block_layout_device.py
```

测试不需要模型文件，覆盖 BF16/INT8 的实际平台加载器、混合 dtype 连续视图、CPU pinned 状态、设备双缓冲与跨 step 回绕、CPU 权重只读、计时上限及错误精度拒绝。在 NPU 上还会执行真实 INT8 MM，并与逐 tensor 加载的结果比较。CUDA 上也可运行这些测试，但不会执行 NPU INT8 计算内核。

benchmark 启动时会验证 pinned CPU → 设备的异步复制、混合 dtype 视图共享存储及逐值一致性。两种方案都关闭 NPU private format，使用 ND 存储；不支持的 torch_npu 版本会在预检中报错，不会退化为普通 CPU 内存后继续测速。

### 使用现有 FP8 权重测传输

```bash
CHECKPOINT_PATH=/path/to/Wan2.1-I2V-14B-720P-Lightx2v/fp8 \
bash scripts/wan/layout/run_benchmark_npu.sh copy
```

`CHECKPOINT_PATH` 可以是单个 block 的 safetensors 文件，或含全部 block 的目录。默认按全部 block 轮转，CPU 权重在计时前部署完毕。baseline 独立分配各 tensor；solution2 直接写入最终连续 pinned block，并使用公用 `ContiguousBlockTransfer` 复制到两套连续设备 buffer。推理期间没有 pack。

该模式按字节复制，不执行 FP8 矩阵计算，也不在 NPU 上创建 FP8 typed view。FP8 scale 按 FP32 部署，baseline 保留当前加载流程中 scale 转换导致的非 pinned 状态。它是 **byte layout microbenchmark**：不复现实际算子的 typed copy、转置处理或计算竞争，不能当作完整 baseline 推理成绩。`weights.json` 和 `transfers.csv` 记录实际字节数、原始 block 编号、dtype 与 pinned 数量。

### BF16 完整推理对照

```bash
MODEL_PATH=/path/to/Wan2.1-I2V-14B-720P \
bash scripts/wan/layout/run_benchmark_npu.sh infer
```

需要完整原始模型目录：DiT safetensors、`config.json`、`models_t5_umt5-xxl-enc-bf16.pth`、`models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`、`Wan2.1_VAE.pth` 及对应 tokenizer。只有当前 FP8 子目录不能运行该模式，不会自动反量化或重新量化。

使用 `configs/wan/layout/baseline_npu.json` 和 `solution2_npu.json`。两份配置只差 `cpu_offload_layout`；单卡、40 步、81 帧、相同输入及 seed，使用 NPU attention/norm/RoPE。T5、CLIP 和 VAE 也启用 offload 以减少单卡驻留内存；这部分设置对两个方案相同，block 计时仅记录 DiT。T5 的 NPU offload 归一化使用已有 torch 实现，避免调用 CUDA 的 SGL kernel。

### INT8 完整推理对照

先将两份 INT8 配置中的 `dit_quantized_ckpt` 都改成同一个 INT8 DiT checkpoint 路径：

```bash
MODEL_PATH=/path/to/Wan2.1-I2V-14B-720P \
bash scripts/wan/layout/run_benchmark_npu.sh infer \
  --baseline-config configs/wan/layout/baseline_npu_int8.json \
  --solution2-config configs/wan/layout/solution2_npu_int8.json
```

DiT 要求匹配 `int8-npu` 的逐输出通道对称 INT8 权重和 FP32 scale；T5、CLIP 使用原始权重。加载器和 benchmark 会拒绝把 FP8 权重当 INT8 使用。

### 计时口径与结果

默认每个方案独立进程，预热 1 次、正式测量 3 次。可在命令后追加 `--warmup 2 --repeats 5`。推理和输入参数可通过 `--help` 查看，benchmark 参数不进入模型 JSON。

结果保存在 `save_results/wan_layout/benchmark_<UTC时间>/`：

- `summary.json` / `summary.csv`：两种方案的中位数、最小值、最大值和速度比。
- `<variant>/result.json`：初始化时间、原始计时样本、设备/PyTorch/torch_npu 信息。
- `<variant>/transfers.csv`：每个采样 block 的 CPU 提交时间、加载流 Event 时间、实际字节数和有效 GB/s；带宽使用十进制 GB。
- `<variant>/latents.pt`：完整推理的最终 latent，保存与比较均在计时之外；汇总包含精确相等判断和最大误差，超过 `rtol=atol=0.01` 时进程失败。
- `environment.json`、`worktree.txt`、两份日志，以及 NPU 环境下的 `npu-smi.txt` 和可获取的 CANN 版本。

`request_s` 包含输入编码、完整 denoise 和 VAE 解码，不写视频；`denoise_s` 单独统计完整采样循环。初始化、预热与正式测量分开。原有调试 profiler 被关闭，只在请求/denoise 边界同步；正式轮不插入逐 block Event。额外诊断请求最多记录 `--samples` 个 block（默认 160），在请求完成后读取 Event，避免每次预取新增同步。

**`load_stream_ms` 不是纯 DMA 时间**：可能包含 CPU 提交间隙、辅助 D2D 和设备竞争。`h2d_copy_calls` 是 Python 复制调用数，不保证等于底层 DMA 任务数。其合计仅对应采样窗口，不能当作整次推理的 H2D 总时间，也不能与计算时间相加。

追加 `--profile` 会在正式测速外导出一次独立传输遍历的 Ascend profiler 数据到 `<variant>/profiler/`。纯 H2D DMA 需要按 profiler 的 Host→Device 任务分析；脚本将 `dma_ms` 留为 `null`，不把 Event 时间冒充 DMA 时间。该 profile 不包含完整推理的计算重叠。

通用存储/传输在 `common/offload/block_layout.py`，平台加载器通过 `BlockLoadContext.bind()` 绑定最终视图，Wan 布局契约在 `models/networks/wan/weights/block_layout.py`。可选 `TransferTimer` 接在 manager 的首次加载和预取入口；未启用时不创建 Event。benchmark 复用 `build_runner()` / `run_request()`，仅在自己的 runner 实例上包装 `run_segment()` 记录 denoise。

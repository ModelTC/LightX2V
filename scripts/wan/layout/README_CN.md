# Wan CPU block offload 内存布局

[English](README.md) | 简体中文

baseline 使用逐 tensor CPU 权重和 H2D 复制；连续布局方案在加载时将权重写入连续 pinned block，推理时整 block H2D。两者使用相同的计算和双缓冲调度。

## CUDA 推理

在仓库根目录运行，每个脚本生成一个视频：

```bash
bash scripts/wan/layout/run_baseline.sh
bash scripts/wan/layout/run_contiguous.sh
```

模型路径和 GPU 编号在脚本开头设置。配置分别为 `configs/wan/layout/baseline.json`、`continuous.json`，后者只增加 `"cpu_offload_layout": "contiguous"`。

默认使用 Wan2.1 I2V 14B LightX2V FP8 DiT/T5/CLIP、BF16 计算和 FlashAttention 3，单卡、seed 42、40 步、81 帧、CFG。`size: [480, 832]` 是面积预算，实际宽高随输入图片确定。输出为 `save_results/wan_layout/baseline.mp4`、`solution2.mp4`，再次运行会覆盖对应视频。

## Ascend 910B 推理

在驱动、CANN、PyTorch、torch_npu 和项目依赖已配置好的机器上，从仓库根目录运行：

```bash
MODEL_PATH=/workspace/code/models/Wan2.1-I2V-14B-720P \
bash scripts/wan/layout/run_baseline_npu.sh

MODEL_PATH=/workspace/code/models/Wan2.1-I2V-14B-720P \
bash scripts/wan/layout/run_contiguous_npu.sh
```

默认选择一张卡（`ASCEND_RT_VISIBLE_DEVICES=0`）、BF16、原始 Wan2.1 I2V 权重，使用 `baseline_npu.json`、`continuous_npu.json`。模型目录需要包含 DiT、T5、CLIP、VAE 及 tokenizer；FP8 checkpoint 不能作为 BF16 原始权重使用。T5、CLIP、VAE 也开启 CPU offload，两个方案配置相同。

`t5_rms_norm_type` 选择 T5 offload block 中的两处 RMSNorm 算子，独立于 DiT 的 `rms_norm_type`。两份 NPU 配置均显式设置为 `"torch"`；自定义 NPU 配置开启 T5 offload 时也应设置该字段。省略时默认使用 `"sgl-kernel"`，保留 CUDA 原有默认行为，T5 不再根据设备类型选择这两处算子。该选项不改变未开启 offload 的 T5 路径，也不改变 encoder 最后的归一化。

两个脚本直接调用 `python -m lightx2v.infer`，各执行一次完整推理，输出为 `baseline_npu.mp4`、`solution2_npu.mp4`。可用 `ASCEND_RT_VISIBLE_DEVICES` 选择卡，`CONFIG_PATH` 指定配置。

Wan NPU block offload 在模型初始化、权重创建前关闭 private format，使用 ND 存储。普通浮点和 INT8 checkpoint 在 dtype 转换前检查矩阵精度。NPU UniPC 小型系数系统在 CPU 求解，保留原来的采样时间表。

NPU 权重回写到非连续 CPU pinned 视图时，由 `lightx2v_platform/base/ascend_npu.py` 的 `copy_to_cpu` 先同步传回连续 CPU tensor，再按目标 stride 写入，避免转置权重错位。该分支保留原 pinned 存储，增加一次 CPU 拷贝；CUDA 仍使用原来的直接复制路径。

layout 脚本沿用 `scripts/base/base.sh` 的计时设置，当前为 `PROFILING_DEBUG_LEVEL=2`，会在计时边界同步设备。需要关闭调试计时时，在脚本的 `source` 行之后设置 `export PROFILING_DEBUG_LEVEL=0`。

## 连续布局

每个 CPU block 按最终 dtype、shape 和转置方式规划布局，分配一块 pinned `uint8` 存储。各 tensor 使用其中的视图，起始偏移按 256 字节对齐；checkpoint 直接写入最终视图，包含需要的 dtype 转换。这里的连续指虚拟地址连续，不要求物理页连续。

`BlockLayout` 描述字节区间，`BlockBuffer.view()` 创建对应 dtype 的视图。`BlockBuffer.allocate(layout, device)` 独立申请存储，并初始化 CPU 对齐填充；`BlockBuffer(layout, storage)` 则绑定已有的连续一维 `uint8` 存储，要求起始地址按 256 字节对齐、容量足够，绑定时不复制或修改内容。外部存储的填充区由调用方初始化，CPU offload 必须提供 pinned 存储。即使传入更大的区间，`buffer.storage` 也只覆盖 `layout.nbytes`；视图持有底层存储，相关搬运与计算完成前不能复用该区间。当前脚本仍为每个 CPU block 独立分配，没有启用全模型内存池或每步申请、释放。

两套设备 block buffer 使用相同布局。预取时一次 `storage.copy_` 将整个 CPU block 传入备用设备 buffer。推理期间 CPU block 权重只读，没有 pack、staging buffer 或重新分配。CUDA RMSNorm 的辅助占位标量保留独立 D2D 复制。

```text
模型权重容器：register_offload_group(blocks, device_slots, prefixes)
  → BaseTransformerModel 在 dtype 转换前记录原始 checkpoint 元数据
  → prepare_contiguous_groups
    → 算子 describe_storage → BlockLayout.build
  → WeightModule.load → BlockLoadPlan.load
    → BlockBuffer.allocate → 算子 load / bind_storage
    → 校验视图、权重覆盖范围和 pinned 状态

BaseTransformerModel._init_offload_manager
  → init_contiguous_groups
  → init_first_buffer(blocks)：选择分组并填充首个 slot
  → prefetch_weights → ContiguousBlockTransfer.copy
  → run_block → swap_blocks
```

`lightx2v/common/offload/block_loader.py` 统一规划和绑定权重，`block_layout.py` 管理存储和传输。Wan、Qwen Image 通过 `WeightModule.register_offload_group` 声明 block 边界和已有设备 slot，不再有 Wan 专用布局加载器或模型、任务白名单。每组要求 tensor 布局一致，并使用两个设备 slot；不同结构的 block 分组管理。切换分组时，先调用 `init_first_buffer`，再进行预取。执行顺序仍由模型原有推理代码负责。

算子通过 `lightx2v_platform/ops/weight_storage.py` 中的存储契约描述 checkpoint 名称、允许的原始 dtype、最终 shape/dtype、转置视图和设备辅助状态。普通加载器进行 dtype 转换前会保存原始元数据，避免错误的量化权重转换为 BF16 后通过校验。通用加载器不导入具体算子类，也不根据设备类型选择算子。未适配的算子、未声明的状态会在加载时拒绝；加载后脱离规划存储的 tensor 也会报错。

能接收 `BlockLoadContext` 的算子直接复用原有 `load`，只有需要不同绑定行为的算子实现 `bind_storage`。辅助设备 tensor 按 block 或 slot 各记录一次，复制时配对；管理器清理时关闭所有已注册的传输分组。

`lightx2v_platform/base/offload.py` 通过现有平台注册表选择内存后端。NVIDIA 使用 PyTorch 分配和异步复制；Ascend 复用这些操作，并在自己的后端准备 ND 存储。新平台需要显式声明并验证 block offload 能力，不能因为设备类型同为 `cuda` 就自动视为已支持。CUDA 导入路径不加载 Ascend 算子；Ascend 的非连续 CPU 目标 D2H 修复仍保留在 `base/ascend_npu.py`。

当前存储契约覆盖普通浮点 MM/Norm/Tensor、FP8-vLLM、INT8-NPU，以及这些算子的矩阵转置。接口通用不代表任意打包量化格式、权重别名、动态布局或所有模型算子均已适配。连续布局仍不支持共享权重、lazy load、并行、LoRA、CUDA Graph；现有启动脚本保持单卡、eager 和 NoCaching。

可在已配置依赖的 CUDA 机器执行存储与接入检查：

```bash
PLATFORM=cuda DTYPE=BF16 SENSITIVE_LAYER_DTYPE=None \
python -m pytest -q test_cases/test_block_buffer.py test_cases/test_block_offload_groups.py
```

Ascend 上将平台改为 `PLATFORM=ascend_npu`，执行同一命令；CUDA 专属用例会跳过。Ascend 算子的存储测试也可在 CUDA 上检查加载规则，但不能替代 Ascend 实机验证。

baseline 的 tensor 独立分配，不保证同一 block 内的地址连续。CUDA FP8 路径中部分 scale 的后续 dtype 转换可能使其不再 pinned；实际 pinned 状态由加载结果决定。

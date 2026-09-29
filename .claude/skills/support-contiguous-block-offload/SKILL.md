---
name: support-contiguous-block-offload
description: 为 LightX2V 模型接入、审查和调试 cpu_offload_layout=contiguous。覆盖公共 block 声明、算子存储契约、NVIDIA CUDA 与 Ascend NPU 平台隔离、baseline/contiguous 入口和正确性验证；用于持久连续 pinned 权重布局，不自动扩展到多进程共享权重或运行时打包。
---

# 连续 CPU Block Offload 接入

简体中文 | [English](SKILL_EN.md)

## 目标与范围

复用项目公共基建：加载时将每个 block 的主体权重写入最终连续 CPU pinned 存储，设备侧用同一布局建立 tensor views，推理时整 block H2D，并沿用模型已有计算和双缓冲调度。推理阶段不反复 pack 或分配 staging buffer。

连续指 block 内虚拟地址连续；block 之间不要求相邻，不保证物理页连续。它不消除权重传输，也不保证推理提速。辅助设备状态可能仍需单独复制。

按用户范围工作：分析请求只给方案；局部修复不重做完整模型接入。完整双平台任务检查 NVIDIA GPU 和 Ascend NPU 两套入口，分别报告实现与验证状态；用户限定单平台时保持该范围。硬件缺失时继续可完成的代码和本地验证，并明确未验证项。

本 skill 聚焦 rank 内私有连续权重。涉及共享权重、host/NUMA 副本时再参考 [support-cpu-block-offload](../support-cpu-block-offload/SKILL.md)，不继承其完整共享接入的交付要求。

## 按任务读取

- 修改 block 声明、加载或调度：读 [公共接入](references/common-integration.md)。
- 接入 CUDA 或修改两平台共用代码：读 [NVIDIA GPU](references/cuda.md)。
- 接入 Ascend：读 [Ascend NPU](references/ascend-npu.md)，涉及共用代码时同时检查 CUDA 回归要求。
- 借鉴模型定义与配置：读 [Wan 与 Qwen 案例](references/model-examples.md) 的对应部分。
- 选择测试或分析结果：读 [验证](references/validation.md)。

引用源码时以当前函数和文件为准，不依赖历史行号。本文及 references 中的仓库路径链接相对于当前 skill 目录。

## 接入流程

1. 从用户脚本追踪最终配置、runner、checkpoint 加载、weights、infer 和 offload manager。确认精度、任务及文本编码器、VAE 等组件范围。
2. 检查模型是否已有原生 block offload、两个设备 slots 和稳定的 block 粒度。记录 CPU block、slot、checkpoint 前缀以及每个权重算子的存储契约。
3. 将缺口放到所属层：模型声明 block；算子描述和绑定自身状态；公共层规划布局并管理生命周期；平台层实现分配、复制和特殊格式。
4. 先验证普通 per-tensor block baseline，再接连续布局。模型已有 block offload 且算子契约齐全时，优先仅补 block 注册；否则明确补齐缺失契约，不能声称注册 block 就能适配任意模型。
5. 提供任务要求的平台入口，执行与改动对应的测试和真实推理对照。最终说明改动、验证证据、限制及未完成的平台验证。

## 架构约束

- 使用 [WeightModule.register_offload_group](../../../lightx2v/common/modules/weight_module.py)、公共 [block_loader.py](../../../lightx2v/common/offload/block_loader.py) 和 [block_layout.py](../../../lightx2v/common/offload/block_layout.py)。不复制 Wan 专用 layout loader 为新模型建立第二套加载系统。
- 每个权重容器和 manager 只持有一个 group，CPU blocks 和设备 slots 的布局必须兼容。权重所有权不能重叠，未描述的权重或有状态算子不能静默跳过。
- 优先复用算子 `load(BlockLoadContext)`；仅在需要特殊绑定时增加 `bind_storage()`，不新增纯转发包装。
- 算子选择由配置决定。设备能力通过现有平台注册机制声明；不在模型或公共 loader 中增加逐芯片分支，也不为 Wan/Qwen 新建平台适配模块。
- NVIDIA 默认算子、精度和未开启连续布局时的行为应保持不变。平台特殊处理放到 `lightx2v_platform`，验证 CUDA 导入隔离和数值回归。
- 保持源存储存活、slot 覆盖依赖和当前 transfer 的关闭语义；不能只检查单次 H2D。

## 双平台交付

完整双平台接入沿用以下目录约定；已有文件保留用户设置，仅修改任务需要的部分：

```text
scripts/<model>/layout/
  run_baseline.sh
  run_contiguous.sh
  run_baseline_npu.sh
  run_contiguous_npu.sh
  README.md
  README_CN.md
configs/<model>/layout/
  baseline.json
  continuous.json
  baseline_npu.json
  continuous_npu.json
```

同一平台的 baseline/contiguous 保持模型、checkpoint、精度、输入、seed、采样、组件 offload 和计算算子一致，尽量只增加 `"cpu_offload_layout": "contiguous"`。脚本用 `contiguous`，配置文件沿用 `continuous` 命名。

脚本保持简洁，明确模型路径、设备选择和结果路径；不把开发机绝对路径当成通用要求。文档给出实际命令和精度要求。计时工具、下载、提交或推送仅在用户要求时处理。

交付时列出每个平台的入口、配置、实际使用的算子、验证范围和证据。区分“已添加入口”“存储/传输测试通过”“完整生成通过”“性能已测量”，不要用测试数量或旧版本记录代替当前版本的支持结论。

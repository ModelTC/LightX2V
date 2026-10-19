---
name: support-static-block-residency
description: 为 LightX2V 模型接入、审查和调试静态 block 常驻权重。用于独立测量计算与 H2D 耗时、直接估算常驻数量、生成配置、均匀分布常驻层，以及事件双缓冲、释放生命周期和 baseline 对比；按任务范围交付 CUDA/NPU 脚本与验证，不自动扩展到共享 CPU 权重或在线调参。
---

# LightX2V 静态 Block 常驻权重接入

简体中文 | [English](SKILL_EN.md)

## 目标与范围

复用模型的正式 block 计算路径，在独立工具中测量代表性 block 的计算时间 `c` 和 CPU 到设备的搬运时间 `d`，直接估算常驻数量 `r` 并写入推理配置。推理按配置均匀分布常驻层，其余层通过两个设备 slot 按需搬运，以尽量少的额外设备内存减少搬运等待。

当前实现针对 DiT blocks。所有 block 的 CPU 主权重仍保留；选中的层在一次去噪阶段持有额外设备权重，结束后释放。文本编码器、VAE、非 block 权重和激活不因此自动采用同一策略。`r=0` 表示没有额外常驻层，仍需要设备缓冲执行计算。

按用户范围工作：分析请求给方案；局部修复、解释或脚本修改只处理对应部分。完整接入包括估测、配置、推理调用链、启动入口和验证。默认发现与复用现有实现，不因使用本 skill 自动增加共享权重、并行、compile、模型下载或提交推送。

## 按任务读取

- 修改估测、配置写入、常驻管理或调度：读 [公共实现](references/implementation-patterns.md)。
- 参考 Wan 或 Qwen Image 的 block、输入与脚本：读 [模型案例](references/model-adapters.md) 的对应部分。
- 选择测试、计时或排查故障：读 [验证与调试](references/validation-and-debugging.md)。

模型尚无可靠的 CPU block offload 时，参考 [support-cpu-block-offload](../support-cpu-block-offload/SKILL.md) 的私有 block、权重绑定和生命周期部分；共享 CPU 权重及 host/NUMA 是独立任务，不继承其完整共享交付要求。需要修改连续布局本身时再读 [support-contiguous-block-offload](../support-contiguous-block-offload/SKILL.md)。

源码路径相对于各 Markdown 文件所在目录；按函数名核对当前实现，不依赖历史行号或旧实验结论。

## 接入流程

1. 从用户脚本追踪有效配置、runner、model、weights、infer 和 manager。记录设备、任务、checkpoint、精度、算子、输入规模、CPU 权重源与现有 slots。
2. 确认模型已有可运行的 block offload 基线，block 的大小与计算是否足以由一个代表性层估计。异构层、MoE 或明显不同的执行路径需先分析适用性，不静默套用同构假设。
3. 在现有 [估测工具](../../../tools/estimate_offload_residency.py) 中补充模型 case：加载一个真实 block，构造匹配正式推理尺寸、dtype、位置编码与状态结构的模拟输入。复用公共采样和数量计算，不另建模型专用搜索器。
4. 生成带 `offload_resident_blocks` 的配置与独立测量报告。推理读取配置，复用 [ResidentBlocks](../../../lightx2v/common/offload/resident.py) 和 [事件 slot manager](../../../lightx2v/common/offload/event_manager.py)，通过模型回调调用正式 `run_block()`。
5. 接通去噪结束、warmup 与异常退出的释放路径；保留 CPU 源，使下一次请求可以重建常驻权重。未选择新路径的配置保持既有行为。
6. 按目标平台交付简洁启动脚本、零常驻 baseline 和中英文说明，运行与改动相应的验证，分别报告实现状态与实测范围。

## 核心约束

- `r` 在独立工具中直接计算；推理不做候选枚举、重叠校准、在线搜索或逐步调节。实现验证不回流成估测器的搜索环节。
- 当前采样分别测量计算与搬运，各预热 3 次、正式采样 10 次、取中位数。沿用设备事件计时，输入重置不计入计算区间。模型加载与测量耗时不等同去噪耗时。
- 总耗时预算 `(N-r)*d <= N*c` 给出估计数量，不保证有限双缓冲下每次搬运都完全隐藏。`d <= c` 时取零常驻是正常结果。
- 常驻索引复用 `uniform_resident_indices()`；推理执行顺序保持不变。非驻留层的 slot 轮换按实际搬运顺序进行，不能简单用逻辑层号取模替代。
- `cpu_offload_block_contiguous` 独立控制布局：`false` 使用逐 tensor 布局，`true` 使用 block 连续布局；仅接受布尔值，默认 `false`。估测与推理保持相同布局和精度；更换设备、权重、输入规模或算子等测量条件后重新估测。
- 当前 `ResidentBlocks` 依赖 `WeightModule` 权重叶子和设备搬运接口。模型接入前检查这些契约，不假定它能直接管理任意原生 `nn.Module`。
- 常驻权重跨去噪步骤复用，并在 VAE 解码前释放；不能每步重建。同步、tensor 存活、slot 复用和异常恢复要求见公共实现。
- 当前校验要求单设备 eager、固定权重与分辨率、`NoCaching`；共享权重、并行、动态适配等兼容性以源码校验为准。保持用户指定的任务和精度，超出边界时说明需要的扩展，不静默删改配置制造成功。
- 工具不按剩余显存自动裁减 `r`。显存还包含两个 slots、非 block 权重、激活及其他组件；数量估计不是显存可运行保证。

## 启动入口与配置

完整双平台接入沿用以下结构。已有模型使用现有目录与命名；任务限定单平台或局部修改时保持该范围。

```text
scripts/<model>/inference/offload_residency/
  estimate.sh
  estimate_npu.sh
  run.sh
  run_npu.sh
  run_baseline.sh
  run_baseline_npu.sh
  README.md
  README_CN.md
configs/<model>/residency/
  baseline.json
  baseline_npu.json
```

基础配置纳入版本管理。baseline 使用 `cpu_offload=true`、`offload_granularity="block"`、`offload_resident_blocks=0`。配置中存在 `offload_resident_blocks` 即选择事件双缓冲路径，值为 0 时没有额外常驻层；省略该字段则保留旧路径，不再设置独立的事件开关。常驻脚本读取估测工具在 `save_results/` 生成的配置；报告和生成配置不作为固定硬件答案写回仓库示例。

脚本顶部明确项目路径、模型路径和设备，沿用 `scripts/base/base.sh`，保留完整 Python 命令与结果路径。开发机绝对路径是可编辑示例，不是通用要求。不为模板增加多层环境变量包装器、自动卡数推导或重复的任务/布局脚本。

启动脚本不添加 `set -e`、`export PROFILING_DEBUG_LEVEL=0` 或 `cd "${lightx2v_path}"`。配置、输入、输出和 `mkdir` 路径直接使用 `"${lightx2v_path}/..."`，让脚本保留调用时的工作目录。profiling 统一沿用 `base.sh`；需要调整时修改公共设置，baseline 与常驻使用相同级别。

同一平台 baseline 与常驻策略保持 checkpoint、task、输入、seed、步数、dtype、算子、布局和其他组件 offload 设置一致。baseline 可单独运行；常驻推理先运行对应估测。生成配置中的 `r` 可手动调低，重新估测会覆盖修改。

`README.md` 为英文，`README_CN.md` 为中文，添加双向语言链接。说明资产准备、修改位置、执行顺序、输出位置及计时口径；不把内部实现说明和历史实验报告堆入使用指南。

## 验证与交付

按 [验证与调试](references/validation-and-debugging.md) 选择覆盖本次变更的检查。对调度或生命周期改动，验证搬运次数、计算结果、连续执行、释放重建和异常路径；对文档修改检查内容与链接即可。

完整接入的交付说明包括：

1. 目标模型、组件、平台与任务范围，以及估测到推理的关键入口。
2. 测量配置、`c`、`d`、`N`、`r`、实际常驻索引及非常驻层数。
3. baseline/常驻脚本、基础/生成配置和中英文指南的位置。
4. 正确性与生命周期证据；任务要求性能时再给首步、稳定去噪和内存对比。
5. 未验证的平台、输入规模或组合及其原因。

硬件不足时完成可执行的交付和静态检查，明确设备验证尚未完成。配置可解析、`--help` 成功或 CUDA 结果不能替代 NPU 实测；小输入验证不等同默认目标规模通过。

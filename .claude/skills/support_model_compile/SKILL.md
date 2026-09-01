---
name: support-model-compile
description: 为 LightX2V 新模型或新推理路径实现、审查、调试和评测 `use_compile`/`torch.compile`。适用于普通推理以及模型原本已支持的 CPU offload model、block、phase 和 lazy-load；指导确认支持矩阵，选择 block/phase 编译边界与缓存键，处理层号、staging buffer、动态 guider、第三方算子、Dynamo 重编译，并用可信 warmup 区分首次编译成本、正式请求延迟和稳态收益。
---

# Support LightX2V Compile

## 目标

用最小改动让目标模型优先复用 LightX2V 的公共 block compile 机制，只支持 eager 已经可运行的模式，并保持 `use_compile=false` 的行为不变。

本 skill 只处理 `use_compile` 驱动的 `torch.compile`。不要混入 `use_magi_compile` 或 CUDA Graph；需要比较时另做实验。

若任务同时要求 warmup，或需要可信性能验收，而目标尚无可靠 warmup，先使用 [adapt-lightx2v-warmup](../support_model_warmup/SKILL.md) 完成 eager warmup，再接入 compile。两者共享正式算子路径，但各自只维护自己的生命周期和分派逻辑。

## 1. 先审计路径

从用户脚本开始，依次检查 config、model、transformer infer、weights 和 offload manager：

```bash
rg -n "use_compile|cpu_offload|offload_granularity|lazy_load" \
  scripts/<family> configs/<family> lightx2v/models/networks/<family>
rg -n "transformer_infer_class|infer_func|infer_with_.*offload|run_block|infer_block|infer_phase" \
  lightx2v/models/networks/<family>
rg -n "class .*TransformerInfer|def (infer_block|run_block)|\\.infer_block\\(" \
  lightx2v/models/networks/<family>
rg -n "cuda_buffers|prefetch|swap_|compute_stream|synchronize" \
  lightx2v/models/networks/<family> lightx2v/common/offload
```

启动 GPU 前解析最终 config，并确认 checkpoint、LoRA/adapter 和输入媒体存在。原配置缺少资产时报告阻塞；仓库内同 task 的有效配置只能用于功能 smoke test，不能替代原配置验收，也不能静默删除 LoRA 或改变模型语义。

不要根据相邻模型或配置名称推断能力。先由代码闭合正式调用链，再用用户原始脚本运行 no-warmup eager baseline，确认正式 shape 可运行；baseline 失败属于原路径问题，不属于 compile 适配。逐项填写：

| 模式 | eager | compile | compute 入口 | block/phase 对象生命周期 |
|---|---|---|---|---|
| normal | 支持/不支持/待验证 | 待实现/不适用 |  | 常驻/重建 |
| CPU model |  |  |  |  |
| CPU block |  |  |  | staging 数量与复用方式 |
| CPU phase |  |  |  | phase 族与 staging 数量 |
| lazy block |  |  |  | 请求间是否重建 |
| lazy phase |  |  |  | 请求间是否重建 |

只为 eager 已支持且用户要求的行增加 compile。某个模式的配置值可解析，不代表加载、infer、buffer 和 cleanup 已闭合。

动代码前写下这份设计记录：

```text
目标 infer 类：
各模式 compute 入口：
block/phase 族及各自签名：
每个族的编译边界：
是否读取逻辑层号：
常驻 block / staging buffer 数量：
block/phase cache key 与预期 cache 数量：
运行时分支和 request-specific 状态：
可能分流的第三方算子：
正式 graph signature（原始 shape/token/frame、packing/padding/SP 后各 rank 的本地 tensor shape、dtype/layout/branch/stage）：
warmup 是否覆盖相同 signature 和真实 leaf kernel：
```

这份记录必须由代码和 eager 实测支撑。无法回答其中一项时，继续检查，不要先实现。

## 2. 选择边界和缓存键

默认按同构 block 族（相同签名和语义）编译一个 transformer block。normal、model offload 和 block offload 可以共享该族的边界，权重传输仍留在图外。

模型只有一种 block 签名时，优先复用 `BaseTransformerInfer`：

1. 让目标 infer 类继承它；
2. 初始化时调用一次 `self.init_compile(config)`；
3. 保持计算入口为 `infer_block(block, *args)`；
4. block 循环调用 `self.run_block(block_idx, block, *args)`；
5. 只有缓存语义不同才覆写 `get_compile_block_key()`。

模型存在多种 block 族时，为每种稳定签名保留独立分派器和缓存，或让族名成为 cache key 的一部分。不要为了塞进一个 `infer_block` 而填充大量 `None` 参数或创建大分支；只有第二个模型也需要相同抽象时才上提公共层。

修改签名前枚举所有 override 和直接调用者。只适配目标脚本实际选择的 infer 类；未验证的 feature caching、AR 或专用 adapter 路径不要顺带声明支持。

按图真正依赖的身份选择键：

| 权重对象与图依赖 | cache key |
|---|---|
| 常驻逐层 block | `block_idx`（默认） |
| staging block 被多层复用，图不读取层号 | `id(block)` |
| staging block 被复用，图读取层号/层级 cache | `block_idx` |
| 固定 phase staging buffer，图与层号无关 | `phase_idx` |
| phase 读取层号 | `(block_idx, phase_idx)` |
| 多种 block/phase 族 | 独立 cache，或 `(family, 上述键)` |

缓存条目同时保存 block/phase 对象身份；同一键换了对象时重新创建 callable。

不要在公共 `run_block()` 中设置 `self.block_idx`。需要层号的模型由自己的循环或专用子类设置；无层号模型不应为 compile 增加隐藏状态。

阅读 [implementation-patterns.md](references/implementation-patterns.md) 获取公共骨架及 Wan、Qwen-Image、LTX2/LTX2.3 的具体对照。

## 3. 处理运行时状态

检查编译边界内的 Python 状态：

```bash
rg -n "self\\.block_idx|self\\.[A-Za-z0-9_]+\\s*=|is_compiling|is_dynamo_compiling" \
  lightx2v/models/networks/<family>/infer lightx2v/common/ops
```

- 层号被 attention、KV cache 或 adapter 深度使用：保留模型自己的字段，按逻辑层缓存。
- 少量 guider/branch 状态：在循环中计算成布尔值或小标量，显式传给 `infer_block`。
- request-specific tensor 在首个 block 内写入 `self`：进入 block 循环前按当前请求创建并刷新。
- dtype、维度和配置布尔值等稳定 Python 值：在初始化时解析一次，不要在 block 图内反复调用环境/config helper。
- 服务会用请求字段扩充共享 config；即使读取的是稳定 key，block 图内访问 `self.config` 也可能产生字典长度/键集合 guard。只缓存图真正需要的初始化常量，不要复制或冻结整份 config。
- 只有专用子类需要层号：只在该子类覆写 `run_block()`。

不要为追求“无状态”大范围重构；只显式化会进入图、影响正确性或导致重编译的状态。

## 4. 接入已有 offload

- **model**：整体 H2D/D2H 通常由 model 层处理；确认 transformer 最终走公共 block 循环，不新增专用 compile。
- **block**：在现有 compute stream 中将直接 `infer_block` 调用换成 `run_block`，保留 prefetch、swap 和同步顺序。
- **phase**：仅当 eager 已有 phase offload 时增加 `infer_phase → get_compiled_phase → run_phase`；按层号依赖选键，不支持的 phase 保持 eager。
- **lazy**：复用与普通 offload 相同的 CUDA staging buffer 和 cache，不维护第二套 lazy compile 逻辑；cache 随 infer 实例销毁，不能让 scheduler/runner 长期持有 compiled closure 和旧 staging 对象。

compile 不会优化图外的 CPU→GPU 拷贝、buffer 交换或 stream synchronize。不要为了 compile 把 offload manager 塞进图。

## 5. 审查算子

逐个检查 attention、RoPE、RMSNorm、量化 MM 和 Triton/第三方算子：

- compile 下是否通过 `is_compiling()`/`is_dynamo_compiling()` 换了 kernel；
- 是否存在动态图对象、mutation、data-dependent Python 分支或 fake/meta 缺失；
- eager 与 compile 是否数值一致；
- 性能对照是否真的使用相同 kernel。

需要隔离第三方实现时，在模块作用域定义 `torch.library.custom_op` 并注册 fake。custom op 只建立 Dynamo 图边界，不会消除其 kernel launch，也不会跨边界融合。算子内部的 Triton、CUDA extension 或第三方 JIT 仍可能按运行时签名编译和特化；按 [自定义算子内层编译与 Triton 动态标量](references/implementation-patterns.md#自定义算子内层编译与-triton-动态标量) 继续检查。

已确认不兼容且没有可靠 fallback 的组合应在初始化阶段明确报错，不要静默换 kernel。只有精确定位到公共算子并验证所有受影响路径后，才修改 `lightx2v/common/ops`。

保持公共默认：

```python
torch.compile(callable, dynamic=None)
```

不要默认加入 `mode="reduce-overhead"` 或宣称开启 CUDA Graph。

出现 cache 数量异常、重编译、kernel 分流、Step 1 冷启动、offload 负收益或 OOM 时，按 [casebook.md](references/casebook.md) 的证据链排查。

## 6. 验证正确性

聚焦单测至少覆盖：

- eager 不调用 `torch.compile`，也不产生无关层号状态；
- 同一键和对象只编译一次，换对象会刷新；
- normal/model 经过公共 dispatcher；
- block/phase/lazy 的 cache 数量和 prefetch/swap 顺序正确；
- guider 代表分支一致；
- 不支持的算子组合尽早报错。

执行：

```bash
python -m pytest -q <targeted-compile-tests>
ruff check <changed-files>
python -m py_compile <changed-python-files>
git diff --check
```

再用用户原始脚本逐一运行支持矩阵。eager/compile 保持相同模型、checkpoint、task、shape、seed、prompt 和 offload 参数。

如果兼容性要求两边使用不同 kernel，明确报告这是“完整优化栈对比”；只有实际 kernel 相同才能归因于纯 compile 收益。

最终扩散输出不一致时，先确认 eager 自身可复现，再比较单 block 或首个 denoise step，避免多步误差放大掩盖起点。视觉语义相近不能替代数值验收；按 [casebook.md](references/casebook.md#14-最终输出相近但数值不一致) 定位。

验收证据分两级：

- **功能 smoke test**：替代配置、较小 shape 或最小算子输入，只证明分派、Dynamo 或算子可运行。
- **目标路径验收**：用户原始脚本、资产、正式 shape 和运行模式完成 eager/compile，并以相同 seed 比较输出 tensor/hash；非确定性路径使用合理容差。

正式 shape 的 eager 本身 OOM 时，不要归因 compile；eager 可运行而 compile OOM 时，记录 tracing/Inductor/graph pool 的额外峰值。较小 shape 的成功不能证明正式 graph 已覆盖。

## 7. 评测稳态收益

两边都启用 warmup，并通过正式调用链覆盖相同的 graph signature、offload 对象生命周期和真实 leaf-op dispatch。graph signature 不只包含原始 shape/token/frame、dtype、layout、branch 和 stage；多模态或序列并行路径还应比较 packing/padding/SP 后各 rank 的本地 tensor shape，尤其是实际 q/k 长度。分辨率相同不等于 graph 相同。目标模型 warmup 不可靠时，先使用 `../support_model_warmup/SKILL.md` 修复，再评测 compile。

- 每个模式先完成一轮 warmup + 正式请求的功能验收；
- 每组至少三轮；
- 使用同卡或等价空闲卡；
- 出现外部进程时终止该轮，空闲后从 warmup 重跑；
- 正式 Step 1 出现目标 graph/Triton 编译时，先修复 warmup 覆盖或动态 guards；
- 多阶段模型分 stage 统计；
- 不把 warmup 编译耗时计入稳态吞吐，但必须单独报告。

每轮分别记录：

- compile/warmup 准备耗时；
- 正式 pipeline 耗时；
- 准备 + 正式 pipeline 的一次性冷启动总耗时；
- 各 stage 的 `infer_main cost`：Step 1、Step 2、Step 3–最后均值、Step 6–最后均值；
- warmup 前后和正式峰值显存。

计算：

```text
speedup = eager / compile
耗时下降 = (eager - compile) / eager
steady = mean(Step 3 ... 最后)
Step 1 绝对额外耗时 = Step 1 - steady
Step 1 相对额外耗时 = Step 1 / steady - 1
```

保留逐轮值并报告均值与离散度，不让单轮峰值被平均数隐藏。compile 稳态更快时，即使 Step 1 的绝对额外耗时更小，相对比例也可能更大；这只是分母效应，不能据此判定重编译。

warmup 后 Step 1 仍慢不等于重新编译。按 [casebook.md](references/casebook.md#7-warmup-后-step-1-仍慢) 区分 graph/kernel 编译、正式尺寸首次资源使用、数据依赖工作量和测量噪声。

分别给出首次请求、稳态吞吐和一次性冷启动结论。把编译移到 warmup 只改变成本发生时间，不等于总延迟下降；one-shot 场景可能仍应使用 eager。

## 完成标准

交付时同时给出：

1. 支持矩阵及未支持原因；
2. 编译边界、cache key 和预期 cache 数量；
3. 最小代码改动与兼容算子说明；
4. eager/compile 正确性证据；
5. 功能 smoke test 与目标路径验收的证据边界；
6. warmup 后分模式、分阶段性能表及显存；
7. 首次编译、正式请求、冷启动总成本、图外 offload 和稳态计算收益的独立结论。

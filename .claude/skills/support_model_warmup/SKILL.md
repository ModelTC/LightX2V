---
name: adapt-lightx2v-warmup
description: 为尚无 warmup 的 LightX2V 模型或新任务设计、实现、审查和验证 `--warmup`。先核对普通推理、CPU offload model/block/phase 和 lazy-load block/phase 的原生与 warmup 支持范围，再复用 Wan/Qwen-Image/LTX2/Lingbot-Video 经验覆盖 compile、MoE、多阶段或并行路径，检查 Encoder、DiT、scheduler 和 VAE decode 是否真正预热，并排查正式 Step 1 冷启动与请求状态污染。
---

# Adapt LightX2V Warmup

## 目标与边界

让 warmup 复用正式请求的真实算子路径，并保证正式请求：

- 一定在 warmup 完整结束后才开始；
- 不重复 warmup 已覆盖 signature 的 kernel、allocator 或 compile 冷启动；
- 不继承 warmup 的 seed、latent、solver history、分支或临时模型；
- eager、lazy-load 和非目标 task 的原有行为不变。

只修改完成该目标所需的 runner/scheduler。不要顺手重构正式 pipeline，也不要因 checkpoint/offload 格式问题修改 `mm_weight.py` 等公共权重基础设施。

若用户同时要求 warmup 和 compile，先用本 skill 完成 eager warmup 与状态隔离，再使用 [support-model-compile](../support_model_compile/SKILL.md) 接入 compile；不要在 warmup 代码中维护第二套编译逻辑。

## 1. 先审计运行模式

不要从脚本参数推断支持。先检查具体 `model_cls`、task、runner、model、transformer infer 和 weights：

```bash
rg -n "cpu_offload|offload_granularity|lazy_load" \
  scripts/<family> configs/<family> \
  lightx2v/models/{runners,networks}/<family>
rg -n "infer_with_.*offload|init_lazy_load|init_(cpu|cuda)_buffer|prefetch|swap_" \
  lightx2v/models/networks/<family> lightx2v/common/offload
```

实现前必须完成下表；“正式推理”和“warmup”要分别判断：

| 模式 | 正式推理 | warmup | 代码证据与生命周期 |
|---|---|---|---|
| normal | 支持/不支持/待验证 | 支持/不支持/待实现 | runner → model → infer |
| CPU model |  |  | 整模上下卡由谁负责 |
| CPU block |  |  | CUDA staging、prefetch、swap |
| CPU phase |  |  | phase buffer 和 phase infer |
| lazy block |  |  | CPU staging、磁盘预取、cleanup |
| lazy phase |  |  | CPU phase staging、磁盘预取、cleanup |

判定规则：

- 配置或脚本出现某个值不等于实现支持；加载、infer、权重 buffer 和卸载链必须闭合。
- family 的基础实现不能代表所有子类。Wan 专用 runner、Qwen layered、LTX2 版本和 task 都要单独确认。
- 正式推理不支持的模式，不要为了 warmup 顺手补齐；正式推理支持而 warmup 尚未支持时，才决定实现或显式拒绝。
- 若 `lazy_load` 只在 block/phase offload 中成立，不要把它当作独立模式或与 model offload 组合。

当前 Wan/Qwen-Image/LTX2 的代码快照和证据见 [implementation-patterns.md](references/implementation-patterns.md#运行模式快照)。代码变化后必须重新核对。

## 2. 从正式请求反推 warmup

先从用户入口脚本追到最终 config、runner 注册项和 `model_cls`，再定位公共入口和真实请求链：

```bash
rg -n "config_json|model_cls|task|runner" scripts/<family> configs/<family>
rg -n "def (warmup|run_warmup|init_modules|run_pipeline|end_run)" \
  lightx2v/models/runners
rg -n "def (prepare|reset|step_pre|step_post|clear)" \
  lightx2v/models/schedulers/<family>
rg -n "run_input_encoder|run_(text|image|vae)_encoder|run_vae_decoder|model\.infer" \
  lightx2v/models/runners/<family>
```

启动 GPU 任务前解析最终 config，并确认模型/checkpoint、LoRA/adapter 和输入媒体路径存在。原配置缺少资产时，先报告阻塞；可以用仓库内有效的同 task 配置做功能 smoke test，但不能把它当作原配置验收，也不能静默删除 LoRA 或改变模型语义。

固定前提：启用 `--warmup` 时，warmup 在进程启动阶段、首个正式 `run_pipeline()` 之前完整执行一次。具体模型的 warmup 代码直接依赖这一公共生命周期，不要兼容请求后的手动重入，也不要增加旧请求状态快照。

`BaseRunner` 在最外层 `init_modules()` 完成后调用一次 `warmup()`；此时 eager 模型已经加载、config 已锁定，已有 compile 初始化也已经完成。`DefaultRunner.warmup()` 负责检查 `--warmup`，并拒绝 disagg、`unload_modules` 和 feature caching 等不支持的模式。因此，一个继承 `DefaultRunner`、完全没有 warmup 的模型，通常只需在最接近的具体 runner 中新增 `run_warmup()`，不要修改 `infer.py`、复制公共入口、改写 config 或在 `run_pipeline()` 中增加 warmup 判断。

用户显式启用 `--warmup` 后，不支持的 model、版本、task 或运行模式必须抛出带原因的 `NotImplementedError`，不能只记录 warning 后继续正式请求。多个模型版本共用同一 runner 时必须检查版本特征，不能只依赖 runner 类型。

沿 `run_pipeline()` 逐行记录：

| 阶段 | 正式方法 | shape 来源 | 会留下的状态 |
|---|---|---|---|
| 输入 | Text/Image/VAE Encoder | input/config | conditioning、mask |
| 去噪 | prepare → step_pre → infer → step_post | latent shape | generator、latent、solver |
| 阶段转换 | upsampler/unpatchify 等 | 上阶段输出 | 新 scheduler 状态 |
| 输出 | VAE decode | 最终 latent | iterator、临时 VAE |
| 收尾 | end_run/clear | 请求边界 | cache、runner fields |

warmup 必须调用这些已有方法，而不是重新实现其中的数学或加载逻辑。不要直接调用带保存结果、完整 profiling 或请求级 cleanup 的 `run_pipeline()`。

## 3. 确定覆盖范围

在已确认的运行模式内，再确定完整 shape signature：分辨率、帧数、文本/图片 token、dtype、stride/layout、CFG/MoE 分支、阶段数、并行模式和加载生命周期。

默认规则：

- 每条 warmup 路径固定两种代表分辨率，定义为 runner 类常量，不从 config 传递。优先覆盖小尺寸/方形和可运行的生产尺寸/非方形；两种 shape 都必须进入真实 DiT 和 decoder。
- 先用目标设备确认 no-warmup 正式 shape 可运行。若正式 shape 本身 OOM，warmup 不负责让它变得可运行；可用两个较低面积的 shape 分别覆盖高度、宽度和长宽比做功能/eager kernel 预热，并明确记录生产 shape 未覆盖。compile 模式不能据此宣称生产 graph 已覆盖。
- OOM 时区分 live allocated tensor 与 allocator reserved cache；allocated 已接近显存上限时，`empty_cache()` 无法解决。
- 若两个 shape 无法覆盖互斥计算图，先说明缺口；不要静默增加第三种分辨率，也不要假装 dynamic 已覆盖。
- 可复用与分辨率无关的文本编码结果。
- I2V/I2I 每个分辨率都执行 Image/VAE Encoder。
- FLF2V 同时把首帧和尾帧送入 Image/VAE Encoder。
- 不跨分辨率复用 shape-dependent 输出。
- compile warmup 必须与正式请求使用相同 dtype、layout、分支和 leaf-op dispatch；仅分辨率相同不足以证明 graph/kernel 已覆盖。
- MoE/高低噪声模型覆盖每个 transformer 分支。
- 多阶段模型走真实阶段转换；并行模型的所有 rank 必须执行相同 collective 顺序。
- 专用子类显式 opt-in，避免通用 warmup 被 VACE、audio、animate、self-forcing 等任务误继承；不支持时直接报错。

不清楚且无法从代码判断的版本、task、分辨率或 lazy 支持范围，再询问用户。

## 4. 实现最小路径

只新增有独立职责的方法。`run_warmup()` 是公共 hook；仅当 eager/lazy 生命周期、主体循环或异常清理需要分层时，再拆出 `_run_warmup()`、`clear_warmup_state()`。一次使用的一行包装直接内联；输入准备只有在多个 task/shape 复用时才提取 helper。

每个目标 shape 至少执行：

```text
必要的 Text/Image/VAE Encoder
  → scheduler.prepare/reset
  → step_pre
  → model.infer
  → step_post
  → VAE decode
  → synchronize
```

在任何可能消耗请求随机数的 Image/VAE Encoder 之前将 `scheduler.generator` 置为 `None`。若正式 I2V 先用该 generator 采样 conditioning、再由 `scheduler.prepare()` 继续生成 noise，warmup 必须保持相同顺序，不能在 Encoder 之后重置 generator。

使用目标模型自己的 InputInfo、shape 计算、scheduler 和 decoder。dummy 数据只替代用户内容，不能替代正式阶段。

选择 step：

- 使用 `step_index=0` 覆盖日志中的正式 Step 1。
- 单一计算图通常只需 Step 0。
- 多模态 scheduler 即使没有显式分支，不同 step 的 unique timestep 数、embedding shape 或 AdaLN 索引布局也可能不同；读取 `step_pre()` 和 pre-infer，选择覆盖每种稳定 layout 的最少代表 step。
- 分支模型选择每个分支的首个有效 step。
- 非连续 step 是否需要 `reset/prepare` 取决于 scheduler：仅在存在多步 solver history、跨分支状态或其他不可复用状态时重置；单步且无历史依赖时不要额外重置。
- 若 Step 0 输出不能进入下一阶段或 VAE，再执行完成 unpatchify/finalize 所需的最后一步。
- 每次 infer 后都执行 `step_post()`。

decoder 返回 generator/iterator 时必须完整消费；只创建 iterator 不算预热。多阶段模型必须用真实 Stage 1 输出进入现有 upsampler/Stage 2 prepare。

CPU model offload 往往只在正式最后一步自动移回 CPU。warmup 若只执行首步，必须在每个 shape 的 `finally` 用现有 `model.to_cpu()` 恢复请求边界；多模型分支要恢复所有实际加载过的模型。

具体骨架和 Wan/Qwen-Image/LTX2 差异按需阅读 [implementation-patterns.md](references/implementation-patterns.md)。

## 5. 隔离状态和内存

只保存 warmup 确实会改写、且正式请求依赖的初始化状态，例如多阶段模型的 Stage 1 `infer_steps` 或 MoE 的 guidance。使用描述用途的名称，如 `stage1_infer_steps`，不要机械增加 `original_*` 快照。

由于 warmup 在首个正式请求前执行，`input_info` 应为 `None`、`inputs` 应不存在；结束时直接恢复这个启动基线，不要保存旧请求状态。

每个 shape 完成或异常后清理：

- latent/prediction/mask、timesteps/sigmas、solver history；
- CFG/MoE 分支和 request-specific RoPE/position cache；
- 临时输入、conditioning 和 transient module。

请求级 `scheduler.clear()` 应释放 generator，使下一请求 seed 生效；warmup 不恢复进入前可能已使用过的 generator，结束后保持 `None`，由正式请求根据自己的 seed 创建。Stage 1→2 仍属同一请求，不要在阶段间清理 generator。

保留 compile graph、kernel、eager 常驻权重和 CUDA allocator cache。重点搜索 warmup 返回到正式首次 DiT 之间的无条件 `empty_cache()`：

```bash
rg -n "empty_cache|maybe_empty_cache|gc\.collect" \
  lightx2v/models/runners/<family>
```

无条件 `empty_cache()` 会保留 kernel 预热，却释放 allocator block/workspace，使正式 Step 1 再次分配内存。eager 路径删除它或使用现有 pressure-aware cleanup；lazy cleanup 则在临时对象和引用释放后保留。

lazy-load 使用：

```text
load transformer → attach scheduler → warmup → synchronize
→ remove offload manager/transient modules → drop references
→ maybe_empty_cache(collect_garbage=True)
```

不要假设现有 lazy-load 可用；先确认 CPU/CUDA buffers、预取/swap 和每次 infer 的 shape cache reset。eager 在 `_run_warmup()` 返回、临时引用消失后才允许 `gc.freeze()`；lazy 不 freeze。当前公共 gate 会拒绝 `unload_modules`，除非任务明确要求，否则不要扩大支持范围。

lazy + compile cleanup 还要断开 `scheduler.transformer_infer`、compiled callable closure、offload manager 和 staging buffer 之间的强引用；只写 `self.model = None` 不够。释放局部 model/multi-model 引用后再回收，并用 `weakref` 单测确认对象确实消失。

不要保留 request-specific RoPE/position cache 来换取表面提速；只有已经按完整 shape signature 安全键控的不可变 cache 才能跨请求复用。block/phase offload 还要确认 warmup 结束后 staging buffer、预取索引和 stream 状态正好处于下一次 infer 可接受的边界。

## 6. 验证

运行：

```bash
python -m pytest -q <targeted-warmup-tests>
ruff check <changed-files>
python -m py_compile <changed-python-files>
git diff --check
```

在 `test_cases/test_<family>_warmup.py` 增加针对性测试，至少覆盖：model/version/task/subclass/mode guard 的异常、每个 shape 的 encoder/step/decode 顺序、iterator 消费、CPU model 最终位置、generator 清空、必要 scheduler 状态恢复、input 启动基线、异常 cleanup，以及 scheduler `clear()` 的请求边界语义。不要在每个模型中重复测试公共 warmup 调用顺序。

只测试模式矩阵中正式推理已支持的行。功能验收时，每个目标模式至少完成一轮 warmup + 正式请求；要给出可靠性能结论时，使用用户原始脚本和 checkpoint 分别运行 warmup/no-warmup 至少三轮。需要时追加 `--return_result_tensor` 避免保存，但不要改变模型、task、config 或 shape。若因资产缺失或原正式 shape OOM 使用替代配置/shape，只能记为 smoke test，并单独保留原配置失败证据。offload、lazy 和多阶段分别对照。

每轮记录：

- 每个 shape/encoder/DiT branch/stage/decode 是否真正执行；
- 同一阶段 `infer_main cost` 的 Step 1、Step 2、Step 3–最后和 Step 6–最后；
- warmup 自身、正式 pipeline，以及 `warmup + 正式 pipeline` 相对 no-warmup pipeline 的冷启动代价；
- warmup 前后显存，以及正式 Step 1 前的 cache 清理；
- traceback 实际发生在加载、warmup 还是正式请求。

验收时确认：

1. warmup 覆盖目标 shape signature，并消除其中可避免的 compile/kernel/allocator 冷启动；
2. 正式 seed、输入和连续请求不受 warmup 状态污染；相同 seed 在确定性路径上比较输出哈希/tensor，非确定性路径使用合理容差；
3. 目标包含 lazy 时，warmup 后能完成正式请求；
4. eager、no-warmup、非目标 task 和正式保存行为不变。

不要要求所有模式的正式 Step 1 都等于稳态：CPU model 的整模上卡、每步 block/phase 传输与同步、lazy 正式请求重建模型/offload manager、请求级 cache 重算都不是 warmup 可以保留的状态。不要以单轮耗时或只有外层 `Warmup cost` 日志为结论；按 [implementation-patterns.md](references/implementation-patterns.md#冷启动边界与典型误区) 区分 warmup 缺陷、生命周期固有成本、原有 offload/lazy 问题和 profiling 偏差。

交付前给出：运行模式矩阵、正式路径与 warmup 路径对应关系、两个 shape 和代表 step 的依据、状态清理清单、功能/性能证据等级、正式请求与冷启动总延迟、测试结果，以及剩余 Step 1 成本的归因。分别说明首请求、稳态吞吐和一次性启动成本的变化，不要把移到服务就绪前的成本称为稳态加速。

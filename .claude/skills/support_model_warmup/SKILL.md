---
name: adapt-lightx2v-warmup
description: 为尚无 warmup 的 LightX2V 模型或新任务设计、实现、审查和验证 `--warmup`。适用于新增 runner warmup，复用 Wan/Qwen-Image/LTX2 经验，覆盖 eager、lazy-load、compile、MoE、多阶段或并行路径，检查 Text/Image/VAE Encoder、DiT、scheduler 和 VAE decode 是否真正预热，并排查正式 Step 1 冷启动、generator/cache/GC 状态污染。
---

# Adapt LightX2V Warmup

## 目标与边界

让 warmup 复用正式请求的真实算子路径，并保证正式请求：

- 不承担可避免的 kernel、allocator 或 compile 冷启动；
- 不继承 warmup 的 seed、latent、solver history、分支或临时模型；
- eager、lazy-load 和非目标 task 的原有行为不变。

只修改完成该目标所需的 runner/scheduler。不要顺手重构正式 pipeline，也不要因 checkpoint/offload 格式问题修改 `mm_weight.py` 等公共权重基础设施。

## 1. 从正式请求反推 warmup

先定位公共入口和目标模型的真实请求链：

```bash
rg -n "def (warmup|run_warmup|init_modules|run_pipeline|end_run)" \
  lightx2v/models/runners
rg -n "def (prepare|reset|step_pre|step_post|clear)" \
  lightx2v/models/schedulers/<family>
rg -n "run_input_encoder|run_(text|image|vae)_encoder|run_vae_decoder|model\.infer" \
  lightx2v/models/runners/<family>
```

当前公共约定是：`BaseRunner` 在最外层 `init_modules()` 完成后调用一次 `warmup()`；`DefaultRunner.warmup()` 负责检查 `--warmup` 和跳过条件。因此，一个继承 `DefaultRunner`、完全没有 warmup 的模型，通常只需在最接近的具体 runner 中新增 `run_warmup()`，不要修改 `infer.py` 或复制公共入口。先以当前代码重新确认这一约定，并检查目标 config 是否因 disagg、`unload_modules` 或 feature caching 被公共 gate 跳过。

沿 `run_pipeline()` 逐行记录：

| 阶段 | 正式方法 | shape 来源 | 会留下的状态 |
|---|---|---|---|
| 输入 | Text/Image/VAE Encoder | input/config | conditioning、mask |
| 去噪 | prepare → step_pre → infer → step_post | latent shape | generator、latent、solver |
| 阶段转换 | upsampler/unpatchify 等 | 上阶段输出 | 新 scheduler 状态 |
| 输出 | VAE decode | 最终 latent | iterator、临时 VAE |
| 收尾 | end_run/clear | 请求边界 | cache、runner fields |

warmup 必须调用这些已有方法，而不是重新实现其中的数学或加载逻辑。不要直接调用带保存结果、完整 profiling 或请求级 cleanup 的 `run_pipeline()`。

## 2. 确定覆盖范围

实现前确认 task、目标分辨率、compile shape、CFG/MoE 分支、阶段数、并行模式和 eager/lazy 生命周期。

默认规则：

- 可复用与分辨率无关的文本编码结果。
- I2V/I2I 每个分辨率都执行 Image/VAE Encoder。
- FLF2V 同时把首帧和尾帧送入 Image/VAE Encoder。
- 不跨分辨率复用 shape-dependent 输出。
- compile/dynamic 使用多个 shape 时，每个 shape 都必须进入真实 DiT。
- MoE/高低噪声模型覆盖每个 transformer 分支。
- 多阶段模型走真实阶段转换；并行模型的所有 rank 必须执行相同 collective 顺序。
- 专用子类显式 opt-in，避免通用 warmup 被 VACE、audio、animate、self-forcing 等任务误继承。

不清楚且无法从代码判断的 task、分辨率或 lazy 支持范围，再询问用户。

## 3. 实现最小路径

优先只新增：

- `run_warmup()`：task guard 和 eager/lazy 生命周期；
- `_run_warmup()`：真实算子路径；
- `clear_warmup_state()`：请求状态清理。

只有 prepare 逻辑被多个 task/shape 复用且能明显缩短主流程时，才新增一个输入 helper。

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

使用目标模型自己的 InputInfo、shape 计算、scheduler 和 decoder。dummy 数据只替代用户内容，不能替代正式阶段。

选择 step：

- 使用 Step 0 覆盖正式请求的首次 DiT 路径。
- 单一计算图通常只需 Step 0。
- 分支模型选择每个分支的首个有效 step。
- 非连续 step 之间使用 scheduler 现有的 `reset/prepare` 清除 solver history。
- 若 Step 0 输出不能进入下一阶段或 VAE，再执行完成 unpatchify/finalize 所需的最后一步。
- 每次 infer 后都执行 `step_post()`。

decoder 返回 generator/iterator 时必须完整消费；只创建 iterator 不算预热。多阶段模型必须用真实 Stage 1 输出进入现有 upsampler/Stage 2 prepare。

具体骨架和 Wan/Qwen-Image/LTX2 差异按需阅读 [implementation-patterns.md](references/implementation-patterns.md)。

## 4. 隔离状态和内存

在最外层 `finally` 恢复 warmup 改动的外部状态：

- scheduler generator、infer steps、sigma/guidance 配置；
- runner 原有 `input_info`、`inputs`；
- 其他正式请求可见配置。

每个 shape 完成或异常后清理：

- latent/prediction/mask、timesteps/sigmas、solver history；
- CFG/MoE 分支和 request-specific RoPE/position cache；
- 临时输入、conditioning 和 transient module。

请求级 `scheduler.clear()` 应释放 generator，使下一请求 seed 生效；Stage 1→2 仍属同一请求，不要在阶段间清理 generator。warmup 保存并恢复进入前的 generator。

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
→ gc.collect + empty_cache
```

不要假设现有 lazy-load 可用；先确认 CPU/CUDA buffers、预取/swap 和每次 infer 的 shape cache reset。eager 在 `_run_warmup()` 返回、临时引用消失后才允许 `gc.freeze()`；lazy/unload 不 freeze。当前公共 gate 会跳过 `unload_modules`，除非任务明确要求，否则不要扩大支持范围。

## 5. 验证

运行：

```bash
python -m pytest -q <targeted-warmup-tests>
ruff check <changed-files>
python -m py_compile <changed-python-files>
git diff --check
```

单测至少覆盖：task/subclass guard、每个 shape 的 encoder/step/decode 顺序、iterator 消费、generator/config/input 恢复、异常 cleanup，以及 scheduler `clear()` 的请求边界语义。

使用用户原始脚本和 checkpoint，分别运行 warmup/no-warmup 至少三轮；需要时追加 `--return_result_tensor` 避免保存，但不要改变模型、task、config 或 shape。lazy 和多阶段分别对照。

每轮记录：

- 每个 shape/encoder/DiT branch/stage/decode 是否真正执行；
- 同一阶段正式 Step 1–5 的 `infer_main cost`；
- warmup 前后显存，以及正式 Step 1 前的 cache 清理；
- traceback 实际发生在加载、warmup 还是正式请求。

验收时确认：

1. warmup 覆盖目标路径，正式 Step 1 多轮落在后续同类 step 的正常抖动范围；
2. 正式 seed、输入和连续请求不受 warmup 状态污染；
3. lazy warmup 后能完成正式请求；
4. eager、no-warmup、非目标 task 和正式保存行为不变。

不要以单轮耗时或只有外层 `Warmup cost` 日志为结论。区分 warmup 缺陷、原有 lazy/offload 问题、checkpoint 格式问题和 profiling 范围偏差。

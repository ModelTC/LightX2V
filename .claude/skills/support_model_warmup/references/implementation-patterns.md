# LightX2V Warmup Implementation Patterns

仅在需要代码骨架、family 特例或性能诊断时阅读本文件。以当前仓库为准，用 `rg` 重新定位接口，不要机械复制。

## 目录

- [参考入口](#参考入口)
- [运行模式快照](#运行模式快照)
- [没有 warmup 的模型从哪里开始](#没有-warmup-的模型从哪里开始)
- [Runner 骨架](#runner-骨架)
- [Wan 模式](#wan-模式)
- [Qwen-Image 模式](#qwen-image-模式)
- [LTX2 模式](#ltx2-模式)
- [Lingbot-Video 模式](#lingbot-video-模式)
- [冷启动边界与典型误区](#冷启动边界与典型误区)
- [Scheduler clear 合同](#scheduler-clear-合同)
- [Allocator cache 反模式](#allocator-cache-反模式)
- [实验模板](#实验模板)

## 参考入口

| 用途 | 参考位置 |
|---|---|
| 公共 warmup 生命周期、GC | `lightx2v/models/runners/base_runner.py` |
| warmup 开关和拒绝条件 | `lightx2v/models/runners/default_runner.py` |
| Wan task/MoE/lazy 模式 | `lightx2v/models/runners/wan/wan_runner.py` |
| Wan distill/MoE 分支 | `lightx2v/models/runners/wan/wan_runner.py` |
| Wan 状态清理 | `lightx2v/models/schedulers/wan/scheduler.py` |
| Qwen T2I/I2I 模式 | `lightx2v/models/runners/qwen_image/qwen_image_runner.py` |
| Qwen RoPE/diffusers 状态清理 | `lightx2v/models/schedulers/qwen_image/scheduler.py` |
| LTX2 多阶段/iterator 模式 | `lightx2v/models/runners/ltx2/ltx2_runner.py` |
| LTX2 latent/sigma 清理 | `lightx2v/models/schedulers/ltx2/scheduler.py` |
| 现有 warmup 单测 | `test_cases/test_{wan,qwen_image,ltx2,lingbot_video}_warmup.py` |

## 运行模式快照

这是当前基础实现的导航，不是永久能力声明。表中每格为“正式推理 / warmup”；适配时仍须按具体 `model_cls`、task 和子类重新检查。

| 模型范围 | normal | CPU model | CPU block | CPU phase | lazy block | lazy phase |
|---|---|---|---|---|---|---|
| Wan 通用 T2V/I2V/FLF2V 路径 | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| Qwen-Image T2I/I2I，非 layered | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| LTX2 T2AV/I2AV | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | — / — | — / — | — / — |
| Lingbot-Video T2I/T2V/I2V | ✓ / ✓ | — / — | — / — | — / — | — / — | — / — |

代码依据：

- Wan offload infer 选择 model/block/phase；weights 同时提供 block/phase CUDA staging 和 lazy CPU staging；runner 的 lazy warmup 复用正式加载并在 `finally` 清理。专用 Wan 子类可能缩小范围，必须单查。
- Qwen-Image model 负责整模上下卡，offload infer 和 weights 实现 block/phase 及两种 lazy staging；runner 支持 T2I/I2I eager/lazy warmup，但拒绝 layered。
- LTX2 offload infer 只接受 model/block，phase 会报错。当前 weights 没有 lazy block CPU staging，runner 也明确拒绝 lazy warmup，因此不能把代码中的 `lazy_load` 分支当作完整支持。
- Lingbot-Video model 直接拒绝 CPU offload，runner 拒绝 lazy/unload；warmup 只覆盖 normal。

“—”表示当前链路不闭合。不要在 warmup 适配中顺手实现缺失的正式 offload/lazy 能力。

## 没有 warmup 的模型从哪里开始

固定前提：公共 hook 保证 warmup 在首个正式请求前执行一次。具体 runner 不处理请求后重入。

1. 先完成主文的运行模式矩阵，区分正式推理能力和 warmup 能力。
2. 确认 runner 是否继承 `DefaultRunner`，以及哪个具体 `init_modules()` 被公共 hook 包装。
3. 沿正式 `run_pipeline()` 标出 encoder、scheduler prepare、单步循环、阶段转换、decoder 和 `end_run()`。
4. 在共享这些算子图的最小 runner 类上新增 `run_warmup()`；不修改 `infer.py` 或公共 hook。
5. 若该 runner 有不同算子图的子类，让子类显式 opt-in。
6. 先完成 normal 单 task，再逐项复用已验证模式的加载和卸载方法。

eager 通常只需要 `run_warmup()`、`_run_warmup()` 和 `clear_warmup_state()`；lazy 若没有可复用的正式 cleanup，再增加一个 cleanup 方法。不要为了统一命名搬动正式函数；只有输入准备在多个 shape/task 中重复时才增加 helper。

## Runner 骨架

按目标 runner 调整占位符：

```python
@ProfilingContext4DebugL1("Warmup")
def run_warmup(self):
    if not self.supports_warmup():
        raise NotImplementedError(...)

    lazy_load = self.config.get("lazy_load", False)
    try:
        if lazy_load:
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)
        self._run_warmup()
    finally:
        if lazy_load:
            self.clean_lazy_load_warmup()

    self._maybe_freeze_gc()
```

要求：

- `supports_warmup()` 只是 task/subclass guard 的占位符；不要只为套骨架新增无意义方法。
- 骨架中的 helper 名只是职责提示；一次使用的薄包装应内联。
- 多个模型版本共用 runner 时同时检查版本特征；显式 `--warmup` 遇到不支持的 model、版本、task 或模式必须报错，不能 warning 后继续。
- 不要在 `run_pipeline()` 中增加 warmup 判断、锁或一次性状态。
- lazy cleanup 必须位于 `finally`。
- `_maybe_freeze_gc()` 必须在 `_run_warmup()` 返回后。
- lazy 模式由公共方法跳过 freeze。

核心循环：

```python
def _run_warmup(self):
    scheduler = self.model.scheduler

    for height, width in self.WARMUP_RESOLUTIONS:
        try:
            scheduler.generator = None
            inputs = self.prepare_warmup_inputs(height, width)
            scheduler.prepare(...)

            # 仅当该 scheduler 不能跨 step gap 复用内部状态时，在此加入 reset。
            for step_index in self.get_warmup_step_indices(scheduler):
                scheduler.step_pre(step_index)
                self.model.infer(inputs)
                scheduler.step_post()

            output = self.run_vae_decoder(...)
            consume_if_iterator(output)
            torch_device_module.synchronize()
        finally:
            self.clear_warmup_state()
            self.input_info = None
            self.__dict__.pop("inputs", None)
```

只保存本实现实际修改且后续不会自动恢复的 infer steps、sigma、guidance 等状态；不要为启动时必然为空的 `input_info/inputs` 增加快照。`clear_warmup_state()` 必须让 generator 保持 `None`，正式请求再按自己的 seed 创建。`prepare/reset` 参数和非连续 step 语义必须来自该 scheduler；仅在多步 solver history、跨分支状态或其他内部状态不能跨 gap 复用时重置。若确实需要重置但没有 `reset(step_index=...)`，重新执行正式 `prepare()`，不要发明兼容接口。

generator 必须在输入准备前清空，因为 I2V 的 VAE Encoder 可能先用它采样 conditioning，随后 scheduler 才继续生成初始 noise。Encoder 后再清空会改变正式请求的随机数消费顺序。

## Wan 模式

使用场景：

- T2V/I2V/FLF2V；
- Wan2.2 dense/MoE；
- changing resolution；
- distill scheduler。

关键点：

- 当前通用路径固定 warmup `480×480` 和 `720×1280`，支持 T2V/I2V/FLF2V；子类必须显式 opt-in。
- T2V 只需要文本输入。
- I2V 每个 warmup shape 执行 Image Encoder 和 VAE Encoder。
- FLF2V 为首尾帧分别构造输入，并把两帧同时送入 encoder。
- 普通单模型通常使用 Step 0。
- MoE 选择 high-noise 和 low-noise 各自第一个有效 step。
- Wan UniPC/MoE 的两个代表 step 不连续时 reset scheduler，避免复用错误的 solver/分支状态。
- Wan MoE 会在分支切换时修改 `sample_guide_scale`；warmup 前保存初始化值并在最外层 `finally` 恢复。
- 专用 Wan 子类必须自行声明支持，避免 VACE、audio、animate、self-forcing 等错误继承。

## Qwen-Image 模式

使用场景：

- T2I；
- I2I；
- packed image latent；
- position/RoPE cache。

关键点：

- 当前实现固定 warmup `480×480` 和 `832×1248`，支持非 layered 的 T2I/I2I。
- T2I 文本编码与分辨率无关时可跨 shape 复用。
- I2I 每个 shape 重新运行包含图片的 Text Encoder 和 VAE Encoder。
- 直接执行一个 `step_pre → infer → step_post`；不要用带额外 profiling 的 `run(total_steps=1)`。
- Step 0 输出可直接进入现有 VAE decode。
- 清理 scheduler 中的 packed latent、image ids、timesteps、RoPE、内部 diffusers `_step_index/_begin_index`。
- 清理 model pre-infer 的 request-specific RoPE cache。
- warmup 只发生在首个请求前；每个 shape 的 `finally` 直接把 `input_info` 设为 `None` 并移除 `inputs`，不要保存旧请求快照。
- 不要因为正式 I2I shape 与 warmup shape 不同就断言首步偏慢；至少做三轮实测。

## LTX2 模式

使用场景：

- T2AV/I2AV；
- Stage 1 + spatial upsampler + Stage 2；
- video decode iterator；
- video/audio latent 双 scheduler state。

关键点：

- 当前实现支持 T2AV/I2AV；单阶段固定 warmup `480×480`、`512×768`，upsampler 路径固定 `480×480`、`1024×1536`。
- 根据 VAE spatial factor 对齐 Stage 1 像素尺寸。
- upsampler warmup 的输入 shape 是最终目标尺寸；Stage 1 使用除以 upsample scale 后的对齐尺寸。
- Step 0 用于覆盖正式首个 DiT forward。
- 最后一步用于 unpatchify video/audio latent；没有最后一步时，输出不能可靠进入 Stage 2/VAE。
- 当前基础 Euler scheduler 没有多步 history，首尾 step 不连续也无需 reset。
- Stage 2 必须调用现有 upsampler/VAE encoder 的准备路径。
- `video_vae.decode()` 返回 iterator；必须迭代完成。
- scheduler clear 同时释放 video/audio latent、prediction、sigma 和 generator。
- `reset_sigmas()` 会在 Stage 2 改变 `infer_steps`；保存为 `stage1_infer_steps`，在每个 warmup shape 开始前和最外层 `finally` 恢复。

### LTX2 lazy/offload 审查

当前只支持 CPU model/block warmup，显式拒绝 phase 和 lazy。仅在仓库后续补齐正式 lazy 推理时，才重新审查并扩展 warmup。

同步检查：

```text
lightx2v/models/networks/ltx2/weights/transformer_weights.py
lightx2v/models/networks/ltx2/infer/offload/transformer_infer.py
lightx2v/common/offload/manager.py
```

验证 CPU/CUDA buffers、disk prefetch、CPU swap 和每次 infer 的 `reset_infer_states()`。加载链未闭合时不能靠 warmup 绕过；若失败在 `load_transformer()`，外层 `Warmup cost` 也不代表 block 已被预热。

## Lingbot-Video 模式

使用场景：T2I、T2V、I2V normal。

关键点：

- 当前固定 warmup `480×480` 和 `320×832`；两个较低面积 shape 分别覆盖目标高度、宽度和横屏形态，避免 warmup 自身越过大视频 live tensor 峰值。
- T2I/T2V 跨 shape 复用文本编码；I2V 每个 shape 用内存图片重走 VAE/VLM 条件编码。
- I2V VAE Encoder 会先消费 scheduler generator；每个 shape 必须先将 generator 置空，再准备输入，最后由 scheduler 继续生成 noise。
- 单一 transformer 路径执行 Step 0 即可；decode 后清理 conditioning、generator、latents、timesteps、sigmas 和 solver history。
- scheduler 继承 Wan 时复用其完整 `clear()`，再清理 Lingbot 独有状态，不要复制一份不完整字段列表。

## 冷启动边界与典型误区

warmup 可以保留 compile graph、已选择的 kernel、eager allocator cache，以及进程或系统允许复用的文件 cache。它不能消除正式请求必须重新执行的工作：

- CPU model offload 的整模 CPU→GPU；
- block/phase offload 每一步的权重传输、buffer swap 和 stream synchronize；
- lazy cleanup 后重建的模型、offload manager、staging buffer 和对象状态；
- 为正确性而清理的 request-specific RoPE/position cache；
- 正式输入独有的 encoder、shape、dtype、layout 或分支。

因此先比较 warmup/no-warmup，再把正式 Step 1 拆成“可避免冷启动”和“生命周期固有成本”，不能只与 Step 2 做相等性判断。

| 误区或现象 | 正确判断 |
|---|---|
| 脚本或 config 接受某个 offload 值，就视为支持 | 必须同时找到 infer 分派、weights staging、manager 初始化和 cleanup |
| 直接调用完整 `run_pipeline()` 最真实 | 会混入保存、profiling、请求 cleanup；应复用其内部正式方法 |
| 用一个 dummy forward 或只创建 decoder iterator | 没覆盖真实 scheduler/分支，iterator 也必须完整消费 |
| 从目标 config 读取一个分辨率 | 每条路径使用两个固定类常量，并覆盖不同 shape regime |
| 原配置缺少 LoRA/checkpoint 时静默删除后继续 | 替代配置只能做 smoke test；原配置仍是未验证/阻塞 |
| 生产 shape 在目标设备本身 OOM，仍强行用于 warmup | 先用 no-warmup 复现；可用较低面积 shape 覆盖不同轴，但不能声称 exact compile graph 已覆盖 |
| 相同 H×W 就能预热 compile | 还要一致的帧/token、dtype、stride/layout、CFG/MoE 和 Dynamo leaf-op 分派 |
| I2V Encoder 后再把 generator 设为 `None` | 会改变 conditioning 与初始 noise 的随机数顺序；必须在可能消费 RNG 的 Encoder 前重置 |
| 所有模型只跑 Step 0 | 分支模型要覆盖各分支；需 unpatchify/finalize 的模型还要跑最后一步 |
| 非连续 step 一律 reset，或一律不 reset | 读取 scheduler history；只在状态无法跨 gap 复用时 reset/prepare |
| 保留 generator 或请求级 RoPE cache 可以提速 | 会污染 seed 或 shape；请求状态必须清理，安全的持久 cache 应由正式实现定义 |
| warmup 后调用无条件 `empty_cache()` 更干净 | 会释放 allocator/workspace，正式 Step 1 再次分配 |
| model offload 的 Step 1 仍慢说明 warmup 失败 | 正式 Step 1 必须重新整模上卡；只归因 warmup 可消除的剩余部分 |
| lazy warmup 后正式请求不应再初始化 manager | lazy cleanup 会释放模型和 manager；这是内存语义，不应为耗时保留悬挂引用 |
| lazy cleanup 只需 `self.model = None` | scheduler 和 compiled closure 仍可能持有 infer/staging；断开引用并用 `weakref` 验证 |
| warmup 日志出现新 Triton 编译就继续增加轮数 | 先找正式请求与 warmup 的 graph guard、shape、分支或 leaf kernel 差异 |
| VAE 的 OOM/32-bit index 或加载失败属于 warmup bug | 用对应 no-warmup 路径复现；相同位置失败是模型/decoder/offload 基础问题 |
| 为修 warmup 改公共权重或 offload 基础设施 | 除非正式推理本身有已确认缺陷，否则保持 warmup 改动在 runner/scheduler |

## Scheduler clear 合同

按 family 的实际字段调整：

| 类别 | clear |
|---|---|
| 随机状态 | generator |
| latent | latent/video/audio state、mask、conditioning latent |
| 当前输出 | cond/uncond/guided noise prediction |
| solver | model outputs、timestep history、last sample、order |
| 请求索引 | step index、begin index、CFG/MoE branch |
| shape cache | request-specific RoPE/position/cu_seqlens |

不要清理：

- 编译 graph；
- kernel cache；
- eager 模式希望保留的 allocator cache；
- 模型常驻权重。

检查 `end_run()` 调用链。若正式请求也调用 `clear()`，确认变更符合重复请求语义。

## Allocator cache 反模式

重点检查 warmup 返回后、正式首次 DiT 之前：

```python
text_encoder_output = self.run_text_encoder(self.input_info)
torch_device_module.empty_cache()
gc.collect()
```

典型时序：

```text
warmup DiT/VAE decode 完成
  → 正式 Text/Image/VAE Encoder
  → empty_cache()
  → 正式 DiT Step 1
```

kernel/compile 仍已预热，但 allocator block/workspace 被释放。典型表现：

- warmup 日志显示 encoder、DiT 和 decode 均已完整执行；
- 正式 Step 1 仍稳定慢于同阶段 Step 2–5；
- Step 1 前出现 GPU 空档，功耗或利用率需要重新爬升；
- scheduler、step 选择和正式 shape 均已确认正确。

处理：

- eager 删除强制清理或使用 `self.maybe_empty_cache()`；
- lazy/unload 在临时引用释放后保留必要清理；
- 分别检查 T2*、I2* 和 decode 分支；
- 对比三组：

```text
warmup + unconditional empty_cache
warmup + pressure-aware/no empty_cache
no-warmup
```

若第二组稳定消除 Step 1 差距，归因于 allocator/workspace 冷启动，不要继续扩大 warmup step 或修改 scheduler。

### Model offload 的两个清理边界

不要把 encoder 后、transformer 上卡前的清理，与 transformer 下卡后的清理视为同一职责；逐处做消融实验：

- encoder 后强制 `empty_cache()` 可能释放 warmup 留下且正式 DiT 可复用的 allocator/workspace。若 pressure-aware 清理能稳定降低延迟且不增加峰值，可只放宽这一处。
- model offload 下卡后若不释放大块 allocator cache，权重虽已回到 CPU，服务空闲显存仍可能接近满卡；下一请求会在实际分配时触发回收/重试，导致 encoder、上卡和 Step 1 变慢。因此这一边界通常仍需显式释放，但必须由连续请求和空闲显存实测确认。

至少连续运行三次相同请求，分别记录 encoder、Prepare DiT/H2D、Step 1、D2H、VAE、E2E 和请求间空闲显存。首请求变快不能证明方案成立；连续请求变慢或空闲显存长期高占用应判定失败。

## 实验模板

占用 GPU 前先解析最终 config，检查 checkpoint、LoRA/adapter 和输入媒体路径。保留用户脚本的模型、task、config、prompt 和输入，仅按实验需要切换：

```text
--warmup / no --warmup
--return_result_tensor
CUDA_VISIBLE_DEVICES=<free GPU>
```

功能 smoke test 每个模式至少完整运行一次；只有要给出可靠性能结论时，每组才要求至少三轮并保存日志。原配置不可运行时，替代配置/shape 的结果必须单独标为 smoke test。提取：

```bash
rg "Warmup:|Warmup completed|Run Text Encoder|Run Image Encoder|Run VAE Encoder|Run VAE Decoder|step_index:|infer_main cost|Traceback|ERROR" <log>
```

多阶段日志分别分组，不能把 Stage 1 和 Stage 2 的耗时混合平均。

每轮开始前和运行中检查 GPU；出现其他进程时停止自己的实验，待 GPU 空闲后重测。按 stage 统计 Step 1、Step 2、Step 3–最后、Step 6–最后，并分别报告 warmup、自身正式 pipeline、`warmup + 正式 pipeline` 和 no-warmup pipeline。排除频率、温度和保存/IO；相同 seed 还要比较输出哈希/tensor 或合理容差。lazy/no-warmup 若在同一 `load_transformer/offload_manager` 位置失败，归类为基础 lazy/offload 问题。

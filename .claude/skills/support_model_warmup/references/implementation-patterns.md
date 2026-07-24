# LightX2V Warmup Implementation Patterns

仅在需要代码骨架、family 特例或性能诊断时阅读本文件。以当前仓库为准，用 `rg` 重新定位接口，不要机械复制。

## 目录

- [参考入口](#参考入口)
- [没有 warmup 的模型从哪里开始](#没有-warmup-的模型从哪里开始)
- [Runner 骨架](#runner-骨架)
- [Wan 模式](#wan-模式)
- [Qwen-Image 模式](#qwen-image-模式)
- [LTX2 模式](#ltx2-模式)
- [Scheduler clear 合同](#scheduler-clear-合同)
- [Allocator cache 反模式](#allocator-cache-反模式)
- [实验模板](#实验模板)

## 参考入口

| 用途 | 参考位置 |
|---|---|
| 公共 warmup 生命周期、GC | `lightx2v/models/runners/base_runner.py` |
| warmup 开关和跳过条件 | `lightx2v/models/runners/default_runner.py` |
| Wan task/MoE/lazy 模式 | `lightx2v/models/runners/wan/wan_runner.py` |
| Wan distill/MoE 分支 | `lightx2v/models/runners/wan/wan_distill_runner.py` |
| Wan 状态清理 | `lightx2v/models/schedulers/wan/scheduler.py` |
| Qwen T2I/I2I 模式 | `lightx2v/models/runners/qwen_image/qwen_image_runner.py` |
| Qwen RoPE/diffusers 状态清理 | `lightx2v/models/schedulers/qwen_image/scheduler.py` |
| LTX2 多阶段/iterator 模式 | `lightx2v/models/runners/ltx2/ltx2_runner.py` |
| LTX2 latent/sigma 清理 | `lightx2v/models/schedulers/ltx2/scheduler.py` |

## 没有 warmup 的模型从哪里开始

1. 确认 runner 是否继承 `DefaultRunner`，以及哪个具体 `init_modules()` 被公共 hook 包装。
2. 沿正式 `run_pipeline()` 标出 encoder、scheduler prepare、单步循环、阶段转换、decoder 和 `end_run()`。
3. 在共享这些算子图的最小 runner 类上新增 `run_warmup()`；不修改 `infer.py` 或公共 hook。
4. 若该 runner 有不同算子图的子类，让子类显式 opt-in。
5. 先完成 eager 单 task，再在现有 lazy-load 本身可运行的前提下复用加载/卸载方法。

eager 通常只需要 `run_warmup()`、`_run_warmup()` 和 `clear_warmup_state()`；lazy 若没有可复用的正式 cleanup，再增加一个 cleanup 方法。不要为了统一命名搬动正式函数；只有输入准备在多个 shape/task 中重复时才增加 helper。

## Runner 骨架

按目标 runner 调整占位符：

```python
@ProfilingContext4DebugL1("Warmup")
def run_warmup(self):
    if not self.supports_warmup():
        logger.warning(...)
        return

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
- lazy cleanup 必须位于 `finally`。
- `_maybe_freeze_gc()` 必须在 `_run_warmup()` 返回后。
- lazy 模式由公共方法跳过 freeze。

核心循环：

```python
def _run_warmup(self):
    scheduler = self.model.scheduler
    missing = object()
    original_generator = scheduler.generator
    original_input_info = self.input_info
    original_inputs = self.__dict__.get("inputs", missing)

    try:
        for height, width in self.WARMUP_RESOLUTIONS:
            try:
                inputs = self.prepare_warmup_inputs(height, width)
                scheduler.generator = None
                scheduler.prepare(...)

                previous_step = None
                for step_index in self.get_warmup_step_indices(scheduler):
                    if previous_step is not None and step_index != previous_step + 1:
                        scheduler.reset(..., step_index=step_index)
                    scheduler.step_pre(step_index)
                    self.model.infer(inputs)
                    scheduler.step_post()
                    previous_step = step_index

                output = self.run_vae_decoder(...)
                consume_if_iterator(output)
                torch_device_module.synchronize()
            finally:
                self.clear_warmup_state()
    finally:
        scheduler.generator = original_generator
        self.input_info = original_input_info
        if original_inputs is missing:
            self.__dict__.pop("inputs", None)
        else:
            self.inputs = original_inputs
```

显式保存本实现实际修改的 infer steps、sigma、guidance 等其他字段。`prepare/reset` 参数和非连续 step 语义必须来自该 scheduler；若没有 `reset(step_index=...)`，重新执行正式 `prepare()`，不要发明兼容接口。

## Wan 模式

使用场景：

- T2V/I2V/FLF2V；
- Wan2.2 dense/MoE；
- changing resolution；
- distill scheduler。

关键点：

- T2V 只需要文本输入。
- I2V 每个 warmup shape 执行 Image Encoder 和 VAE Encoder。
- FLF2V 为首尾帧分别构造输入，并把两帧同时送入 encoder。
- 普通单模型通常使用 Step 0。
- MoE 选择 high-noise 和 low-noise 各自第一个有效 step。
- 两个代表 step 不连续时 reset scheduler。
- 专用 Wan 子类必须自行声明支持，避免 VACE、audio、animate、self-forcing 等错误继承。

## Qwen-Image 模式

使用场景：

- T2I；
- I2I；
- packed image latent；
- position/RoPE cache。

关键点：

- T2I 文本编码与分辨率无关时可跨 shape 复用。
- I2I 每个 shape 重新运行包含图片的 Text Encoder 和 VAE Encoder。
- 直接执行一个 `step_pre → infer → step_post`；不要用带额外 profiling 的 `run(total_steps=1)`。
- Step 0 输出可直接进入现有 VAE decode。
- 清理 scheduler 中的 packed latent、image ids、timesteps、RoPE、内部 diffusers `_step_index/_begin_index`。
- 清理 model pre-infer 的 request-specific RoPE cache。
- 不要因为正式 I2I shape 与 warmup shape 不同就断言首步偏慢；至少做三轮实测。

## LTX2 模式

使用场景：

- T2AV/I2AV；
- Stage 1 + spatial upsampler + Stage 2；
- video decode iterator；
- video/audio latent 双 scheduler state。

关键点：

- 根据 VAE spatial factor 对齐 Stage 1 像素尺寸。
- upsampler warmup 的输入 shape 是最终目标尺寸；Stage 1 使用除以 upsample scale 后的对齐尺寸。
- Step 0 用于覆盖正式首个 DiT forward。
- 最后一步用于 unpatchify video/audio latent；没有最后一步时，输出不能可靠进入 Stage 2/VAE。
- Stage 2 必须调用现有 upsampler/VAE encoder 的准备路径。
- `video_vae.decode()` 返回 iterator；必须迭代完成。
- scheduler clear 同时释放 video/audio latent、prediction、sigma 和 generator。
- `infer_steps` 会在 Stage 2 改变；每个 warmup shape 开始前恢复 Stage 1 设置，最外层再恢复原值。

### LTX2 lazy/offload 审查

同步检查：

```text
lightx2v/models/networks/ltx2/weights/transformer_weights.py
lightx2v/models/networks/ltx2/infer/offload/transformer_infer.py
lightx2v/common/offload/manager.py
```

验证 CPU/CUDA buffers、disk prefetch、CPU swap 和每次 infer 的 `reset_infer_states()`。若失败在 `load_transformer()`，warmup 尚未进入第一个 shape，不能把外层 `Warmup cost` 当成已经预热 block。

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

## 实验模板

保留用户脚本的模型、task、config、prompt 和输入，仅按实验需要切换：

```text
--warmup / no --warmup
--return_result_tensor
CUDA_VISIBLE_DEVICES=<free GPU>
```

每组至少三轮并保存日志。提取：

```bash
rg "Warmup:|Warmup completed|Run Text Encoder|Run Image Encoder|Run VAE Encoder|Run VAE Decoder|step_index:|infer_main cost|Traceback|ERROR" <log>
```

多阶段日志分别分组，不能把 Stage 1 和 Stage 2 的耗时混合平均。

按 stage 比较 Step 1–5 的三轮趋势，排除并发负载、频率和保存/IO。lazy/no-warmup 若在同一 `load_transformer/offload_manager` 位置失败，归类为基础 lazy/offload 问题。

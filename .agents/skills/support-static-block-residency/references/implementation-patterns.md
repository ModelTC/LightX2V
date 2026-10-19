# 估测、常驻管理与调度

简体中文 | [English](implementation-patterns_en.md)

## 源码与职责

| 入口 | 职责 |
|---|---|
| [estimate_offload_residency.py](../../../../tools/estimate_offload_residency.py) | 单 block case、计时、数量计算、配置与报告输出 |
| [resident.py](../../../../lightx2v/common/offload/resident.py) | `uniform_resident_indices()`、公共配置校验、`ResidentBlocks` 准备与释放 |
| [event_manager.py](../../../../lightx2v/common/offload/event_manager.py) | `EventSlotWeightAsyncStreamManager` 的两个固定 slots、ready/free 事件与 `run_blocks()` |
| [weight_module.py](../../../../lightx2v/common/modules/weight_module.py) | 权重叶子遍历、CPU/设备状态及加载接口 |
| [block_loader.py](../../../../lightx2v/common/offload/block_loader.py) | `OffloadGroup`、连续布局准备与配置校验 |
| [set_config.py](../../../../lightx2v/utils/set_config.py) | `build_startup_config()`，合并与规范化模型运行配置 |

## 独立估测流程

```text
基础 JSON + CLI + 有效 dtype/算子环境
    → build_startup_config()
    → build_<model>_case()
        → load_block()：一个 CPU block + 两个设备 slots
        → 正式算子 + 匹配实际 shape 的模拟输入
    → measure_block()：分别测 d、c
    → estimate_resident_count(N, c, d)
    → 生成配置 + 测量报告
```

`load_block()` 按模型前缀从 `SafetensorsCheckpoint` 取一个 block 的真实权重，沿用正式 block 类、量化方案及运行 dtype 转换，再创建设备 slots。选择连续布局时使用 `prepare_contiguous_group()` 和正式 transfer 路径；默认逐 tensor 搬运。不能用随机权重矩阵、一次裸 memcpy 或通用 GEMM 代替正式 block 来声称等价估测。

模拟的是激活与条件输入。其 shape、dtype、文本长度、位置编码、图文状态等必须反映目标配置。估测只加载必要 block 与模型元数据/tokenizer，不要求加载整个 DiT、文本编码器或 VAE 执行完整推理。

新增模型 case 优先沿用当前返回契约：

| 字段 | 使用方式 |
|---|---|
| `blocks`、`manager` | 公共函数从 CPU 源搬到设备 slot，测量后同步和清理 |
| `compute` | 无参数回调，调用正式单 block 计算，返回 tensor 序列供有限值检查 |
| `reset` | 无参数回调，重置可能被原地修改的输入；排在计时起点之前 |
| `block_bytes` | 当前加载/布局下的 block 权重字节数，用于报告 |
| `grid`、`pixel_size`、`text_tokens` | 记录实际测量的输入规模 |
| `infer`、`pre` | 当前 case 保存的模型对象与输入，真实 block 对照检查也会使用 |
| `text_lengths` | 当前 Qwen case 的正负提示词长度，其他模型按需要记录 |

新增 CLI 模型选择与 case 分派即可复用公共测量；按实际需要调整报告分支，避免新模型误用 Wan 的帧数字段。模型特异性构造见 [模型案例](model-adapters.md)。

## c 与 d 的测量口径

`measure_ms()` 在指定设备 stream 上使用启用计时的 Event：输入准备 → start → action → end → 等待 end。每项执行 `WARMUP=3` 加 `REPEATS=10` 次，只保留后 10 次耗时并取中位数。计时起点之前的输入重置不进入该次样本。

`measure_block()` 先在搬运流上测 `copy_block_to_buffer()`，得到 `d`；同步后，在计算流上测已装载权重的单 block，得到 `c`。计算结果做有限值检查。两组测量独立，不测并发争用，也不运行候选常驻组合或重叠校准。

报告中的 `compute_samples_ms`、`transfer_samples_ms` 保留原始正式样本，`compute_ms`、`transfer_ms` 是中位数。`elapsed_seconds` 是该工具的阶段性总耗时，包含此前的准备工作，不能用来替代单 block 耗时或完整推理时间。

## 数量与均匀分布

设总 block 数为 `N`，常驻数为 `r`。使用单 block 代表各层时，一次完整 block 序列的计算预算是 `N*c`，需要搬运的预算是 `(N-r)*d`。令后者不超过前者：

```text
(N-r)*d <= N*c
r >= N - N*c/d

d <= c：r = 0
d > c ：r = N - floor(N*c/d)
```

`estimate_resident_count()` 要求 `N` 为正整数，`c`、`d` 为有限正值。直接调用现有函数，保留整数取整与边界检查，不另加候选枚举、经验安全系数或自动显存裁减改变当前策略。

这是总预算近似下的最小整数，不是实际调度达到完全隐藏的充分条件。有限预取窗口、首尾开销、层间差异及并发带宽争用可能留下等待。实际收益由推理对比确认，验证结果不自动反馈成另一套选数算法。

`uniform_resident_indices(N, r)` 使用零起始索引：

```text
r = 0：()
r > 0：tuple(floor(k*N/r) for k in range(r))
```

例如 `N=8, r=2` 得到 `(0, 4)`。数量恰为 `r`，索引唯一且有序，按环状计的相邻间隔相差至多 1。均匀指选中的常驻层分布；不改变模型执行顺序，也不把常驻层集中到前 `r` 层。

## 配置与报告

基础配置保留模型、算子及其他 offload 设置。`cpu_offload_block_contiguous` 原样传递给估测和生成配置，未指定时输出 `false`；报告中的 `layout` 由该布尔值推导。工具建立运行配置时设定 `cpu_offload=true`、`offload_granularity="block"`、`offload_resident_blocks=0`，测量后将结果写入 `offload_resident_blocks`。输入中遗留的独立事件开关会从输出配置中移除。Wan 同时保存规范化后的 `num_frames`。

显式 `--output-config` 指定输出；省略时在原配置旁生成 `<stem>.resident.json`。报告使用输出文件名的 `.profile.json` 后缀。`write_json()` 在目标目录创建临时文件并通过 `os.replace()` 替换单个 JSON，防止半写入文件。

报告包含 `N` 对应的 `num_blocks`、`resident_blocks`、`streamed_blocks`、`resident_indices`、输入规模、设备、dtype、量化和布局信息；推理使用配置中的数量重新计算索引，不读取 profile 来调度。

对于同构 block，报告的 `resident_weight_bytes` 近似为 `r*block_bytes`，`execution_buffer_bytes` 为 `2*block_bytes`。这是权重预算，不含激活、非 block 权重、算子 workspace、其他组件及 allocator 开销。当前即使 `r=N` 也不承诺自动省去两个 slots；CPU 主权重始终保留。

## 推理调度与生命周期

Wan、Qwen Image、Qwen-Image-2.1 和 MiniMax-H3 在配置包含 `offload_resident_blocks` 时选择事件路径。值为 0 时仍使用事件双缓冲，值大于 0 时额外常驻对应数量的 block；省略该字段则保留 legacy 路径。`r=0` 的事件 baseline 与 legacy 是两个可用于不同目的的对照。

`ResidentBlocks.prepare()` 幂等地计算索引，保存权重叶子的原状态，调用选中 block 的 `to_cuda_async()`，并记录常驻数、搬运数和索引。它依赖 `named_weight_leaves()` 与 `pin_*` CPU 主权重等契约。新模型若不满足这些接口，先补齐对应适配，不能只实例化该类宣称完成接入。

`run_blocks()` 的现有顺序：

1. 计算流和搬运流等待 caller stream，包含常驻准备及输入准备的依赖。
2. 按逻辑顺序筛出非常驻层，最多先预取两个到各自 slot。
3. 顺序遍历所有 block：常驻层直接计算；其他层等待 slot ready，计算后记录 free，再在同一 slot 排入下一个非常驻层。
4. slot 复制等待上一消费者的 free 事件；计算等待复制的 ready 事件。slot 编号只在非常驻层后轮换。
5. caller stream 等待计算流，再接收输出。异常时同步计算/搬运流并重置事件记账。

模型回调保持正式计算语义，包括逻辑层号和状态顺序。Qwen 的文本与图像是两路状态，不能丢掉其中一路。跨 stream 使用的 caller 输入需要正确 `record_stream()` 或等价的存活保证；交回 caller 的输出同样要处理 allocator 生命周期。stream 等待只建立执行依赖，不能代替存储生命周期管理。

常驻权重在首次需要时准备，跨步骤及 CFG 分支复用。runner 在去噪结束、warmup 结束或失败时通过 `finally` 调用 `release_resident_blocks()`，在 VAE 解码前释放额外设备权重。`ResidentBlocks.release()` 等待设备操作完成、恢复原属性引用；模型释放入口随后调用 `reset_slots()`。重置记账本身不等待设备完成，不能单独用它释放正在使用的资源。

下一请求复用 CPU 主权重重新准备，验证 release → prepare → infer。不要为了释放设备权重删除 CPU 主副本，或依赖 GC 的偶然时机避免读写冲突。

## 当前配置边界

公共 `validate_residency_config()` 要求 CPU block offload、`AI_DEVICE` 为 `cuda` 或 `npu`，以及单设备 eager、固定权重/分辨率、`NoCaching`。当前拒绝 lazy loading、共享 CPU 权重、并行、LoRA/adapter、变分辨率、compile 和 CUDA Graph 等组合；具体字段以函数为准。

模型校验另负责模型名、任务和 conditioning。新模型保留已有配置的行为，仅在选择事件/常驻路径时做相应校验。当前限制描述实现范围，不代表原理上永远无法组合；扩展兼容性须有对应实现和验证，不能绕过校验即可宣称支持。

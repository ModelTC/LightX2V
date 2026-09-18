# 验证与故障定位

简体中文 | [English](validation-and-debugging_en.md)

按改动选择验证，区分 CPU 契约测试、CUDA／多进程机制验证和目标模型验收。只修改文档时检查引用与指令一致性即可；真实模型接入不能只做静态检查。

## 启动脚本与配置验收

在 `scripts/<model>/offload/` 核对统一共享启动脚本及中英文 README。当前脚本显式传入 `--shared_cpu_weight_scope host`；验证 NUMA 时改为 `numa`，或直接执行对应完整 Python 命令。`SHARED_CPU_WEIGHT_SCOPE=numa bash script.sh` 不会覆盖当前脚本。追踪 `--config_json` 和 CLI 合并结果，确认有效 scope；共享开关必须进入实际 loader，而不只是被 JSON 解析。

针对交付文件执行 `bash -n` 和 JSON 解析。比较两种模式的有效配置除 scope 外的推理字段，并检查有效 dtype、模型路径、输入、seed、输出路径、显卡列表和进程数。CLI scope 应覆盖旧 JSON 的同名值且不进入请求参数；省略 CLI 时保留 JSON／内部默认。验证需要改路径或参数时可用临时副本，不把临时配置生成逻辑加回正式脚本。`git diff --check` 检查改动格式；新建但未追踪的文件也应单独检查，不依赖 git diff 自动覆盖它们。

启动前核对：

- 当前 checkpoint、必要的 AdaLN cache／量化文件和输入媒体确实存在；H3 cache 按模型 variant、权重、步数和 flow shifts 匹配，Hunyuan 还需上游代码及兼容的 FlashInfer。
- `torchrun` world size 与有效配置中的 `tensor_p_size * cfg_p_size * seq_p_size` 匹配，可见设备与 local rank 映射有效。
- Linux SysV／CUDA 注册条件和实际 GPU NUMA 拓扑满足目标模式；NUMA strict 的失败不能通过静默改 scope 或关闭 strict 掩盖。
- 按当前脚本约定从仓库根目录调用。不要仅检查 `model_path`；JSON 内的其他相对资产路径也要正确解析。只有任务明确要求任意工作目录启动时，才实现并验证该能力。

以相同输入分别执行真实 host、NUMA 入口，记录命令、有效配置、代码版本、资产指纹、硬件拓扑、日志和输出位置。若只在一个 NUMA 域运行成功，应注明没有覆盖多个 NUMA replica 的场景。

## 选择聚焦测试

先发现当前仓库实际存在的测试，不把历史提交中的临时测试名当作现成命令：

```bash
rg --files test_cases | rg '(shared|offload|hunyuan|qwen|wan|h3)'
```

当前仓库没有原先记录的 `test_shared_pinned_arena.py`、`test_shared_weight_map.py` 等共享专项测试；现有 shell 用例也不自动覆盖共享机制。若任务需要新增或临时验证，按以下维度选择，而非依赖固定文件名：

| 改动范围 | 聚焦验证 |
|---|---|
| arena／manifest | view 布局、SysV 跨进程物理共享、注册区域、回滚及关闭失败状态 |
| replica planner | host／NUMA／auto 分组、非连续 rank／device 顺序等拓扑边界 |
| coordinator | Store 状态交换、阶段不匹配、延迟／超时、失败传播及清理；CPU 等待期间不发起 NCCL collective |
| shared weight map | 共享多消费者、owner 保留、算子零拷贝采用与私有分支行为 |
| Wan adapter | FP8 dtype、scale 舍入、schema／内容签名与配置约束 |
| Qwen adapter | 跨 shard manifest、scope 参数传递、BF16 解析／转置约束 |
| Hunyuan adapter／slots | storage-TP slices、SP／CFG 复用、MoE pack 布局、逻辑 KV 层号和异构 block 族 |
| H3 共享组件 | 组件选择、源／运行 dtype、schema 与签名；AdaLN variant 匹配 |
| H3 原生模块 offload | 参数别名、nonpersistent buffer、恢复原 storage、slot 复制、tile 与释放重建 |

改动公共 arena／coordinator 则增加相应公共测试和受影响模型测试。按实际变更运行 Python 编译、仓库已有 lint 等检查，不为可逆的文档或脚本命名变化编写复刻实现的测试。

新 adapter 的有意义测试应覆盖：小型合成 checkpoint 的 header/index 与内容变化、目标 dtype／layout、与基线一致的转换、私有和共享分界、scope 参数传递，以及绑定后 storage 身份。增加失败检查应对应真实语义约束，不是穷举所有内部参数类型。

区分 CPU fake runtime 测试与需要 CUDA 的测试，后者可能 skip。报告 passed／failed／skipped 和原因；mock 通过不能证明真实 `cudaHostRegister`、设备 event 或 DMA 正确。`--help` 只验证入口导入／参数解析，不等于缓存生成或模型推理通过；默认 dtype 变更也需要对应精度的运行证据。

## 证明权重实际共享

机制证据至少覆盖：

1. 相同 replica 内的进程挂接相同 SysV segment，跨 host／IPC／NUMA 组的分配符合计划。段号须连同 host 和 IPC namespace 解释，不能跨机器仅比较整数 `shmid`。
2. 每组只有 leader 执行共享 payload 的 populate；follower 不额外构造完整私有共享权重。摘要计算 I/O 与 payload 物化分开统计。
3. 每个进程的算子 CPU view 地址、dtype、shape、stride 与本地 manifest 一致，注册／pinned 检查通过；不是仅检查 loader 返回值。
4. 私有 non-block 权重与动态状态仍独立，共享源在推理后保持不变。需要测试跨进程写入可见性时仅使用合成测试段，不修改真实模型共享权重。

预期的共享 payload 物理容量近似为：

```text
sum(每个实际 replica 的 manifest.nbytes)
```

按组件分别计算，并另计私有权重、metadata、注册开销、checkpoint 映射与加载临时内存。NUMA 模式通常有多个完整 replica，不能用 host 模式的一份权重大小验收它。

不要把各 rank 的 RSS 相加当作物理占用：同一共享页可能在每个进程的 RSS 中重复出现。结合 `/proc/<pid>/smaps_rollup` 的 PSS／Private 信息、唯一共享段与 arena 大小、系统物理内存变化判断；在有权限时用 `/proc/<pid>/numa_maps` 或 `numastat -p` 核对页分布。PSS 也不是单个权重组件的精确大小，须控制其他进程、文件映射和采样时机。

记录初始化峰值和稳定内存，避免最终占用下降却掩盖“所有 rank 先完整加载”的高峰。

## 正确性、连续请求与失败路径

优先对比相同并行规模的私有 block、共享 host、共享 NUMA。保持 checkpoint、task、输入、shape、步数、seed、精度与算子一致。单卡与 SP 的结果还会受到计算顺序影响，应单独解释。

基线可复现且计算路径相同时可使用 tensor／hash 比较；非确定性算子采用有依据的容差，记录最大／平均误差等指标。视频容器的字节 hash 可能包含编码或元数据差异，应优先比较编码前 tensor 或解码后帧／音频数据。只凭视觉相近不能证明权重转换正确。

不一致时先比较实际采用的 CPU 权重、量化 scale、单 block 和首个 denoise step，再追踪最终输出，避免扩散多步放大误差掩盖根因。

生命周期验证按受影响路径选择：

- 同一模型实例连续至少两个请求，确认 slot 状态、首块准备、源地址与输出有效。
- 有 GPU 缓冲阶段释放时，验证 release → rebuild → infer，CPU source 仍来自原 arena。
- VAE 多 tile、图像条件 encode 和文本／视觉分支，覆盖本次任务实际经过的路径。
- 在可控测试中模拟局部 preflight、populate、register、binding 或计算失败，确认错误传播与资源清理；不得通过关闭仍在使用的 arena 获得“无泄漏”。

GPU 峰值不能简单等同两个 block 的大小：另计非 block 权重、激活、attention workspace、其他组件与 allocator 保留内存。若 slot 释放后 allocated 降而 reserved 不降，先区分对象残留与缓存分配器行为。

## 性能证据

只有任务要求性能结论时进行有代表性的重复测量。保持同一组空闲设备、资产、算子和并行度，保留每轮结果；存在缓存或 JIT 时注明冷／热状态，不用单轮最好值下结论。

分别记录：

- checkpoint 检查／摘要／加载时间；
- SysV 创建、NUMA 绑定、CUDA 注册、populate／attach 时间；
- 首次请求与首次 denoise step；
- 稳定 denoise step、文本编码、VAE 和端到端时间；
- CPU 初始化峰值／稳定物理占用，以及 GPU allocated／reserved 峰值。

host 减少 replica 数，NUMA 可能改善本地访存及 H2D 路径；具体收益取决于拓扑与带宽。共享不自动减少 H2D 字节数，也不保证加速。需要归因拷贝／计算重叠时使用设备 timeline；只统计 Python enqueue 时间会漏掉异步设备耗时。

任务包含 warmup／compile 时使用对应 skill 检查真实算子和 shape 覆盖，并将准备成本与稳定成本分别报告；不为一次 offload 功能验收强制引入编译系统。

## 按证据定位问题

| 现象 | 先检查的证据 | 常见处理方向 |
|---|---|---|
| 开了共享但 CPU 物理占用仍接近多份 | 算子最终 pointer、Private／PSS、加载峰值 | 排查消费路径 clone／cast／pin、完整私有预加载与未释放临时字典 |
| pinned 或注册失败 | tensor-aware 注册区域、CUDA 错误、本进程映射、系统限制 | 修复真实注册问题；共享路径不能退回私有 pin 冒充成功 |
| NUMA 模式失败 | GPU PCI 与 NUMA 发现、容器可见拓扑、mbind 错误 | 区分未知拓扑与绑定权限／策略失败，不静默改 host |
| rank 卡在初始化 | 最后一次 Store 阶段、组件顺序、各 rank preflight、缺失 rank 与等待期限 | 对齐状态交换顺序并检查 CPU 加载进度；使用共享协调超时，不以增大 NCCL 超时替代定位 |
| 输出漂移 | 实际 dtype、scale 舍入、转置 stride、kernel 和 seed | 对齐加载语义，再追踪首个数值分歧 |
| 随机错误或第二次请求失败 | ready/free/completion 记录、slot 覆盖时机、owner 是否仍有效 | 修复跨 stream／请求依赖和释放顺序 |
| host／NUMA 实际效果相同 | 有效配置 scope、参与 GPU NUMA 节点、replica 日志 | 确认 scope 切换已写入实际运行配置；单 NUMA 域可能合理地产生相同 replica 数 |
| 共享后速度没有提升 | 加载／注册成本、H2D 字节、稳定 step 与拓扑 | 分别报告内存收益和速度；不要把共享等同加速 |

每次修复后重跑能覆盖根因的检查和受影响路径；已通过且无新风险的检查不需要反复扩大。

## 最终证据记录

交付说明填写实际值，未运行项写明原因：

| 模式 | 脚本／有效配置 | task／组件／并行规模 | replica 与共享验证 | 正确性／连续请求 | CPU／GPU 内存 | 验证状态 |
|---|---|---|---|---|---|---|
| 私有 block 基线 | 实际路径 | 实际条件 | 不适用共享项 | 实测结果 | 实测结果 | 已验证／未验证 |
| host | 实际路径 | 与基线对齐 | 实测结果 | 实测结果 | 实测结果 | 已验证／未验证 |
| NUMA | 实际路径 | 与基线对齐 | 实测结果 | 实测结果 | 实测结果 | 已验证／未验证 |

区分代码／脚本交付完成、功能 smoke test 通过、目标路径验收通过。小 shape、替代资产、单 rank 或 mock 的成功只覆盖对应范围。历史报告可引用为背景，但不能替代本次证据。

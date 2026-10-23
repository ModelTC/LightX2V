# 算子 Benchmark Agent 指南

代码负责执行、计算、校验与事实表；Agent负责选择、诊断、补证和解释；产品质量取舍由开发者决定。CLI、schema和指标查README，不另维护一份。机器路径、设备顺序、输入文件和一次性实验记录留本地。

同一边界适用于所有算子：模型/请求侧的捕获和适配由Agent组织，生产调用、固定输入计时、autotune与精确事实留工具；已承诺支持的稳定adapter及必要验证随Git交付，一次性实验材料不充当正式依赖。

## 一次任务怎样推进

1. 明确硬件、算子类型、输入/精度来源和目标。支持模型shape、理论sweep、observed backend审计和SP单算子调优；理论数据与真实模型数据分别说明。
2. 转成canonical suite并inspect。模型适配留模型侧；缺call_count不阻塞逐shape比较，但不能据此推模型加权收益。
3. 在目标设备probe，核对生产registry、未默认import实现和dispatch。按硬件、依赖、输入筛选；默认比较目标类型全部eligible候选，有意缩小范围时说明未测项；例如“只跑SLA全部实现”，从当前catalog选`sla.`候选再展开显式列表，不向CLI传前缀通配符。
4. 新目录运行，同进程预热后用event-list计时；默认参数见[README](README.md#性能测量与实际输入)。普通warmup不替代库autotune。续跑须通过工具的输入、参数、环境身份校验。
5. 从raw/report看延迟与波动、错误阶段和覆盖缺口。`measured_winner`只是本轮已测稳定项最快，不是采用决定。需要解释时选择最小诊断，不默认重跑全部矩阵。
6. 给当前硬件/输入/精度下的建议及依据。质量未获授权时报告性能排序和差异；不把strict_equal或完整覆盖自动写成产品质量批准。
7. 保存suite、命令、raw、报告、必要tensor与退出状态。正式工具产物、Agent推断、开发者决定分别说明，不建立额外任务控制面；恢复先核进程和现有产物再重试。

用户明确的信息不重复问。只有真实输入、必要模型语义、资源或质量取舍无法推断时提阻塞；本地可完成的事项继续。功能/计时机制未变时复用有明确源码身份的证据，不为小改动重复GPU矩阵。只使用获准资源，不干扰他人；受干扰计时作废后有界重测。

资源观测复用 [gpu_watch](../runtime/README.md)，任务期限/收尾/重试由Agent按实际环境落实。正式汇总须同时核命令成功、资源check通过和本人任务收尾；受扰原始数据保留但不混入正常计时。

## 候选覆盖与补齐

- Dense、sparse和SP是独立类型，不混合排名。Sparse用`算法.实现.入口`命名；回放、稀疏率和硬件是输入/条件，不放进后端名。相同kernel不代表包含路由/布局的生产路径相同。
- 每种类型都对账完整选项，报告中的覆盖表保留纳入、硬件/依赖/输入拒绝、执行失败、未测与未接入。依赖缺失不是性能落选；不以“没adapter”作为永久拒绝理由。
- 命中gap先追真实生产入口及参数/输出合同，能复用便补adapter和所需验证；无法补时写明具体外部条件及恢复动作。新registry项由工具暴露，未登记dispatch由Agent查源码；工具不声称扫描任意Python分支。
- 稀疏同时核对attention、mask generator、sparse operator；卷积分别核对conv2d/conv3d registry与native分发；MoE核对权重/激活/路由及shard布局。不要为缺失元数据猜生产参数。
- `not validated`表示已有实现尚缺准入证据。与本请求相关且无外部阻塞时主动验证，不将其当永久排除；无须穷举未来所有组合。

工具从实际registry与adapter描述对账；类型提示不是候选白名单，未接入不等于生产不支持。已知延迟注册模块显式import，导入失败保留`unresolved_import`缺口和实际异常。`eligible`只是预查通过，不证明kernel可执行；目录指纹只标识本次已发现集合。

换模型/硬件、backend或依赖升级、以及续跑前，核对当前源码/依赖身份和实际生产调用，重核精度、输入与架构假设，并检查未登记或延迟dispatch。同名实现变化不保证改变catalog指纹，指纹相同不能证明源码未变。变更影响输入、执行或计时合同时新开实验；确认无影响才记录依据复用已有证据，不静默混用旧raw。

缺口的输入准备按当前生产实现核实，以下仅是诊断方向，不是永久拒绝规则：

| 入口 | 优先核实的输入或语义 |
|---|---|
| paged FA3 / MPS chunk / mixed-precision Sage2 | paged KV、page table/cache lengths；query chunk；实际混合精度 |
| draft / flex / nbhd / radial / SVG及其mask | 目标block/调度、frame/context/grid或reference布局，不能只用QKV猜 |
| RainFusion / SVG2 / SOL | step/cache生命周期与grid；聚类状态/变长块；tau/threshold及实际selector路径 |
| native FP8卷积 | 权重、scale、累加类型和Conv2d单深度转换的实际调用 |

General新组合复用通用prepare，按[descriptor合同](README.md#backend)对账子项；接线存在不等于组合可执行。缺口相关且条件允许时主动补接线/验证；稳定实现随Git，具体缺失输入和恢复动作写进既定覆盖表，不把本地脚本当正式能力。

稀疏adapter不统一猜head数、序列长度或head_dim支持范围，先尝试实际prepare/预检并保留错误阶段、类型及原因；仍校验输入文件/shape身份，并拒绝当前prepare会忽略的causal语义。具体kernel已有可核实的寻址限制时，在实际consumer入口校验，不能依赖shape/finite检查发现错读；普通SLA Triton要求同shape，AR允许异长但要求同B/H/D。错误后不悄悄fallback或修改输入；illegal memory access/device assert等CUDA上下文错误先终止本进程，核日志/进程后有界重启，不能在已损坏上下文中续测。

## 中断与原始记录恢复

- 新运行先核进程/设备和既有产物；依[工具恢复边界](README.md)查catalog和失败raw。不得在同一损坏上下文续测；核失败原因后用新进程有界恢复。未知CUDA异常也先判断上下文可信度，不能把异常捕获当作安全证明。
- `--append`跳过所有完整记录，包括错误行，不自动重试失败项。保留原始错误；需重测时另存新目录，按相同suite/参数及输入环境身份检查后只合并无重复键的有效记录。fatal行在报告中是measurement_error，不能记作合法排除或完整覆盖；历史进程中fatal之后的计时先作废，无法确定来源时不复用。
- raw解析失败先定位行与进程状态。只有确认最后一行被截断、其前各非空行均为完整JSON对象时，才可在新恢复目录复制完整前缀；保留原raw、被排除尾部原始字节、来源路径/哈希和恢复说明，再按原参数/身份append。完整错误行、受扰行不能借恢复删除；中间损坏、非法UTF-8或无法确认截断原因时停止并定位，不做自动修补。格式完整但末尾没有换行的JSON对象也不是截断行。

## 输入、计算量和数值

- 计算量来自共用数学和core/backend事实，不靠kernel名称或profiler的record_shapes。统计默认关闭，独立eager诊断；不在正式计时、compile或CUDA graph内部采集。
- 扩展统计先用现有基类/共同公式；动态稀疏在拥有最终mask/ranges的生产位置提供必要事实；共用sparse operator自行调用`record_sparse_attn`，统一查`enabled("attention")`，别名表不是接入清单；实际区域随attention统计自动取得，额外MoE分布与输入导出单独控制。固定top-k可用shape与sparsity，数据相关选区必须读实际区域；尾块/调度padding未知时留缺口。
- 选代表真实输入，再保持输入、seed、精度、backend及配置一致。Sparse正式比较只用真实RoPE后QKV；模型侧记录分支/step/block和采集位置。默认单卡稀疏率0.8，不自动展开参数搜索；内容相关路由有波动时另作最小采样，不能将一次计算量视为所有调用恒定。
- 输入分布敏感的MoE使用下面的任务侧输入准备流程。专家直方图仍是合成性能输入，不能还原真实输出；不同模型的量化/shard/专家结构不强制归一成公共文件协议。
- 单卡bench不内置FP32参考或比较输出归档。新backend接入、算法或实现变化时，主动复用项目/库测试或独立小脚本做数值验证；保存输入、实际backend、容差来源和结论。不要把shape/finite通过写成数值正确，也不为普通性能任务逐case重算参考。
- 稀疏实际量从同次生产区域的共用统计读取，与数学参考分开；正式计时关闭统计。实际/稠密等价效率分别说明，后者可超过100%，不是物理利用率。
- 性能输入回放只固定实际负载，不证明输出数值正确；需要质量结论时单独验证，非零差异交开发者决定。不要为凑通用功能把一次性验证脚本纳入bench核心。
- 测试优先覆盖共用数学oracle、真实producer→consumer和异常/默认关闭；不要为每个backend复制大量mock。import/probe、源码审查、CPU/mock和目标设备真实执行分别说明，不能相互替代。
- 增加或改变生产接线时，核真实符号/签名，并在获准设备用相关真实输入确认实际backend路径、输出和fallback；数值/性能按变化风险补验。结论注明模型/输入、设备、精度、backend/依赖版本及已验证层级；缺资源时保留“native未验证”和下一步，不将CPU通过写成支持，也不将未测写成不支持。本轮具体未测项和运行收据留任务验收记录，不反复复制进长期指南。

## MoE真实输入准备

1. 沿模型调用追到实际MoE consumer，确定目标step/block、并行rank、路由在何处完成；记录模型/权重、请求、捕获位置及缩层或裁剪的代表性边界。优先复用`WorkloadStats(on_moe_inputs=...)`回调；当前backend未提供所需表示时，在任务脚本沿真实入口取数据，不为本次实验扩张公共导出框架。
2. 保存实际activation、专家IDs/scales与实际权重/scale/bias表示；检查gate顺序、dtype、布局/stride、专家shape、TP/EP分片、token dropping/padding和共享专家。调用会复用或改写buffer时，保留最后安全时点的值。只保存计数/直方图或使用同seed随机数都不能代替真实tensor。
3. 保存/加载脚本和格式由当前任务拥有，放本地实验目录；文件可分开保存，避免每个backend反复导出巨量相同权重。使用适合该模型的checkpoint引用或张量文件，固定文件内容身份、数据布局及来源；不能仅因字段写了`model_inference`就当真实证据。
4. 按[MoE输入接口](README.md#计算量统计与真实moe回放)复用plugin及适用的内置prepare，加载/搬运和身份校验放计时外。逐项核实际shape/dtype/路由/权重与本次case及production API一致；任务adapter名称与已有项区别，保留真实production backend/精度/依赖，显式选择，不将合成测例混入真实输入排名。
5. 输入身份必须覆盖全部实际tensor及被引用的权重/量化元数据，不能只hash一个不随tensor变化的空描述。布局无法复用内置factory时直接准备真实production调用；不要解量化或丢分片语义来伪造适配。
6. 所有候选固定同一组实际逻辑输入；必要packing/布局转换按真实backend实施并记录，放prepare内还是正式调用内以生产开销语义裁决。公共bench保留生产调用、固定输入的event-list计时及已接入FlashInfer autotune；复用父prepare即可复用其同bucket调优/正式调用，禁止在计时中重新加载或搜索。不同模型的额外调优在任务侧按真实库API处理。
7. 计算量继续取公共公式和实际生产入口事实；固定top-k同构MoE总量不靠专家分布猜，负载差异仍需真实IDs。出现不等大专家、丢token或未知量化/shard合同，先报告精确缺口，不能套现有同构公式。
8. 新输入接线先做一次最小的真实producer→加载→production consumer数值核对，保持必要独立参考；记录容差及输出差异，不能把finite/shape检查当正确性。性能与质量分别运行，常规性能任务不重复全套参考。
9. 保存输入/插件源码、依赖、suite、命令和退出状态。重启先核产物/进程与输入身份再续跑；文件变化必须改身份并开新实验，不静默混用旧raw。一次性实验脚本及tensor/raw等任务数据留本地；模型特有不等于一次性。对已承诺支持的模型，稳定可复用的adapter和必要验证应放在相应模型/backend侧随Git交付，不能依赖未提交本地插件、个人路径或旧会话；多个模型才复用的公共框架仍须有实际消费者依据。

## 何时运行计算量回归

Agent主动按改动影响选择验证，不等待开发者逐次提醒；测试时机、环境、命令和收据在本节维护，不新增自动触发的CI。

- 改公式、会话/采样或producer→consumer接线后，先运行受影响的测试文件或用例；跨职责的代码改动在交付前集中执行三组回归，首次使用新测试环境也应运行三组。
- 仅改注释/文档且执行语义未变，可复用有明确源码身份的既有结果，记录未重测依据；不因一次普通bench任务或每次小改动默认重跑全部测试。
- 使用独立CPU环境或已核实依赖相同的环境，保存源码/依赖身份、命令、日志及退出状态；需要结构化收据时导出JUnit到本次新产物目录。失败按原因修复后定向重验，不把跳过当通过。
- CPU回归只证明数学和Python接线。native执行、数值或性能有实质变化时，另选必要的目标设备验证，遵守资源授权；不自动扩展GPU矩阵。

先准备独立的Linux/Python 3.12环境，避免改动现有推理环境。以下命令从源码根目录执行：

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r tests/requirements-compute-stats.txt
CUDA_VISIBLE_DEVICES= PLATFORM=cuda SKIP_PLATFORM_CHECK=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q tests/test_compute_stats.py tests/test_core_op_stats.py tests/test_sparse_stats.py
```

已有同依赖环境可直接执行pytest。需要机器可读结果时，追加`--junitxml=/path/to/new-results/compute-stats.xml`，将路径换成本次新产物目录；同时保留日志、退出状态与源码/依赖身份。`PLATFORM=cuda`和`SKIP_PLATFORM_CHECK=1`只允许导入现有平台模块，实际用CPU wheel执行，不代表CUDA后端已通过。

CPU wheel版本配对见[PyTorch官方安装表](https://pytorch.org/get-started/previous-versions/#v2110)。纯公式使用独立token oracle，接线测试保留实际重排/LUT/ranges、采样/异常和并发反例；默认GPU不可用时仍运行这些CPU用例，不以跳过代替通过。原生CUDA数值/性能仍另需目标环境验证。

## 何时运行bench回归

改变候选、回放、执行或报告逻辑后，按影响选择`test_operator_bench.py`、`test_moe_replay.py`、`test_bench_execution.py`；交付前集中运行这三组。SP显式比较由其所属SP功能单独验证，计算量采样/会话保障复用前节回归，不在bench重复建设。无自动CI，不因普通运行重跑全矩阵；CPU替身只证明Python合同，native路径改变另需实际后端验证。

## 硬件峰值与调优

更换硬件或peak不匹配时，Agent主动查厂商规格/架构文档，按实际SKU、SM、计算/累加精度建立显式peak文件。记录来源、dense与结构化稀疏倍率、单位；工具只校验并消费，不内置联网或旧卡常量。缺可信peak先报告绝对时间和计算效率。

FP8等不同累加类型峰值不同，应提供variants且raw有实际accum_dtype。未知精度不猜；输入精度名义peak不等于混合实现的物理指令效率。互联peak另按GPU/world size/拓扑及通信模式匹配。

FlashInfer MoE由adapter在计时外调当前token点，正式调用以False上下文固定同bucket；普通warmup不能代替搜索。检查多shape往返仍命中正确缓存，不把旧fallback成绩改称已调优。GEMM动态量化与稀疏路由的生产开销属于各自完整算子时间。

## SP单算子


- SP 使用独立 `sp_bench.py`、SP suite 和 `torchrun`；进程数必须等于 `sp_size`。
- L1 production 端到端延迟是候选排序依据；L2 layout/communication 与 L3 collective 只用于解释 L1，不参与 winner 排名。
- L2/L3 日常入口应优先读取 L1 `report.json` 的逐 case 正式 winner，不要求用户重新描述长 candidate ID；结果同时保存机器可读 JSON 和面向人的 `diagnostic_report.md`。
- Ulysses `head_parallel=true` 时默认枚举 `1..local_heads` 的全部 `head_parallel_group_size`，并与 bulk 路径共同参与 L1 排名。group size 不必整除 local head 数；候选身份、推荐配置和 shape 描述必须记录尾组实际 head 数。
- Aux 诊断必须遵循 production 通信语义：Ulysses 的 replicated aux Q/K/V 绕过 main QKV A2A，aux output 执行 all-gather；Ring 的 replicated aux Q/K/V 绕过 main K/V rotation。L3 必须拆分 main/aux 实际 payload，并把未通信的 aux 明确标为 bypass，不能用全量 dense tensor 估算替代。
- L3 分量必须独立计时并分别报告 latency、Bus bandwidth、spread 和 peak efficiency，禁止按整体延迟或总字节比例反推。混合路径的整体效率与分量效率同时保留；`all_to_all`、`all_gather`、`ring_p2p` 分别匹配自己的 peak profile。
- Ulysses 与 Ring 只在共同支持且已验证的 attention 语义上公平比较。Ring 的 GQA 必须排除；H100 当前 Ulysses/Ring aux 合法矩阵已验证 BF16/FP16、无量化/FP8 communication 和 fusion 开关，FP4 与新增 dense leaf 在专项验证完成前记录为 `not validated`，不能据此宣称 production 不支持。
- `sp_bench` 已支持 dense leaf 与真实 QKV replay 驱动的 sparse adapter。Sparse 候选必须使用独立 `--sparse-backend` 合同，不能把 sparse backend 冒充 `--dense-backend`。
- Sparse grouped Ulysses 必须使用同一 replay、同一 sparse leaf 和通信配置的 bulk Ulysses 做 reference 验证；bulk 只作为 reference，不能作为被测候选与自身比较。报告必须保留 `reference_candidate`，且不得把该结果解释为 sparse leaf 对 dense attention 的精度证明。
- Sparse Ring 只有在 backend 满足分块输出、LSE 和跨 block 合并合同时才进入候选；否则记录 `not_supported` 或 `not_applicable`，不阻塞 Ulysses。
- Sparse SP 候选必须从当前 backend catalog 派生。已知兼容 leaf 应进入 `SPARSE_SP_BACKENDS`；新出现的 eligible sparse adapter 若未完成 SP 合同分类，必须进入 `unmapped_sparse_sp_backends` 并阻断完整推荐，不能静默过滤。
- 互联 peak 必须匹配 GPU、world size 和 topology fingerprint；缺失时仍报告延迟和带宽，但不发布通信效率。
- NCCL `ALGO`、`PROTO` 和 channel 数只作为运行环境身份记录，不进入自动 sweep；需要比较时由用户显式设置环境并使用独立输出目录。
- 当前算法范围只有 Ulysses 与 Ring，不加入 `kv_all_gather` 第三路径。
- 每次新模型、新硬件或新需求开始时，必须主动审计相关候选的 exclusions，不能把 `not validated` 当作普通跳过项。这里的含义是“实现可能存在，但尚无足够证据进入推荐矩阵”。
- 只要 `not validated` 组合会影响本次候选完整性，且预计 3–5 轮工作内可验证、没有外部阻塞，Agent 必须立即登记显式 correctness tolerance，使用 `sp_bench validate-reference` 补 dense reference，再同步更新准入代码、测试和文档；不得等待用户再次提醒。
- “本次矩阵完整”指当前请求涉及的模型 shape、precision、硬件依赖和 backend 选项均有明确结论，不要求提前穷举未来所有交叉组合。结束任务时，不得静默遗留与本次推荐相关的 `not validated` exclusion。
- 只有缺少目标硬件、必要依赖、真实输入或模型语义等外部条件时才允许延期。延期项必须记录未完成内容、阻塞原因、触发补齐的条件和下一步动作；条件在后续任务中满足时，Agent 应主动恢复验证。
- Sparse SP 必须使用真实 replay，但采集拓扑不得与目标 SP 度数绑定。Artifact 必须描述原始 token 顺序下的全局完整、RoPE 后 Q/K/V；采集时可以使用单卡、模型并行或 sequence shards，benchmark 再按 production 模型语义切成用户指定的 SP 度数。
- Sparse L2/L3 必须复用与 L1 相同的 replay 输入；禁止因诊断方便而退化为随机 Q/K/V。
- 对“复制最大 conditioner 前缀、剩余 main 序列不 padding 切分”的模型，使用 `sp_bench derive-replay-suites`从同一完整 replay 派生每个目标 SP 的 suite；`conditioner_tokens` 必须来自模型语义，不能根据整除性猜测。
- 用户未指定 SP 范围时默认搜索 `SP=2/4/8`，也接受指定卡数。每个 SP 度数必须独立派生 replicated aux 与 sharded main 边界、验证 head/sequence 合同并发布各自 winner；不得用一个 SP 的局部分片冒充完整输入，也不得沿用另一个 SP 的 aux 边界。
- 整模采集显存不足时，先审计 transformer block 的结构、attention 合同和目标捕获点。可以删除目标层之后的层并在捕获后立即停止；若选择 block 0 等早期同构代表层，也可以构造缩层模型。不得删除目标层之前会改变其输入分布的计算后仍宣称是该层真实输入。缩层依据、保留层范围和代表性边界必须写入 provenance。
- MiniMax-H3可作为包含production aux切分和dynamic sparse路径的输入捕获参考；实际目标模型仍由本次任务确定。Agent负责选代表block/step、生成replay/suite并执行，不将历史参考模型变成新任务默认。Sparse Ring leaf不满足分块LSE合同时记录不支持。

## 报告与持久化

工具输出结构化事实和简表。交付时按[REPORT_TEMPLATE.md](REPORT_TEMPLATE.md)生成最终中文REPORT.md并逐项核对本次涉及类型；SP须有独立入口。不要将CLI简表当最终页面，不为新请求把完整renderer重新加入公共代码。

- 推荐数值筛查只用于已有独立比较。NRMSE / 最大绝对误差÷参考RMS可用：FP16/BF16非近似路径0.5%/5%，FP8或Sage2为5%/25%，INT8为2%/10%，MoE为1%/10%。按实际backend/计算精度选择；未知或不适用不套值。超出时标注“超出推荐筛查值”，零参考看绝对误差；不覆盖SP显式门限、失败状态或开发者质量要求，单卡不因此重算参考。
- 仅对影响本次目标的异常/缺口给具体恢复建议：设备先核可见性，依赖核版本/符号，硬件/输入核实际合同，未接入查生产入口；不静默改精度/shape，无新证据不重复失败任务。补测据成本/波动/缺口确定，采用范围限定当前输入/硬件。
- 保留raw/catalog/report.json及身份，Agent解释/筛查/决定引用事实且单独保存；不重写原始记录或把建议说成工具自动发现。交付前按模板逐行核对数值/单位/输入及覆盖；必要时用小脚本做算术核对，一次性脚本留本地。

可复用代码、schema、接口说明和本指南随Git迁移；机器信息、权重、tensor、raw、trace、一次性脚本留本地。通用探索成熟后按职责迁入对应模块，不能把本地脚本称正式自动能力。所有结论限定实际输入、硬件与证据范围。

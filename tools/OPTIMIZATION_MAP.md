# 带代码导航的优化方法地图

用于给定模型和平台后的优化调查：先找已有做法，再沿生产调用判断是否适用。条目表示项目中存在可参考实现；不表示所有模型都已接入、当前配置已经启用，或在新平台上有收益。源码、依赖与PR的具体核对版本、覆盖范围和未覆盖来源保存在当前任务记录中；本地图不是全部模型、后端或PR的穷尽清单，也不以源码核对代替性能测量。

**使用前按任务实际固定的源码/依赖版本核相关变化，必要时更新受影响条目。** 核实际HEAD及相关依赖，检查下列入口的配置、构造、调用与消费者；结合新增模型、实现和可获取的相关PR补齐、合并或修正条目，更新核对版本，不为查方法自动切到上游最新。路径或行为失效时记录替代实现，未合入PR只作待核来源。没有相关变化时注明核验范围与复用依据，不要求修改地图。本文提供方向，任务源码决定支持，实际观测决定优先级。

对当前任务另列“已生效、可测、缺事实、不适用、已验证/拒绝”等处置，绑定版本、证据和重开条件；不因一次无收益或质量拒绝从方法地图删除整类方法。先确认当前成本，再选择最小充分验证，不要求逐项实现或遍历全部组合。多个现有开关也可能互斥。完整推进与恢复见[探索skill](../.agents/skills/explore-model-performance/SKILL.md)，变化触发与补证见[项目/依赖核验](../.agents/skills/explore-model-performance/references/version-refresh.md)。

## 任务pipeline覆盖表与方法库

- **任务pipeline覆盖表**由[报告模板](EXPLORATION_REPORT_TEMPLATE.md)第二部分承载，按本任务真实阶段→模块→调用及数据/状态关系展开，包含加载、搬运、通信、等待和输出；不预设所有模型共用一条pipeline。给对象和跨模块边界明确归属，父节点汇总子项，嵌套或异步重叠时间不直接相加。
- **本页方法库**保留通用机制、代码导航、前提与验证方向，由任务覆盖表跨阶段引用。下方现有入口混合了机制与对象导航，不是互斥、完备或固定不变的方法全集；attention、MoE、VAE等对象入口还需展开其真实方法子项。
- 两表交叉发现机会：沿真实pipeline防漏对象/阶段，沿方法库和当前实现防漏方法/变体。只展开任务相关、有适用依据的关系，不填写全笛卡尔表；子项展开及准出按skill执行。若有新路径或新方法，补充对应一侧。沿输出消费者核最末必需计算及未消费的头、尾层和返回值，不只检查开关；裁剪时保留隐状态索引、最终归一化及状态释放合同，先在对应模块核严格输出。
- 同一方法知识只在本库维护；任务生效状态、变体及处置沿用唯一工作包，由任务覆盖表展示，不再另建重复的方法状态表。同一候选可被多对象/机制引用，但实测结果、依赖和收益只维护一份。例如VAE入口可关联compile、低精度、切块和并行，量化与激活融合也可同时关联多种机制，不分别累计其收益。

## 快速索引

| 方法／对象入口 | 主要调查的成本 | 代表参考入口 |
| --- | --- | --- |
| 整阶段驻留/offload | 请求间重复搬运与显存占用 | [Qwen encoder/runner](../lightx2v/models/runners/qwen_image_21/qwen_image_21_runner.py) |
| block/phase异步预取、磁盘流式加载 | 权重搬运、主存容量、启动与等待 | [offload manager](../lightx2v/common/offload/manager.py) |
| 连续 pinned block 布局 | 多次小搬运、CPU提交开销 | [block layout](../lightx2v/common/offload/block_layout.py) |
| 部分block静态常驻 | 反复搬运与可用显存的取舍 | [resident blocks](../lightx2v/common/offload/resident.py) |
| host/NUMA共享CPU权重 | 多rank的主存副本与NUMA访问 | [shared coordinator](../lightx2v/common/offload/shared_weight_coordinator.py) |
| GEMM后端与低精度 | 矩阵计算、权重带宽、动态量化 | [MM weights](../lightx2v/common/ops/mm/mm_weight.py) |
| 稠密attention后端 | attention计算与布局 | [Flash](../lightx2v/common/ops/attn/flash_attn.py) / [Sage](../lightx2v/common/ops/attn/sage_attn.py) |
| 稀疏attention算法与实现 | 选区、路由及attention计算 | [general sparse](../lightx2v/common/ops/attn/general_sparse_attn.py) |
| MoE后端与执行配置 | 专家计算、路由分布及分片 | [fused MoE](../lightx2v/common/ops/moe/fused_moe.py) |
| 受约束卷积改写为GEMM | 每patch小GEMM、im2col与布局 | [Neopp patch/dense embedding](../lightx2v/models/networks/neopp/infer/pre_infer.py) |
| norm/RoPE/激活/残差融合 | 小算子、临时量与launch | [Qwen block](../lightx2v/models/networks/qwen_image_21/infer/transformer_infer.py) |
| 同输入量化复用、projection合并 | 重复量化与多个投影调用 | [optimized linear](../lightx2v/common/ops/mm/optimized_linear.py) |
| 固定线性LoRA的动态分支与加载合并 | 重复低秩计算与一次合并/重载成本 | [MM LoRA](../lightx2v/common/ops/mm/mm_weight.py) / [H3 merger](../lightx2v/models/networks/minimax_h3/lora.py) |
| torch.compile | Python/算子调度与可融合计算 | [transformer infer](../lightx2v/common/transformer_infer/transformer_infer.py) |
| 同进程warmup与库autotune | 首用、编译与未调优路径 | [runner](../lightx2v/models/runners/default_runner.py) / [autotune](../lightx2v/common/flashinfer_autotune.py) |
| CUDA graph | 重复调用的CPU发射开销 | [AR graph stack](../lightx2v/common/transformer_infer/autoregressive_cuda_graph.py) |
| 条件KV预填充与及时释放 | 固定条件的重复计算和阶段显存重叠 | [Qwen model](../lightx2v/models/networks/qwen_image_21/model.py) |
| 固定采样设置的AdaLN预计算 | step/block重复调制计算 | [H3 AdaLN cache](../lightx2v/models/networks/minimax_h3/adaln_cache.py) |
| AR KV与参考条件缓存 | 历史token及参考内容的重复计算 | [KV manager](../lightx2v/common/kvcache/manager.py) |
| 跨步特征缓存 | 去噪步骤中的近似复用 | [Wan caching](../lightx2v/models/networks/wan/infer/feature_caching/transformer_infer.py) |
| SP及通信/布局/heads流水 | attention分布、通信和等待 | [Ulysses](../lightx2v/common/ops/attn/ulysses_attn.py) / [Ring](../lightx2v/common/ops/attn/ring_attn.py) |
| TP、CFG与混合并行 | 每卡计算shape、权重容量与分支并发 | [MMWeightTP](../lightx2v/common/ops/mm/mm_weight.py) / [parallel配置](../lightx2v/utils/set_config.py) |
| PipeFusion | patch流水与多卡阶段利用 | [Flux2 driver](../lightx2v/models/networks/flux2/infer/pipefusion/pipeline_driver.py) |
| VAE计算、切块和空间/时间并行 | 解码/编码计算与激活峰值 | [Qwen VAE](../lightx2v/models/video_encoders/hf/qwen_image_21/vae.py) / [H3 VAE](../lightx2v/models/video_encoders/hf/minimax_h3/video_vae.py) |
| 轻量autoencoder替换 | 编解码模型本身的计算 | [Wan LightVAE](../lightx2v/models/video_encoders/hf/wan/lightvae.py) / [TAE](../lightx2v/models/video_encoders/hf/tae.py) |
| 异步解码、后处理和输出流水 | GPU/CPU等待、格式转换与编码 | [H3 streaming](../lightx2v/models/video_encoders/hf/minimax_h3_causal/streaming.py) |
| 蒸馏、CFG与变分辨率取舍 | 步数、分支数与每步token量 | [step distill](../lightx2v/models/schedulers/wan/step_distill/scheduler.py) / [resolution](../lightx2v/models/schedulers/wan/changing_resolution/scheduler.py) |

## 内存、权重与搬运

### 整阶段驻留/offload

- **做法与代码**：`QwenImage21Runner.run_text_encoder`在编码阶段onload并在`finally`释放设备权重；[QwenImage21TextEncoder](../lightx2v/models/input_encoders/hf/qwen_image_21/qwen3vl.py)保留pinned CPU主副本，`to_cpu`恢复引用。关闭对应offload可保留设备权重。
- **复用边界**：参考权重owner和阶段生命周期；不能把它当逐层offload，模型/vision分支及部分失败的清理要沿调用核实。
- **版本变化与重开**：编码器新增CPU主副本复用时，旧搬运结论需重核参数、量化buffer和释放路径；[create_pin_tensor](../lightx2v/common/ops/utils.py)可回退普通CPU内存，副本存在或`non_blocking=True`不证明实际pinned、重叠或加速。
- **验证**：按声明的阶段/请求范围核实际搬运、结束占用、首用/同进程热态延迟、显存和输出。多模块驻留组合须核跨阶段权重同时存活和实际峰值，不能只看单项分别装得下。

### block/phase异步预取与磁盘加载

- **做法与代码**：`WeightAsyncStreamManager.prefetch_weights/prefetch_phase`配合[event manager](../lightx2v/common/offload/event_manager.py)双buffer；[Qwen offload infer](../lightx2v/models/networks/qwen_image_21/infer/offload/transformer_infer.py)展示模型接线。磁盘来源另看[SafetensorsCheckpoint](../lightx2v/common/offload/safetensors_checkpoint.py)和[Qwen MPS streaming](../lightx2v/models/networks/qwen_image_21/infer/offload/mps_transformer_infer.py)。
- **复用边界**：CPU预取、设备搬运与计算依赖分开；磁盘流式是特定路径的容量方案，不能视为CUDA通用加速开关。
- **验证**：核buffer复用/最后使用点、存储和CPU瓶颈；正常计时与独立时间线核等待/重叠，保留主存、显存和首次读取成本。

### 连续pinned布局

- **做法与代码**：`BlockLayout/BlockBuffer/ContiguousBlockTransfer`把权重组织为连续块；[prepare_contiguous_group](../lightx2v/common/offload/block_loader.py)负责准备与检查，[H3示例](../scripts/minimax_h3/inference/offload_layout/README_CN.md)对比逐tensor与整block搬运。
- **复用边界**：布局、量化scale、别名和目标buffer由实际权重表示决定；连续布局与共享权重、常驻策略是不同能力，不默认互相兼容。
- **验证**：比传输次数、CPU提交、相关阶段/请求延迟与内存；核权重绑定、输出及异常清理。

### 静态部分常驻

- **做法与代码**：`ResidentBlocks`按固定数量均匀选择常驻block，[estimate_offload_residency](estimate_offload_residency.py)独立估测数量；[Qwen示例](../scripts/qwen_image_21/inference/offload_residency/README_CN.md)展示baseline与常驻比较。
- **复用边界**：当前`validate_residency_config`限定单设备eager、固定权重/分辨率和NoCaching，并拒绝多种并行、compile、LoRA、共享权重等组合；不是运行时自动调度器。
- **验证**：输入/设备变化后重新估测，核实际显存余量与声明范围的收益；不能把估测数量当任意请求的安全容量。

### host/NUMA共享权重

- **做法与代码**：`materialize_shared_weight_arena`协调CPU共享，[numa_weights](../lightx2v/common/offload/numa_weights.py)按域组织；[H3 adapter](../lightx2v/models/networks/minimax_h3/shared_block_weights.py)和[使用入口](../scripts/minimax_h3/inference/shared_offload/README_CN.md)展示模型集成。
- **复用边界**：主要目标是减少CPU物理副本；核IPC/NUMA、rank组、loader与原生算子的零拷贝表示，不承诺host模式一定更快。
- **验证**：看实际物理内存、NUMA访问、加载/错误协调、清理及请求表现；各rank的虚拟映射不能直接当物理副本相加。

## 核心计算与融合

### GEMM后端与低精度

- **做法与代码**：`MMWeightTemplate/MMWeightQuantTemplate`及注册实现提供不同权重/激活/累加路径；[converter](convert/converter.py)和[Qwen FP8 profile](convert/quant/qwen_image_21_fp8_f16_accum.py)展示checkpoint表示与范围配置。
- **复用边界**：权重dtype、scale、qmax、动态量化、bias/LoRA和TP需匹配消费者；同称FP8的实现不可随意换权重。硬件/依赖准入由当前backend核实，不维护静态后端全集。
- **验证**：固定真实shape/输入比较完整生产算子调用，计入动态量化成本；有模块交互或请求结论时再补相应范围，精度变化另按目标核数值/媒体。
- **实现演进**：早期FP8-F16来自SM120，后续[累加后端](../lightx2v/common/ops/mm/fp8_f16_accum.py)已有SM90 FP16/FP32路径；核实际扩展/注册、checkpoint范围和模型启用调用，不能沿用旧硬件名单，也不能把配置名或fallback当新kernel执行。

### 稠密attention后端

- **做法与代码**：`FlashAttn*`、`SageAttn*`等经[attention注册入口](../lightx2v/common/ops/attn/__init__.py)选择；核目标模型真正传入的Q/K/V与调用方法。
- **复用边界**：head布局、causal/varlen、KV表示、mask、精度、硬件与依赖共同决定支持；同名backend在新版本仍需核实。
- **验证**：比较完整生产调用和输出，必要时真实输入bench；不能只比某个内部kernel或以import成功宣称可执行。

- **结构化mask的实现选择**：当同一Q的可见K/V可以分成互斥且覆盖原mask的少量子集时，可调查复用`apply_with_lse`与[Ring的LSE合并](../lightx2v/common/ops/attn/ring_attn.py)；子集内统一score偏置可加到该子集LSE。分区逻辑留在拥有mask语义的模型侧，先证明帧/位置/padding与偏置等价，再核实际库接口、临时量及完整合并调用；全局交互或请求结论补对应证据。部分输出的低精度舍入仍可引入差异，不预先批准质量或保证加速。

### 稀疏attention

- **做法与代码**：`GeneralSparseAttnWeight`组合[mask generator](../lightx2v/common/ops/attn/sparse_mask_generator.py)与[sparse operator](../lightx2v/common/ops/attn/sparse_operator.py)，包含生成选区、重排、计算和还原；[DynamicSparseAttnWeight](../lightx2v/common/ops/attn/dynamic_sparse_attn.py)保留另一种集成路径。
- **复用边界**：算法、执行backend及各自依赖分开核实；一路缺依赖不能关闭整个稀疏父项，继续核当前可达变体及其模态、层/step准入。SLA/top-k和数据相关选区不能混为一种固定稀疏率，某模型的质量拒绝不等于其他模型不支持。
- **验证**：用真实RoPE后QKV比较含router、重排、计算与还原成本的完整生产调用及最终选区计算量，按实际层/step/模态覆盖核成功调用和fallback；eligible只表示准入，不表示已执行。按变更风险和声明范围核数值/生成质量，请求提速须请求证据，不同稀疏配置不默认等价。

- **依赖接口与wrapper边界**：返回值不足或矩形输入受限时，先核实际安装库的公开ABI与下游消费者；wrapper丢弃的LSE/路由信息不等于库或硬件缺能力。扩展共享consumer时保留默认返回合同，并将range/padding/块尺寸等事实沿真实producer传递。
- **动态路由与LSE**：固定选区的native计时不含每次mask变化的plan、ranges与布局成本，不能直接外推请求收益。用于分区合并前核实际API的LSE对数底与尺度；例如已验FlashInfer稀疏API返回log2 LSE，Magi返回自然对数，需按当前依赖合同转换。attention输出接近不足以证明LSE合并等价。

### MoE后端

- **做法与代码**：`FusedMoETemplate`及FlashInfer/Torch等实现；[HunyuanImage3 MoE权重](../lightx2v/models/networks/hunyuan_image3/weights/common.py)构建实际phase/shard backend，[transformer调用](../lightx2v/models/networks/hunyuan_image3/infer/transformer_infer.py)传入真实专家索引和权重。
- **复用边界**：专家配置、激活/量化表示、分片、capacity/drop及输入路由分布分别核实；特定multi-micro路径有专门约束，不能当通用MoE开关。
- **验证**：真实固定输入与路由，比较完整MoE调用并完成必要autotune；需解释层级交互时再核整层。直方图不代替真实tensor，计算量不代表负载均衡。

### 受约束卷积改写为GEMM

- **做法与代码**：[Neopp的`conv_as_gemm/extract_feature`](../lightx2v/models/networks/neopp/infer/pre_infer.py)把不重叠patch/dense embedding窗口展开为矩阵，复用原权重和bias做一次GEMM；由`use_conv_as_gemm`独立控制。
- **复用边界**：当前helper要求kernel等于stride、padding=0、dilation=1、groups=1，窗口须按`(C, kh, kw)`对应权重，并保留原卷积dtype/量化与后处理合同；不是任意Conv2d/Conv3d替换。布局重排成本及浮点舍入仍需核实。
- **验证**：确认原路径确有逐patch或低效卷积成本，核窗口/输出布局与数值，再比较完整调用；原PR的ROCm收益不外推到其他平台。

### 小算子与激活融合

- **做法与代码**：`QwenImage21TransformerInfer._finish_block`调用[triton_ops](../lightx2v/models/networks/qwen_image_21/infer/triton_ops.py)融合残差/norm/scale等；`apply_qk_rms_norm`由[norm实现](../lightx2v/common/ops/norm/rms_norm_weight.py)提供。H3的[apply_swiglu_linear](../lightx2v/common/ops/mm/optimized_linear.py)展示激活与量化融合。
- **复用边界**：保留归约、舍入、clamp、bias/LoRA与实际buffer语义；Qwen融合有输入规模门槛，小输入不保证更快。
- **验证**：先核当前路径是否已经融合，比较数值、辅助计算与完整调用；高频小算子只提供方向，收益按实测范围报告，宣称请求提速才须请求证据。

### 量化复用与projection合并

- **做法与代码**：`make_projection_group`共享一次输入量化，可选合并GEMM；[Wan.prepare_autoregressive_projection_groups](../lightx2v/models/networks/wan/model.py)展示QKV/cross-KV及offload buffer接线。
- **复用边界**：必须是同一实际输入，核支持的权重/TP表示、输出布局、LoRA及fallback；相同shape不证明可共享。权重搬运后旧group的存储可能失效。
- **验证**：确认量化次数实际减少，核packed输出消费者、权重切换和显存生命周期，再做局部/请求比较。group创建成功不证明本次未fallback；用实际quant/GEMM调用核路径。共享helper可能绕过MM入口统计，此时保留其明确覆盖缺口，不能把记录变少当计算量减少。
- **估计提示**：不同projection的输入宽度可不同；估计量化收益先核实际元素数/搬运量，不按减少的调用次数直接等比例缩放耗时。
- **打包输出**：[projection group](../lightx2v/common/ops/mm/projection_group.py)返回的split可带行stride；连续扁平地址读取的消费者不能直接使用。核隐式reshape/copy和临时存活，原模块可绑定packed存储视图，不直接假定长期多一份权重；合并GEMM不减少理论计算量。

### 固定线性LoRA的动态分支与加载合并

- **做法与代码**：`MMWeightTemplate.apply_lora`每次调用执行低秩分支；[MiniMaxH3LoraAdapter](../lightx2v/models/networks/minimax_h3/lora.py)展示加载时合并后重绑权重。两条路径分别处置，不以其中一条已生效代替另一条的调查。
- **复用边界与成本**：按实际目标层、原权重来源及量化、lazy、TP等guard核可达性；比较重复调用可消除的成本与一次合并、重载/重绑及搬运成本，不能从局部权重子集外推整模型准备成本。
- **数值与验证**：区分delta计算dtype、合并累加、最终权重存储与推理输出舍入；动态分支改为权重合并还会改变结合顺序，FP32合并计算不保证最终exact。按声明范围核输出/生成质量和收益，局部误差或加速不代替请求结论。
- **状态复用**：合并或重应用后沿真实owner核packed权重、offload与compile等状态更新。缓存是否失效取决于LoRA实际目标及其权重/数值依赖；有证据支持时可复用无关缓存，不强迫全部失效，也不因某次目标无关就沿用到其他LoRA。

## 执行与重复调用

### torch.compile

- **做法与代码**：`BaseTransformerInfer.init_compile/run_block`按实际block缓存compiled callable；[Qwen VAE](../lightx2v/models/video_encoders/hf/qwen_image_21/vae.py)展示将计算段编译而把collective留在外面。
- **复用边界**：核模型实际经过`run_block`、shape变化、graph break、自定义算子、offload与依赖版本；源码支持不表示推荐配置已打开或一定有收益。
- **验证**：区分编译/首用/同进程稳态，比较实际执行路径、内存和数值；重启进程需要重新建立进程内状态。
- **数值条件**：保留显式FP32输入/权重，不证明编译后的中间舍入与eager相同。核当前编译器的精度选项及实际输出；支持时可将精度转换模拟作为独立条件候选，与默认编译比较误差和成本，不将某版本的一次结果升级为全局默认。

### warmup与autotune

- **做法与代码**：`DefaultRunner.warmup`调用模型自己的`run_warmup`，[Qwen runner](../lightx2v/models/runners/qwen_image_21/qwen_image_21_runner.py)给出实例；`FlashInferAutotune.session`提供库策略缓存/搜索边界，MoE bench另按[执行指南](benchmarks/AGENTS.md)组织调优。
- **复用边界**：模型warmup、库autotune、profiler scheduler warmup目的不同；已有cache文件也不证明覆盖本次输入，普通预热不能替代必要搜索。
- **验证**：确认同进程代表性调用、状态恢复和本次shape的策略，分开首用与稳态；自动预热收敛不作为已交付能力。

### CUDA graph

- **做法与代码**：`AutoregressiveCUDAGraphStack`维护捕获/重放，显式处理输入、bindings与被修改buffer；[Wan AR](../lightx2v/models/networks/wan/infer/autoregressive/transformer_infer.py)和[H3 AR](../lightx2v/models/networks/minimax_h3/infer/autoregressive/transformer_infer.py)展示调用。
- **复用边界**：固定布局/地址、状态更新、权重迁移、采样分支及并行通信均需核对；不能把任意有状态pipeline包进重放。
- **验证**：核捕获次数、失效/重新捕获、实际重放输出和稳态收益；保留graph额外显存及首用成本。

## 重复计算与状态复用

### 条件KV预填充与释放

- **做法与代码**：`QwenImage21TransformerModel.prefill_condition_kv/clear_condition_kv`为单请求创建条件cache；runner在去噪后、VAE前释放，并在`end_run`清理。
- **复用边界**：必须证明条件分支在所复用步骤中不变；当前是请求内复用，不能自动扩成跨prompt共享。已有条件预填充不能再作为“待开启”的新收益。
- **验证**：核prefill次数、异常/新请求清理、VAE阶段峰值和输出；显存下降与请求加速分别报告。

### AdaLN预计算

- **做法与代码**：`load_persistent_adaln_cache`检查manifest、表项、shape与dtype；[构建工具](cache_minimax_h3_adaln/cache_minimax_h3_adaln.py)生成表，[H3 infer](../lightx2v/models/networks/minimax_h3/infer/transformer_infer.py)消费。
- **复用边界**：与权重、模型variant、实际条件分支的时间步组合、采样时间步/flow shift和相关投影合同绑定；不同变体的自动权重身份校验范围不同，使用者仍须核实际权重来源。固定采样设置下的预计算不同于近似跨步缓存。
- **验证**：核命中/失效与真实读取，保持相同请求比较；改变采样或权重后不能沿用旧表或旧结果。

### AR KV与参考条件缓存

- **做法与代码**：`KVCacheManager`组织static/rolling等cache；[H3 causal KV](../lightx2v/models/networks/minimax_h3/causal/kv_cache.py)和[MiniMaxH3ReferenceCache](../lightx2v/models/runners/minimax_h3_causal/reference_cache.py)展示历史token和参考条件两类复用。
- **复用边界**：核token位置、窗口/slot、step、分支、量化表示、参考内容及owner；参考条件cache与attention KV不是同一对象。
- **验证**：命中/失效、跨请求隔离和输出分别核；稀疏/量化/滑窗变体可能改变计算或数值，不能统一承诺exact。

### 近似跨步缓存

- **做法与代码**：`WanTransformerInferTeaCaching/TaylorCaching/MagCaching`等复用或预测中间特征；[H3 DPCache](../lightx2v/models/networks/minimax_h3/infer/feature_caching/dpcache/transformer_infer.py)提供另一种调度/校准接线。
- **复用边界**：模型、噪声调度、阈值/校准、跳过区间和质量约束共同决定适用性，不能把一个模型的参数直接迁移。
- **验证**：报告实际跳过/重算成本、端到端收益和最终媒体差异；exact要求下不得自行接受近似。
- **短调度的可达性**：同时核路径搜索的跳距、强制计算步和校准成本的历史anchor窗口。只放宽搜索边，不能补出原矩阵未校准的有限成本；显式扩展时两端共用同一参数并绑定校准身份，默认行为保持兼容。唯一可行路径只证明该策略可执行，不证明成本在多策略中选优；因子应来自当前请求，首用校准数值也不等于稳态误差保证。

- **固定步位的轻量接入**：先观测实际跨步残差，再决定有界复用；末步不必最稳定，hidden误差也不直接等于最终媒体可接受性。若只缺单个模型接线，把必要残差/dtype状态与reference/segment清理留在模型owner，而不复制整套通用引擎。明确实际完整层栈次数与solver更新次数，核当前步head/并行汇总/反馈仍执行；阈值、步位、模型或精度变化重新核质量与收益。

## 多卡、VAE与输出

### SP与通信优化

- **做法与代码**：`UlyssesAttnWeight`拆分head并可组合布局/A2A、量化、QKV融合和head group流水；`RingAttnWeight`旋转KV并合并输出/LSE；[Qwen infer](../lightx2v/models/networks/qwen_image_21/infer/transformer_infer.py)展示模型参数消费。
- **复用边界**：真实SP组、TP后的局部heads、主序列/复制aux和leaf能力必须匹配；当前Ring不接受head流水或任意Ulysses布局backend。
- **验证**：比较含通信/布局的完整SP调用；需解释层级交互或请求效果时补对应证据，数值和真实并行配置另核。

### TP、CFG与混合并行

- **做法与代码**：`MMWeightTP`处理线性分片/归约；[set_config](../lightx2v/utils/set_config.py)建立实际mesh，[H3权重分片](../lightx2v/models/networks/minimax_h3/weights/tensor_parallel.py)、[Qwen TE](../lightx2v/models/input_encoders/hf/qwen_image_21/qwen3vl.py)和[Wan.infer](../lightx2v/models/networks/wan/model.py)提供DiT/encoder/CFG参考。
- **复用边界**：核模型自己的配置准入，TP/SP/CFG不同组的输入/权重与bias/归约语义；关闭CFG与并行计算CFG是不同选择，前者涉及产品质量。H3的[Text Encoder](../lightx2v/models/input_encoders/hf/minimax_h3/qwen3vl.py)可按WORLD组分片，独立于DiT mesh；按实际encoder绑定核TP度数、offload和输出，不由DiT TP字段推断。
- **验证**：固定全局输入与资源，按问题选完整层/阶段或请求比较；局部GEMM shape退化、collective、输出还原及显存都要考虑，层结果不直接外推请求。

### PipeFusion

- **做法与代码**：`Flux2PipelineDriver`分配模型阶段及图像patch，区分同步预热与异步流水；[Flux2 runner](../lightx2v/models/runners/flux2/flux2_runner.py)的`_run_pipefusion`负责实际接入。
- **复用边界**：模型分段、patch/KV和调度强相关；该参考的异步路径包含跨patch复用，不能当任意模型的精确pipeline并行开关。
- **验证**：核阶段/patch负载、等待与存储存活，比较完整去噪和质量；别用某阶段局部最快替代全流水结果。

### VAE计算、切块与并行

- **做法与代码**：`QwenImage21VAE`包含compile、切块、空间并行和[混合FP8 conv](../lightx2v/models/video_encoders/hf/qwen_image_21/fp8_conv.py)；`MiniMaxH3VideoVAE`提供时空tile、并行编解码和相关低精度路径。共用卷积参考[Conv2d](../lightx2v/common/ops/conv/conv2d.py)/[Conv3d](../lightx2v/common/ops/conv/conv3d.py)。
- **复用边界**：核halo/边界、norm/attention感受野、tile拼接与scale；Qwen当前切块明确不兼容FP8 VAE。不同切点/量化可能有质量变化。
- **验证**：真实输入下按模块/阶段核耗时、峰值、边缘/时序差异；请求效果补请求证据。更省显存不等于更快，tile额外计算和通信需计入。
- **中间量**：当前Qwen VAE的`ImageRMSNorm`、`ImageDupUp3D`和残差块还展示private buffer复用、只物化保留帧及原地操作；先证实别名/最后使用点、dtype与边界，再核峰值及质量，不重复把已生效路径算成新收益。

### 轻量autoencoder替换

- **做法与代码**：`WanLightVAEDecoder`和`TAEHV`提供更小解码器的集成参考；入口可沿[Wan runner](../lightx2v/models/runners/wan/wan_runner.py)及目标模型配置查找。
- **复用边界**：需要匹配checkpoint、latent表示、归一化、条件编码和时序状态；这是模型替换，不能当无损实现替换。
- **验证**：性能与最终图像/视频质量单独验收，保留原decoder对照，不因局部误差小就批准产品使用。

- **轻量decoder输入域**：按实际checkpoint训练约定核latent缩放与dtype，不能由原生VAE的反标准化公式推定替代decoder输入域。模型需要参考编码不必排除只替换decoder；沿factory确认native encoder保留，再核跨clip反馈与完整媒体。

### 异步解码与输出流水

- **做法与代码**：`MiniMaxH3StreamingVideoDecoder.push/finish`处理流式窗口；[H3 causal runner](../lightx2v/models/runners/minimax_h3_causal/minimax_h3_causal_runner.py)连接真实生成，[StreamingVideoWriter](../lightx2v/utils/streaming_video.py)承担异步编码/收尾。普通图片后处理可参考[Qwen runner](../lightx2v/models/runners/qwen_image_21/qwen_image_21_runner.py)的OpenCV路径。 非流式音视频保存见[H3 runner](../lightx2v/models/runners/minimax_h3/minimax_h3_runner.py)到[encode_video](../lightx2v/utils/ltx2_media_io.py)的`video_codec_options`入口，可先在实际保存调用上选配置。
- **复用边界**：核stream依赖、pinned/uint8转换、队列背压、音画顺序和异常收尾；编码参数本身可能改变质量；线程模式/数量可能耦合lookahead与RGB→YUV重格式化，须核实际执行参数，不能把编码器自报线程数当整个管线线程预算。禁用输出不算同一请求优化。
- **验证**：按任务目标报告首个输出、持续处理或完整完成等实际边界；声明完成须等待相应finish/mux，核媒体内容、帧数和内存，不能只测异步提交。

- **本地输出收尾**：区分按播放时长估计的等待与真实FIFO/线程/TCP EOF/FFmpeg退出及mux完成，参考[VideoRecorder](../lightx2v/utils/video_recorder.py)和模型覆盖的stop。只在能保留完整排空、异常清理与成片内容时考虑取消估计等待；`wait=False`在某些模型表示abort，不能按名字当无损flush。

## 需要单独质量授权的算法取舍

### 蒸馏、CFG与变分辨率

- **做法与代码**：`WanStepDistillScheduler`、[H3四步LoRA入口](../scripts/minimax_h3/inference/distill/minimax_h3_t2av_dmd.py)及`WanScheduler4ChangingResolution`展示减少步数/分支/每步token量的做法。
- **复用边界**：权重/LoRA、采样器、guidance及分辨率调度必须配套；不通过随意少跑几步或缩小输入冒充等价加速。
- **验证**：在开发者明确允许的范围比较完整请求和生成质量，分别报告算法取舍与实现加速；新训练/蒸馏工程不自动进入普通探索任务。

## 组合工具与维护边界

先核当前问题相关的已有成本事实，缺口按问题选择[正常pipeline日志](../lightx2v/utils/profiler.py)、[模块profile](profile/AGENTS.md)或[算子/SP bench与模块并行bench](benchmarks/AGENTS.md)，不要求先测完整请求或逐层升级。同步/non-sync、嵌套、重叠与排除成本须核实，不能直接相加；请求或全局交互结论补对应范围的证据。执行资源规则见[runtime](runtime/AGENTS.md)。沿已有工具产物保留预期/实测与未知，不另建计时/报告系统。

项目还包含[分离部署](../lightx2v/disagg/README.md)等服务能力；可按任务需要沿源码补进地图，当前CLI探索不自动扩展为服务容量优化。新方法按“机制、当前参考实现、复用前提、最小验证”补入，不按模型复制完整清单，也不维护固定的所有backend/组合名单。

## 方法来源举例

下表解释实现来历和历史取舍，现行生产调用链决定适用性；PR自述性能不代替本任务实测。开放PR可提供反例线索，须核当前代码是否已覆盖，不能按状态直接判定缺陷或复制旧patch。

| 来源 | 可追溯的方法与边界 |
| --- | --- |
| [#1533](https://github.com/ModelTC/LightX2V/pull/1533)、[#1535](https://github.com/ModelTC/LightX2V/pull/1535) | Qwen原生接入、条件KV预填充与block融合；已有能力先核是否生效 |
| [#1543](https://github.com/ModelTC/LightX2V/pull/1543)、[#1561](https://github.com/ModelTC/LightX2V/pull/1561) | Qwen阶段offload、FP8、并行与VAE；历史SLA因质量撤回且专用构造接线删除，DiT compile在当时5090路径无收益，均不等于永久否定机制 |
| [#1479](https://github.com/ModelTC/LightX2V/pull/1479)、[#1657](https://github.com/ModelTC/LightX2V/pull/1657) | SM120到Hopper的FP8累加/融合演进；范围、依赖和模型启用要分别核 |
| [#1503](https://github.com/ModelTC/LightX2V/pull/1503) | 固定设置AdaLN离线缓存；实际条件分支与表项覆盖一起核 |
| [#1650](https://github.com/ModelTC/LightX2V/pull/1650) | H3 causal参考复用、解码graph及输出流水；任务和状态owner不可省略 |
| [#1614](https://github.com/ModelTC/LightX2V/pull/1614) | 受约束卷积改写为GEMM；窗口合同和平台收益分开 |
| [#1627](https://github.com/ModelTC/LightX2V/pull/1627) | 减少VAE临时量、只物化有效数据；原地写的别名和数值合同 |
| [#1655](https://github.com/ModelTC/LightX2V/pull/1655) | TE CPU主副本复用；参数、量化buffer、实际pinned与收尾路径分别核 |

---
name: support-cpu-block-offload
description: 为 LightX2V 模型接入、审查和调试 CPU block offload 与多进程 CPU 权重共享（shared_cpu_weights）。用于权重适配器、共享加载、GPU 缓冲调度、异步复制生命周期，以及 host／NUMA 启动入口和验证；按用户范围处理局部审查或完整接入。
---

# LightX2V CPU Block Offload 与权重共享接入

简体中文 | [English](SKILL_EN.md)

## 目标与职责

沿用模型的正式推理路径，实现 CPU 常驻权重、GPU block 缓冲复用，以及多进程共享 CPU 权重。完整接入须交付 `scripts/<model>/offload/` 下支持 host、NUMA 两种模式的可运行脚本和配置，不能只增加开关或加载接口。

block offload 减少 GPU 权重驻留；CPU 权重共享减少同一共享域内的 CPU 权重副本。每个 rank 仍持有自己的 GPU staging buffer，仍需 H2D 复制。不要将其描述为 activation offload、GPU 权重共享或消除权重传输。

当前公共共享后端依赖 Linux SysV shared memory 与 CUDA host registration。普通 block offload 的设备适配范围更广；不能据此宣称共享后端已支持 XPU 或其他设备。

按任务读取参考资料：

- 接通加载、算子绑定或推理调度：读 [implementation-patterns.md](references/implementation-patterns.md) 对应部分。
- 复用 Wan、Qwen Image、MiniMax H3、Hunyuan Image 3.0：读 [model-adapters.md](references/model-adapters.md) 对应模型，区分现有条件和通用约束。
- 选择测试、验证内存收益或定位故障：读 [validation-and-debugging.md](references/validation-and-debugging.md)。

模型尚未具备原生推理骨架时，结合 [support_new_model](../support_new_model/SKILL.md)；任务包含 compile 或 warmup 时，分别结合 [support_model_compile](../support_model_compile/SKILL.md)、[support_model_warmup](../support_model_warmup/SKILL.md)。不要仅因新增 offload 就顺带实现这些功能。纯解释、审查、定位或明确限定为私有 block 的任务按用户范围执行，不自动扩展为完整共享接入。

## 1. 从启动脚本闭合调用链

依次检查脚本、最终配置、runner、model、weights、infer 和 offload manager。配置包含 JSON、模型配置、CLI 覆盖及 `DTYPE`、`SENSITIVE_LAYER_DTYPE` 等环境设置。

```bash
model=wan
rg -n 'cpu_offload|offload_granularity|shared_cpu|lazy_load|release_block' \
  "scripts/${model}" configs "lightx2v/models/networks/${model}"
rg -n '_load_shared_cpu_weights|_init_weights|_init_offload_manager|infer_with_blocks_offload' \
  "lightx2v/models/networks/${model}" lightx2v/models/networks/base_model.py
rg -n 'prefetch|swap_blocks|wait_ready|record_free|close_shared_cpu_weights' \
  lightx2v/common/offload "lightx2v/models/networks/${model}"
```

按实际任务补读文本编码器、VAE 和 runner；不要假定 `shared_cpu_weights=true` 会共享整个 pipeline。逐组件记录：

| 组件 | 权重格式与推理 dtype | CPU 常驻范围／私有范围 | block 类型及 GPU 缓冲 | 共享加载／关闭入口 |
|---|---|---|---|---|
| DiT | 据源码填写 | 据源码填写 | 数量、schema、stride | 调用者与 owner |
| 文本编码器、VAE 等任务内组件 | 分别填写 | 分别填写 | 整体、分块或不 offload | 各组件独立核对 |

同时建立支持矩阵：普通路径、私有 CPU block、共享 host、共享 NUMA，以及任务实际涉及的 SP、TP、量化、LoRA、compile 等组合。每项区分“已验证”“明确不支持”“尚未验证”，附代码或运行依据；配置可解析不等于实现闭合。

先复现同 task、checkpoint、输入和精度的基线。全量 GPU 驻留无法运行时，可用已验证的私有 CPU block 路径作基线，说明比较边界。不要删掉用户要求的量化、LoRA 或并行设置来制造成功结果。

## 2. 接通私有权重的 block offload

明确 block 的计算入口、CPU 权重来源和 GPU slot 的 schema。多种 block 类型分别规划兼容的缓冲，不假定全模型所有层同构。

- LightX2V `WeightModule` 路径：沿用 `state_dict()`、`load_state_dict()`、`offload_block_cuda_buffers` 和 `_init_offload_manager()`。
- 原生 `nn.Module` 路径：优先使用 `ModuleCPUWeights`、`NativeModuleBlockSource`、`NativeModuleBlockSlot`，保留参数别名及运行时 buffer 的处理。
- 优先复用模型已有 `WeightAsyncStreamManager` 调度。需要固定 slot 的 ready/free 事件协议时，采用 `EventSlotWeightAsyncStreamManager`；不为风格统一重写已正确的模型循环。

检查首块加载、末块处理、下一个 denoise step、调用方 stream 与 compute stream 的依赖，以及输出交回调用方时的同步。GPU slot 的复用要保持 shape、dtype、stride 和算子所需布局。

## 3. 接入共享 CPU 权重

模型 adapter 负责 checkpoint schema、tensor 选择、精度转换和权重签名；公共 coordinator 负责拓扑分组、创建、挂接、注册和错误传播。

1. 通过 header/index 和 meta tensor 构造 manifest，区分共享 tensor 与 rank 私有 tensor。不要先让所有 rank 完整加载一份共享 payload。
2. 将最终运行 dtype、布局及影响数值的转换策略纳入适配器契约；量化 scale 的转换顺序必须与基线一致。
3. 所有 rank 以相同组件顺序调用 `materialize_shared_weight_arena()`，只由每个 replica 的 leader 填充共享 payload。
4. 返回 `SharedWeightViewMap`，让消费者通过 `consume_weight()` 直接采用共享 view；支持多个消费者引用同一共享权重。
5. 在模型或组件上保存 owner，绑定后检查共享 view 的消费完整性、CPU 地址、dtype、shape、stride 和 pinned 状态。

共享 view 进入消费路径后，禁止以无条件 `clone()`、`contiguous()`、dtype 转换、重新 pin 等方式生成私有副本。需要物化的转换放在 leader 填充阶段；允许保持存储身份的转置等 view 操作。

各进程映射的虚拟地址可以不同。验证使用本地 arena 基址和 manifest 布局，不能比较跨进程 `data_ptr()` 是否相等。

`host` 按 host、IPC namespace、权重签名建立副本；`numa` 进一步按参与 rank 对应的 NUMA node 分组。每组 leader 和副本数由实际拓扑决定，不能写死为 rank 0、8 卡或两个 NUMA 节点。

当前模型对 TP、LoRA、量化、AdaLN cache 等条件的限制见案例文件。只为已确认不兼容的组合增加明确检查，不把某个 adapter 的限制复制成所有模型的禁令。

## 4. 明确生命周期

- 共享 CPU 源保持不可变；请求状态、KV cache 和可变运行时 buffer 留在各 rank 本地。
- GPU slot 覆盖前等待消费者完成；计算前等待 H2D 完成。reset 事件记账不能替代等待上一请求或 tile 的完成事件。
- 临时 weight map 销毁后 owner 仍须存在。关闭 arena 前，所有使用者必须停止访问，所有相关 DMA 必须完成。
- GPU 缓冲的阶段释放与 CPU arena 的最终关闭分开。释放后，下一请求应能从原 CPU 源重建 GPU 缓冲。
- 多组件共享初始化保持所有 rank 的组件与状态交换顺序一致。当前 CPU 加载状态通过分布式 Store 协调，不为等待 CPU 加载发起 NCCL collective。局部 preflight／绑定错误使用现有协调机制传播，释放已创建的本地资源。

修改 stream、释放或重建路径时，验证连续请求和异常路径。不要依靠 Python GC 的偶然时机保证设备操作安全。

## 5. 交付统一启动入口

每个模型在 `scripts/<model>/offload/` 保留一个可直接编辑的共享 offload 启动脚本、英文 `README.md` 和中文 `README_CN.md`，并引用一份纳入版本管理的 JSON 配置。脚本和共享配置使用任务无关的文件名，沿用模型现有命名；通过命令中的 `--task` 选择模型已支持的任务。默认示例为 H3 的 `t2av`、Qwen／Hunyuan 的 `t2i`、Wan 的 `i2v`：

```text
scripts/<model>/offload/
├── run_<model>_block_shared_offload.sh
├── README.md
└── README_CN.md
```

配置优先放 `configs/<model>/offload/`；Wan 沿用 `configs/offload/block/`。当前四个示例默认 host + 8 卡，脚本显式设置 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` 和 `--nproc_per_node=8`。Hunyuan 的默认布局是 TP2 × SP2 × CFG2，其余三个是 SP8。

共享方式由推理命令中的 `--shared_cpu_weight_scope host|numa` 选择；该 CLI 参数覆盖旧 JSON 的同名字段，属于启动配置，不进入请求参数。当前四份 JSON 不重复填写 scope；省略 CLI 时沿用 JSON，二者都未设置时内部默认仍为 `auto`。不要把脚本的 host 默认与内部 auto 默认混为一谈。

当前脚本不读取 `TASK`、`CONFIG_JSON`、`SHARED_CPU_WEIGHT_SCOPE` 等环境变量来覆盖命令，也不转发 `bash script.sh` 后追加的参数或自动推导卡数。修改路径、显卡、task、scope 和输入时，编辑脚本中的完整命令；修改并行参数时同步编辑 JSON。若不改文件，直接执行完整 Python 命令。不要重新引入环境变量包装器、临时 JSON 生成器或重复的 scope／卡数／任务启动脚本。

默认共享配置包含 `cpu_offload=true`、`offload_granularity="block"`、`shared_cpu_weights=true`、`shared_cpu_weight_backend="sysv"`、`shared_cpu_weight_strict_numa=true`、`shared_cpu_weight_register_chunk_mb=128`、`lazy_load=false`。额外组件仅开启已接通的对应共享开关。私有 block 基线可用于验证，不因该用途在共享入口目录新增永久脚本或专用配置；已有的其他入口不属于自动清理范围。

脚本和配置还须满足：

- 同一任务切换 host／NUMA 时使用同一 checkpoint、task、输入、shape、steps、seed、dtype、算子和并行规模，切换仅改变共享 scope。
- 在脚本顶部填写 `lightx2v_path`、`model_path`；遵循 `scripts/base/base.sh` 的变量契约，用当前环境的 `python -m torch.distributed.run ... -m lightx2v.infer ...` 启动。从仓库根目录运行；JSON 中的相对 checkpoint 路径也以工作目录为准。
- 手动保持显卡列表、`--nproc_per_node` 和 JSON 拓扑一致。进程数须等于有效配置中的 `tensor_p_size * cfg_p_size * seq_p_size`；缺省维度按 1 计。不要静默覆盖用户指定的并行参数。
- 先读默认值，再决定是否保留环境设置。当前四个入口沿用 `base.sh` 的 `DTYPE=BF16`、`SENSITIVE_LAYER_DTYPE=None`；`None` 表示跟随主 dtype。外部已设置的值会被保留，因此比较结果时记录有效 dtype。Wan 从旧 FP16 示例改为默认 BF16 后，不能沿用旧 FP16 验证结论。
- `torchrun` 只在多进程且环境未设置时默认 `OMP_NUM_THREADS=1`。Hunyuan runner 会根据 `HUNYUAN_IMAGE3_REPO_PATH` 补充上游导入路径，不需在脚本中重复追加该 `PYTHONPATH`。不为模板统一增加重复 export 或 `set -e`；确有语义需要的设置单独说明。
- 显式 NUMA 模式不能静默退回 host；未知拓扑、严格 NUMA 绑定失败应报错。可见卡数仍受模型注意力头和并行实现限制。
- 两份 README 写清默认任务、`--task` 支持范围及输入要求、NUMA 切换、其他卡数、配置位置、参数默认值与优先级、环境及资产要求、组件共享范围。修改 task 时同步输入参数、模型 variant、缓存和输出；不能只改任务名。使用说明中不堆放实验报告。

## 6. 验证与交付

按 [validation-and-debugging.md](references/validation-and-debugging.md) 选择与改动相关的测试：先验证 schema／存储身份／同步契约，再分别运行 host 和 NUMA 的真实入口。多 rank 共享与注册必须有相应运行证据；CPU mock 通过不能替代设备验收。

最终交付包括：

1. 组件范围、支持矩阵、关键调用链和仍存在的限制。
2. 权重 adapter、调度与生命周期改动，以及对应聚焦测试结果。
3. `scripts/<model>/offload/` 下支持 host／NUMA 的统一脚本、一份共享配置和中英文 README。
4. 同配置私有／共享结果对比、连续请求验证、共享存储及内存证据。
5. 如任务要求性能，分别报告加载／注册、首次请求、稳定推理的时间与内存。

只完成接口、未接通任一 scope 的运行方式，不算完整接入。资产或硬件不足时保留已完成的交付，明确标注未验证项和所缺条件；不能将脚本存在、mock 通过或小 shape smoke test 写成目标路径验收通过。

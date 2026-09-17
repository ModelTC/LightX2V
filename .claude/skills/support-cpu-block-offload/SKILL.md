---
name: support-cpu-block-offload
description: 为 LightX2V 模型接入、审查和调试 CPU block offload 与多进程 CPU 权重共享（shared_cpu_weights）。适用于新增模型适配器、接通共享权重加载与 GPU 缓冲调度、处理共享存储及异步复制生命周期，并交付 scripts 模型目录下 offload 子目录中的 host、NUMA 两种启动入口、对应配置和验证记录。
---

# LightX2V CPU Block Offload 与权重共享接入

## 目标与职责

沿用模型的正式推理路径，实现 CPU 常驻权重、GPU block 缓冲复用，以及多进程共享 CPU 权重。完整接入必须同时交付 `scripts/<model>/offload/` 下的 host、NUMA 两种可运行入口及对应配置，不能只增加开关或加载接口。

block offload 减少 GPU 权重驻留；CPU 权重共享减少同一共享域内的 CPU 权重副本。每个 rank 仍持有自己的 GPU staging buffer，仍需 H2D 复制。不要将其描述为 activation offload、GPU 权重共享或消除权重传输。

当前公共共享后端依赖 Linux SysV shared memory 与 CUDA host registration。普通 block offload 的设备适配范围更广；不能据此宣称共享后端已支持 XPU 或其他设备。

按任务读取参考资料：

- 接通加载、算子绑定或推理调度：读 [implementation-patterns.md](references/implementation-patterns.md) 对应部分。
- 复用 Wan、Qwen Image、MiniMax H3：读 [model-adapters.md](references/model-adapters.md) 对应模型，区分现有条件和通用约束。
- 选择测试、验证内存收益或定位故障：读 [validation-and-debugging.md](references/validation-and-debugging.md)。

模型尚未具备原生推理骨架时，结合 [support_new_model](../support_new_model/SKILL.md)；任务包含 compile 或 warmup 时，分别结合 [support_model_compile](../support_model_compile/SKILL.md)、[support_model_warmup](../support_model_warmup/SKILL.md)。不要仅因新增 offload 就顺带实现这些功能。纯解释、审查、定位或明确限定为私有 block 的任务按用户范围执行，不自动扩展为完整共享接入。

## 1. 从启动脚本闭合调用链

依次检查脚本、最终配置、runner、model、weights、infer 和 offload manager。配置包含 JSON、模型配置、CLI 覆盖及 `DTYPE`、`SENSITIVE_LAYER_DTYPE` 等环境设置。

```bash
rg -n 'cpu_offload|offload_granularity|shared_cpu|lazy_load|release_block' \
  scripts/<model> configs lightx2v/models/networks/<model>
rg -n '_load_shared_cpu_weights|_init_weights|_init_offload_manager|infer_with_blocks_offload' \
  lightx2v/models/networks/<model> lightx2v/models/networks/base_model.py
rg -n 'prefetch|swap_blocks|wait_ready|record_free|close_shared_cpu_weights' \
  lightx2v/common/offload lightx2v/models/networks/<model>
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
- 多组件共享初始化保持所有 rank 的 collective 顺序一致。局部 preflight／绑定错误使用现有协调机制传播，释放已创建的本地资源。

修改 stream、释放或重建路径时，验证连续请求和异常路径。不要依靠 Python GC 的偶然时机保证设备操作安全。

## 5. 必交付 host／NUMA 启动入口

每个完整接入的模型都必须具备 `scripts/<model>/offload/`。已有目录直接补齐；采用该模型现有命名习惯，名称须能辨认 task 和 scope。最小结构示意：

```text
scripts/<model>/offload/
├── run_<model>_<task>_block_shared_host.sh
├── run_<model>_<task>_block_shared_numa.sh
└── README.md
```

两个脚本分别引用纳入版本管理的 host、NUMA JSON 配置。配置沿用该家族目录约定，优先放 `configs/<model>/offload/`；Wan 等已有布局可继续沿用。仅提供修改 scope 的说明、一个 `auto` 入口或两个指向同一 host 配置的脚本，均不满足交付要求。

| 配置项 | host | NUMA |
|---|---|---|
| `cpu_offload` | `true` | `true` |
| `offload_granularity` | `"block"` | `"block"` |
| `shared_cpu_weights` | `true` | `true` |
| `shared_cpu_weight_backend` | `"sysv"` | `"sysv"` |
| `shared_cpu_weight_scope` | `"host"` | `"numa"` |
| `shared_cpu_weight_strict_numa` | `true` | `true` |
| `shared_cpu_weight_register_chunk_mb` | 默认 `128` | 相同值 |
| `lazy_load` | 当前共享路径为 `false` | 相同值 |

上表是当前完整模型接入的默认配置。仅扩展某个编码器或 VAE 时，使用其真实组件开关，清楚说明 DiT 是否参与，不为凑表格开启任务范围外的组件。

脚本和配置还须满足：

- 两种方式保持 checkpoint、task、输入、shape、steps、seed、dtype、算子和并行规模一致，除 scope 及输出文件名等标识外不引入无关差异。
- 根据脚本位置定位仓库，引用 `scripts/base/base.sh` 时遵循它的变量契约；允许通过环境变量或已有 CLI 覆盖模型路径、输入输出、Python 和可见设备，避免个人绝对路径。
- 进程数可通过入口配置，且与有效 JSON 中 `tensor_p_size * cfg_p_size * seq_p_size` 一致。可提供匹配的配置路径覆盖，或使用经过验证的临时配置生成方式；不能只改 `--nproc-per-node`，也不能发明现有 CLI 不接受的并行参数。
- 显式 NUMA 模式不能静默退回 host。未知拓扑、严格 NUMA 绑定失败等应给出真实错误；`auto` 只能作为额外入口。
- 多任务模型按本次承诺支持的任务提供成对入口。兼容任务可以复用 JSON，由 CLI 设置 task；不要仅凭同模型名称就宣称所有任务都已验证。
- README 列出两种命令、配置位置、可覆盖变量、并行规模、环境和资产要求、组件共享范围，以及各模式验证状态。

私有 block 基线脚本值得保留以便对照，但不能替代 host／NUMA 两个共享入口。已有案例缺少 NUMA 脚本时，在该模型的接入任务内补齐，不能照抄缺口。

## 6. 验证与交付

按 [validation-and-debugging.md](references/validation-and-debugging.md) 选择与改动相关的测试：先验证 schema／存储身份／同步契约，再分别运行 host 和 NUMA 的真实入口。多 rank 共享与注册必须有相应运行证据；CPU mock 通过不能替代设备验收。

最终交付包括：

1. 组件范围、支持矩阵、关键调用链和仍存在的限制。
2. 权重 adapter、调度与生命周期改动，以及对应聚焦测试结果。
3. `scripts/<model>/offload/` 下 host／NUMA 成对脚本、对应配置和 README。
4. 同配置私有／共享结果对比、连续请求验证、共享存储及内存证据。
5. 如任务要求性能，分别报告加载／注册、首次请求、稳定推理的时间与内存。

只完成接口、缺少任一 scope 的启动入口，不算完整接入。资产或硬件不足时保留已完成的交付，明确标注未验证项和所缺条件；不能将脚本存在、mock 通过或小 shape smoke test 写成目标路径验收通过。

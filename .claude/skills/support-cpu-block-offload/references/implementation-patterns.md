# 公共实现与接入契约

路径相对仓库根目录。以下是现有实现的导航，修改前用函数名定位当前源码；不依赖固定行号。按任务读取加载、调度或生命周期部分。

## 启动到权重绑定

```text
scripts/<model>/offload/*.sh
  → lightx2v/infer.py
  → utils/set_config.py: build_cli_inputs → build_startup_config
  → init_parallel
  → models/runners/runner_factory.py: build_runner
  → runner.init_modules / load_model / load_transformer
  → 模型构造函数
  → BaseTransformerModel._init_weights → _init_weights_impl
      ├─ 私有路径：_load_ckpt / _load_quant_ckpt
      └─ 共享路径：模型._load_shared_cpu_weights
          → 模型 adapter preflight
          → materialize_shared_weight_arena
          → SharedWeightViewMap(private, shared, owner=allocation)
      → pre / transformer / post weight 构造与 _apply_weights
      → _validate_shared_cpu_weights
  → 模型._init_infer → _init_offload_manager
```

配置与设备初始化见 `lightx2v/utils/set_config.py`、`lightx2v/models/runners/default_runner.py`。`set_init_device()` 根据 offload 选择 CPU；runner 管编排，模型和组件 loader 管权重。保留家族已经支持的 checkpoint 格式，不因增加共享强制转换全部模型格式。

[base_model.py](../../../../lightx2v/models/networks/base_model.py) 的关键契约：

| 入口 | 接入责任 |
|---|---|
| `_load_shared_cpu_weights(unified_dtype, sensitive_layer)` | 模型 override，返回带 owner 的共享 mapping；私有非 block 权重仍走正确加载与 dtype 策略 |
| `_init_weights_impl()` | 在 `_apply_weights()` 消费 mapping 前保存 `_shared_cpu_weight_owner`；构造权重对象，完成共享绑定校验 |
| `_init_weights()` | 协调各 rank 的模型初始化错误，失败后关闭已保存 owner |
| `_validate_shared_cpu_weights()` | 默认验证 transformer block；更严格的布局或额外共享组件由对应模型／组件补充 |
| `_init_offload_manager()` | 将 GPU block／phase 缓冲交给 infer manager；额外 CPU staging 是 lazy 分支，不能与完整共享 CPU 源混淆 |
| `close_shared_cpu_weights()` | 设备传输和访问停止后关闭 owner；后续不得继续使用旧 view |

如果接入路径直接传入 `weight_dict`，确认是否绕过默认共享分支及 owner 保存；不要假定配置开关会替所有自定义加载方式接管生命周期。

## Manifest、replica 与共享 arena

核心文件：

- [checkpoint_metadata.py](../../../../lightx2v/common/offload/checkpoint_metadata.py)：`read_safetensors_header()`、`checkpoint_content_digest()`。
- [shared_pinned_arena.py](../../../../lightx2v/common/offload/shared_pinned_arena.py)：`SharedWeightManifest`、`ReplicaPlanner`、`SharedPinnedArena`、`CudaHostRegistration`。
- [shared_weight_coordinator.py](../../../../lightx2v/common/offload/shared_weight_coordinator.py)：`validate_shared_weight_config()`、`coordinate_rank_local_error()`、`materialize_shared_weight_arena()`。

模型 adapter 首先确认 shard/index 完整性、tensor 名字、层号、shape、源 dtype，以及目标运行 dtype。用 meta tensor 描述最终 CPU 存储，构造确定性 manifest；共享签名应覆盖 checkpoint 内容、组件选择和影响数据解释／数值的转换版本。仅文件名或文件大小不足以辨认权重。

`checkpoint_content_digest()` 可复用仍有效的缓存摘要，否则扫描文件计算摘要。因此 follower 可能发生文件 I/O；“只有 leader 填充共享 tensor payload”不等于“只有 leader 读过任何 checkpoint 字节”。

协调顺序：

1. 所有 rank 发现 host、IPC namespace、当前 GPU PCI／NUMA 信息，并交换 manifest 与 policy。
2. 核对各 rank 的布局和策略，规划 replica；每组选择最小 rank 为 leader。
3. leader 创建 SysV segment，按策略绑定 NUMA，注册本进程映射，调用 adapter 的 `populate(views)`。
4. 交换每组的共享段描述符；同组 follower 挂接同一段，并为各自映射做 CUDA host registration。
5. 协调挂接／注册结果，返回本进程的 `SharedArenaAllocation` owner。

实际使用中应保留 coordinator 的分阶段错误传播；不要让一个 rank 在 peers 已进入 collective 后单独返回。它处理可协调的局部失败，不保证硬退出或通信组本身失效时仍能正常完成 collective。

| scope | 分组规则 |
|---|---|
| `host` | 同 host、IPC namespace、weight signature 一份完整共享权重 |
| `numa` | 上述边界内，每个参与的 GPU NUMA node 一份完整共享权重 |
| `auto` | 每个 host／IPC／signature cohort 独立判断；该组 NUMA 全部已知时用 NUMA，否则用 host |

NUMA 是复制完整共享权重到多个内存域，不是将权重按 rank 切片。显式 `numa` 遇到未知拓扑会失败；`shared_cpu_weight_strict_numa` 控制内存绑定失败策略，不负责把未知拓扑改成 host。host 也不能跨 IPC namespace 共享同一 SysV 段。

`SharedWeightManifest.from_tensors()` 定义对齐布局；现有 adapter 通常用 4096 字节对齐。`tensor_views()` 在 segment 上构造 tensor view。`shared_cpu_weight_register_chunk_mb` 是注册区域的目标分块大小，不是 arena 容量；tensor-aware 划分避免切断单个 tensor，大 tensor 对应注册区域可以超过该值。

NUMA 内存策略要在首次触页／注册前建立。普通 `pin_memory()` 或 `Tensor.share_memory_()` 不能直接替代这套“共享段＋每进程 CUDA 注册＋拓扑策略”的契约。

## 从共享 view 到算子

[shared_weight_map.py](../../../../lightx2v/common/offload/shared_weight_map.py) 将私有和共享权重分开管理：

- `SharedWeightViewMap.take()` 对私有值执行消费移除；对共享值记录已消费，但保留它给其他消费者。
- `consume_weight()` 返回 `(tensor, is_shared)`，兼容原普通 dict 的消费方式。
- `validate_shared_operator_views()` 检查 manifest 全部被消费，并验证最终算子持有共享存储。

`lightx2v/common/ops/utils.py:create_default_tensors()` 在共享分支直接保留 tensor 或 `.t()` view，在私有分支调用原有 pinned tensor 构建逻辑。算子的 `state_dict()` 应暴露 CPU pin source；`load_state_dict()` 将它复制到已有 GPU buffer。审计目标模型使用的全部算子，包括 embedding、普通 tensor、norm、量化 scale 和模型专用权重路径。

当前验证器的指针关系是：

```text
expected_ptr = arena.address + spec.offset + spec.storage_offset * spec.itemsize
```

同时验证 CPU device、dtype、shape／stride 及 `is_pinned()`。普通二维权重可接受原布局或转置布局；若算子明确要求某种方向，应像 Qwen 一样增加语义校验，防止方阵因 shape 相同而漏检。

共享不代表零成本 GPU 使用：CPU 绑定无额外 payload 复制，H2D 仍是各 rank 的私有设备复制。共享 CPU 存储按不可变契约使用，当前机制不等于操作系统强制只读映射；in-place dtype／量化／LoRA 更新必须单独设计，不能写入共享源。

## 两种 GPU 缓冲调度

### WeightModule 与 stream swap

[manager.py](../../../../lightx2v/common/offload/manager.py) 的 `WeightAsyncStreamManager`：

- `init_cuda_buffer()` 注册已有 staging 对象；`init_first_buffer()` 准备首块并更新初始化标志。
- `prefetch_weights()` 在加载 stream 中复制下一块到备用 buffer。
- 当前块在 compute stream 计算；`swap_blocks()` 等待相关工作后交换两个 buffer 的角色。

CPU 保存全部 block 权重，GPU 对同构 block 族通常只保留两个 slot。不可变权重无需每步从 GPU 写回 CPU。非 block 权重、激活、workspace、KV cache 和其他组件的显存另算。

首块与末块策略属于调用方协议：有的模型跨 step 预取回 block 0，有的每次循环重新初始化。跟踪 `need_init_first_buffer` 和实际循环，不将其中一种策略机械复制到全部模型。compute stream 还必须等待调用方产生的输入；返回前也要建立调用方使用输出所需的依赖。

### 固定 slot 与 ready/free 事件

[event_manager.py](../../../../lightx2v/common/offload/event_manager.py) 的 `EventSlotWeightAsyncStreamManager`：

```text
prefetch_to_slot(slot, block)
  → load stream 等待该 slot 上次 free
  → H2D → 记录 ready
wait_ready(slot, caller_compute_stream)
  → compute stream 等待 ready → 执行 block
record_free(slot, caller_compute_stream)
  → 记录计算完成依赖，允许后续覆盖
```

`record_free()` 必须在最后一个消费该 slot 的计算之后记录；Python 侧 pending 被清除不意味着 GPU 已完成。调用方使用实际执行计算的 stream，不能依赖不匹配的默认 stream。

`reset_slots()` 只清理记账。连续请求、VAE tile 或阶段重入前，先用 completion event／同步完成旧工作依赖，再 reset。事件调度可减少不必要等待，但不保证总延迟一定下降，须实测。

## 原生 nn.Module 与生命周期

[module_adapter.py](../../../../lightx2v/common/offload/module_adapter.py) 提供：

| 工具 | 用途与限制 |
|---|---|
| `module_tensors()` | 枚举参数与 buffer，保留 tied name 和 nonpersistent buffer，避免遗漏仍引用旧设备的别名 |
| `assign_module_tensor()` | 显式绑定 Parameter／buffer，不自行管理同步或 owner |
| `ModuleCPUWeights` | 保存 CPU source；`activate()` 将指定子模块移至 GPU；`restore()` 恢复原 source |
| `NativeModuleBlockSource` | 用 detach view 暴露 CPU 权重，不克隆其 storage |
| `NativeModuleBlockSlot` | 复制 meta 骨架并分配 GPU tensor，通过 `load_state_dict()` 复制权重；避免深拷贝完整 CPU block |

模型负责验证一个 slot 可服务的 block schema，并保留算子需要的 stride。动态 buffer 不是因为被 `module_tensors()` 枚举就自动进入共享 manifest；它是否可共享取决于不可变性和 checkpoint 契约。

生命周期顺序：

```text
创建 owner → CPU view 绑定 → 初始化／重建 GPU slot → 异步复制与计算
  → 等待阶段工作结束 → 可释放 GPU slot（CPU view 与 owner 保留）
  → 下次请求可重建 slot
  → 最终停止所有访问并完成 DMA → unregister → detach
```

删除临时 mapping 不关闭 owner；owner 要由实际持有者保存。释放 GPU slot 不应重读 checkpoint 或销毁共享 CPU 源。原生模块在阶段结束时恢复原 CPU view，避免通过 `.cpu()` 创建新私有副本。

`CudaHostRegistration.close()` 在同步或 unregister 失败时保留必要状态以便重试；不能强行 detach 仍被设备使用的页。SysV 自动移除标记与最终 detach 的行为由公共实现负责，不能用全局 `ipcrm` 清理其他任务的共享段。

compile 如参与任务，应将复制和调度留在计算图外，并按 staging 对象生命周期处理编译缓存；不用新 offload manager 绕开原 compile／warmup 生命周期。

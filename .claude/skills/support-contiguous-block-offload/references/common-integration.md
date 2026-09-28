# 公共接入

简体中文 | [English](common-integration_en.md)

## 源码索引与调用链

| 文件 | 主要职责 |
|---|---|
| [weight_module.py](../../../../lightx2v/common/modules/weight_module.py) | `register_offload_group`、`iter_offload_groups`、`named_weight_leaves` 和加载计划分派 |
| [base_model.py](../../../../lightx2v/models/networks/base_model.py) | 原始 checkpoint 元数据、`_apply_weights`、`_init_offload_manager` |
| [block_loader.py](../../../../lightx2v/common/offload/block_loader.py) | `OffloadGroup`、`BlockLoadPlan`、`prepare_contiguous_groups`、配置与 checkpoint 校验 |
| [block_layout.py](../../../../lightx2v/common/offload/block_layout.py) | `BlockLayout`、`BlockBuffer`、`BlockLoadContext`、`ContiguousBlockTransfer` |
| [manager.py](../../../../lightx2v/common/offload/manager.py) | group 选择、首块加载、预取、双缓冲交换与清理 |
| [weight_storage.py](../../../../lightx2v_platform/ops/weight_storage.py) | `TensorMetadata`、`WeightStorage`、`StorageDescription` |
| [offload.py](../../../../lightx2v_platform/base/offload.py) | `get_block_offload_backend` 和内存操作接口 |

```text
模型 weights 注册 group
  → BaseTransformerModel._apply_weights
  → prepare_contiguous_groups
  → BlockLoadPlan 读取各 leaf 的 describe_storage
  → WeightModule.load → BlockLoadPlan.load
  → BlockBuffer.allocate → BlockLoadContext → load / bind_storage

BaseTransformerModel._init_offload_manager
  → WeightAsyncStreamManager.init_contiguous_groups
  → init_first_buffer → prefetch_weights → ContiguousBlockTransfer.copy
  → 原有 block 计算与 swap_blocks
```

注册 group 不会替代模型的推理循环。模型必须在正确的计算边界使用所选 slot，并保留其原有同步顺序。

## 模型声明

先查模型现有 block 权重与 offload buffer 的构造。当前 `OffloadGroup` 要求非空 CPU blocks、恰好两个 device slots，以及逐 block 对应且以 `.` 结尾的 checkpoint 前缀。

```python
self.register_offload_group(
    "blocks",
    self.blocks,
    self.offload_block_cuda_buffers,
    (f"transformer_blocks.{i}." for i in range(self.blocks_num)),
)
```

这是 Qwen 前缀的示例，按目标 checkpoint 修改。仅在模型实际创建了 block offload slots 的分支注册；不要把 phase buffer 当成完整 block。

- 同一 group 的每个 block 和两个 slots 必须有相同相对算子路径、形状、dtype、转置和 offsets。总字节数相同不代表兼容。
- `TensorSpec.name` 不参与布局相等比较，因此层号不同的 checkpoint 名称可使用同一布局；其余结构仍需一致。
- 异构 block 划分为多个 group。推理在每个 group 边界调用 `init_first_buffer`，再预取该 group。
- 当前 manager 按 blocks 容器身份选择 group，注册和推理应使用同一容器，不在运行时重建临时列表。
- 前缀下的 checkpoint tensors 必须被完整且唯一地描述。共享/tied 权重所有权、多消费者或原生 `nn.Module` 不能假定已兼容此接口；先检查现有表示能力，提出所需扩展。

## 算子存储契约

`TensorMetadata` 区分 checkpoint 原始 dtype 与加载后的 dtype。通过 `WeightStorage(name, attr, shape, dtype, transpose)` 描述最终存储；通过 `StorageDescription.auxiliary` 描述随 block 复制的设备状态。

1. 使用 checkpoint 实际 shape 描述加载视图，用 `transpose` 表达算子的计算视图；不要先复制成另一份转置权重。
2. 在 checkpoint 被统一 cast 前保留元数据。仅检查 cast 后 dtype 无法识别 FP8 文件误用于 BF16 路径。
3. 检查 weight、bias、scale 和其他运行时状态，不能只描述矩阵。量化格式和 scale 的布局、转换规则由所属算子提供，不由模型名推断。
4. 优先让现有 `load()` 通过公共工具消费 `BlockLoadContext`。需要特殊绑定的算子实现 `bind_storage()`，保持 buffer 属性及计算属性对应。
5. 无状态算子可以给出空状态，但仍保留其初始化 `load()`；未知有状态算子明确报错。

平台的普通浮点 RMSNorm/LayerNorm 默认继承 [norm 模板](../../../../lightx2v_platform/ops/norm/norm_template.py) 的存储描述和绑定；新芯片子类只需实现计算。公共加载器优先调用 `bind_storage()`，因此子类若在 `load()` 中增加转换或辅助状态初始化，也必须明确适配连续绑定，不能假定这些步骤会自动执行。

标准逐通道预量化 MM 使用 [MMWeightPerChannelQuantTemplate](../../../../lightx2v_platform/ops/mm/template.py)：子类声明 `checkpoint_dtype`、`weight_need_transpose` 和计算内核，即可继承权重校验、FP32 scale、可选 bias 和连续绑定。NPU INT8、MLU INT8、MUSA FP8 已共用此模板。打包或需额外转换的格式保留自身存储契约，不直接套用该模板。

平台若提供标准 PyTorch pinned allocation、typed view、异步 copy、stream/event 接口，可像 MLU/MUSA 一样声明 `block_offload_backend = TorchBlockOffload`，不另建同构 backend。平台的特殊 D2H/H2D 规则通过设备类的 `copy_to_cpu` / `copy_transposed_weight_to_device` 提供，公共工具按注册平台选取，缺省保持原生复制。声明支持仍需目标芯片实测；已有模型算子和运行环境也必须可用，配置不能补齐缺失的内核。

适配参考：[MM](../../../../lightx2v/common/ops/mm/mm_weight.py)、[RMSNorm](../../../../lightx2v/common/ops/norm/rms_norm_weight.py)、[LayerNorm](../../../../lightx2v/common/ops/norm/layer_norm_weight.py)、[DefaultTensor](../../../../lightx2v/common/ops/tensor/tensor.py)。不要为每个算子创建内容相同的转发函数。

## 加载与内存

`prepare_contiguous_groups` 在任何 block 消费 checkpoint 前完成所有计划及一致性检查。原始 metadata 缺失时只能描述当前 tensor，不能据此证明转换前精度正确。

`BlockBuffer` 用一维连续 `uint8` 存储容纳混合 dtype，按当前 `ALIGNMENT_BYTES` 对齐 tensor 起点，padding 只初始化一次。`view()` 建立 typed views；`BlockLoadContext.take()` 将 CPU 源复制到最终视图并消费源项，设备侧只绑定视图。

加载后检查计划与实际的地址、设备、dtype、shape、stride、pin 状态和状态项。不要在绑定后无条件调用 `clone()`、`contiguous()` 或重新 pin，造成 view 脱离计划存储。加载上下文引用 checkpoint，用完后不应被长期保留。

此路径仍可能在加载阶段持有 checkpoint 临时 tensor，不应宣传为磁盘直接读入最终存储或零加载峰值。减少推理阶段 pack 与减少加载峰值是不同目标。

## 传输与生命周期

`ContiguousBlockTransfer.copy()` 对主体 block storage 发起整体复制，辅助设备状态单独复制。保持源权重只读，CPU 存储和设备 views 在所有使用者完成前存活。

- 首块初始化完成后才能计算；预取目标 slot 不能仍被上一 block 使用。
- `swap_blocks` 和 group 切换负责调度同步。transfer 的初始化 ready event 与 `record_stream` 不替代每次 slot 复用依赖。
- 下一 step、CFG 的下一分支和连续请求仍从正确的首块开始，不能只证明一次 block 遍历正确。
- 清理覆盖所有注册 group，不只是当前 active transfer；初始化失败的部分资源也需要可释放。
- 接口中 `cuda_buffers`、`create_cuda_buffer` 等是已有命名，实际设备由平台决定；不为命名统一改写所有调用者。

当前配置边界以 `validate_contiguous_config()` 为准：单 rank、block 粒度、静态私有权重、eager、NoCaching、敏感层与主精度一致。共享、lazy load、并行、LoRA/adapter、compile/graph、动态量化等组合目前有显式限制。不要静默关闭用户配置来通过验证，也不要把这些限制写成未来永远无法支持的原则。

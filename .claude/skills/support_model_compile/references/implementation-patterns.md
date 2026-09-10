# LightX2V Compile Implementation Patterns

实现代码或选择缓存键时阅读本文件。路径反映当前仓库；先用 `rg` 确认，不要机械复制。

## 目录

- [参考入口](#参考入口)
- [公共 block 接入](#公共-block-接入)
- [多种 block 族](#多种-block-族)
- [Wan：保留逻辑层号](#wan保留逻辑层号)
- [Qwen-Image：复用 staging identity](#qwen-image复用-staging-identity)
- [LTX2/LTX2.3：显式传递 guider 分支](#ltx2ltx23显式传递-guider-分支)
- [Lingbot-Video：normal MoE](#lingbot-videonormal-moe)
- [Offload 接入](#offload-接入)
- [第三方算子叶子](#第三方算子叶子)
- [自定义算子内层编译与 Triton 动态标量](#自定义算子内层编译与-triton-动态标量)
- [测试骨架](#测试骨架)

## 参考入口

| 用途 | 位置 |
|---|---|
| 公共 block compile/cache | `lightx2v/common/transformer_infer/transformer_infer.py` |
| Wan normal/offload | `lightx2v/models/networks/wan/infer/transformer_infer.py`、`lightx2v/models/networks/wan/infer/offload/transformer_infer.py` |
| Qwen-Image normal/offload | `lightx2v/models/networks/qwen_image/infer/transformer_infer.py`、`lightx2v/models/networks/qwen_image/infer/offload/transformer_infer.py` |
| LTX2 normal/offload/AR | `lightx2v/models/networks/ltx2/infer/transformer_infer.py`、`lightx2v/models/networks/ltx2/infer/offload/transformer_infer.py`、`lightx2v/models/networks/ltx2/infer/ar_transformer_infer.py` |
| Lingbot-Video normal/MoE | `lightx2v/models/networks/lingbot_video/infer/transformer_infer.py` |
| Qwen kernel guard | `lightx2v/models/networks/qwen_image/weights/transformer_weights.py` |
| custom-op 叶子 | `lightx2v/common/ops/rope/flashinfer_rope.py` |
| RMSNorm compile 分流 | `lightx2v/common/ops/norm/rms_norm_weight.py` |
| 单测 | `test_cases/test_transformer_compile.py`、`test_cases/test_ltx2_compile.py` |

## 公共 block 接入

```python
class NewTransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        ...
        self.init_compile(config)

    def infer_block(self, block, x, inputs):
        ...
        return x

    def infer_blocks(self, blocks, x, inputs):
        for block_idx, block in enumerate(blocks):
            x = self.run_block(block_idx, block, x, inputs)
        return x
```

公共 `run_block()` 只分派：

```python
def run_block(self, block_idx, block, *args):
    if self.use_compile:
        return self.get_compiled_block(block_idx, block)(*args)
    return self.infer_block(block, *args)
```

不要设置层号，也不要增加单一：

```python
self.compiled_block = torch.compile(self.infer_block, dynamic=None)
```

单入口容易让所有 block、guider 和可变状态共享 guards。使用公共 per-key cache；缓存命中同时校验对象身份。

## 多种 block 族

先按签名和语义分族。例如具有 double-stream 与 single-stream block 的模型会使用不同方法、返回值和 offload manager。

- 每个族使用独立分派器和缓存，或把 family 加入 key；
- normal 可用 `(family, block_idx)`，无层号依赖的 offload 可用 `(family, id(block))`；
- 两个族的 `block_idx=0` 不能命中同一 cache；
- 不要用大量可选参数把不同签名伪装成一种 block；
- 只有形成跨模型复用时才扩展公共 base，否则保留短小的模型专用 dispatcher。

这类模型仍按同一流程检查 layer state、staging identity、算子兼容性和 warmup，只是编译入口不止一个。

## Wan：保留逻辑层号

Wan attention 会把 `self.block_idx` 传给下游，多个子类还用它索引 KV cache、adapter 和层级策略。

- 在拥有循环的方法中，于 `run_block()` 前设置 `self.block_idx`；
- normal 和 block offload 都按 `block_idx` 缓存；
- phase 读取层号，按 `(block_idx, phase_idx)` 缓存；
- 为不同 phase 类型提供稳定 callable；
- 特殊 post-adapter phase 没有验证时保持 eager。

即使 offload 只使用少量 staging buffer，也不要直接改为 `id(block)`；除非先移除整个图内的层号依赖。

## Qwen-Image：复用 staging identity

Qwen-Image block 和四类 phase 不读取逻辑层号。

- normal 使用默认 `block_idx`；
- block offload 返回 `id(block)`，cache 数量等于 staging block 数；
- phase offload 按 `phase_idx` 缓存四个入口；
- lazy block/phase 自然复用同一 cache；
- 不设置 `self.block_idx`。

Qwen-Image 当前拒绝 `use_compile=true + one-pass RMSNorm`。配置为 `sgl-kernel` 时仍需检查实际 compile 分支，不能只看 backend 名称。

## LTX2/LTX2.3：显式传递 guider 分支

当前参考路径支持 normal、model offload 和 block offload，不因 compile 新增 phase offload。

- normal/model 使用默认 `block_idx`；
- block offload 不读取层号，使用 `id(block)`；
- 把 guider 状态在循环中计算成有限布尔参数：

```python
self.run_block(
    block_idx,
    block,
    vx,
    ax,
    inputs,
    block_idx in self._mm_skip_video_self_blocks,
    block_idx in self._mm_skip_audio_self_blocks,
    self._mm_skip_a2v,
    self._mm_skip_v2a,
)
```

让 `infer_block()` 接收含义明确的 `skip_*` 参数，避免单一入口读取变化的 Python set。

LTX2 self-attention 的 `cu_seqlens` 依赖请求 shape。compile 时在进入 block 循环前创建，下一请求按新 video/audio 长度刷新，避免首层 mutation 改变后续 guards。

LTX2 AR 使用层号索引 KV cache。只在 AR 子类的 `run_block()` 中设置 `_ar_block_idx` 后调用 `super()`；不要把它提升到公共 base。

## Lingbot-Video：normal MoE

当前参考路径只支持 normal，直接复用公共 `block_idx` cache。初始化时固定 compute dtype 等 Python 配置，避免在 block 图内调用环境 helper；MoE 的 `torch._grouped_mm` 仍是外部 kernel，其 workspace 和路由工作量需要用代表性 token/step 验证。不要因 compile 适配顺手增加原路径不支持的 offload/lazy。

## Offload 接入

block offload 仅替换 compute 调用：

```python
with torch_device_module.stream(self.offload_manager.compute_stream):
    current_block = self.offload_manager.cuda_buffers[0]
    x = self.run_block(block_idx, current_block, x, inputs)
```

保持 `init_first_buffer → prefetch → compute → swap` 的原顺序。lazy-load 继续复用相同 staging 对象；如果请求间会重建对象，公共 cache 的 identity 校验会刷新 callable。

phase offload 保持三层职责：

```text
infer_phase         选择并执行 eager phase
get_compiled_phase  按 key 创建/复用 callable
run_phase           在 eager/compile 间分派
```

不要复制 prefetch/swap 循环来维护“compile 版 offload”。

## 第三方算子叶子

需要阻止 Dynamo 进入第三方 Python 时，在模块作用域注册：

```python
@torch.library.custom_op(
    "lightx2v::new_op",
    mutates_args=(),
    device_types="cuda",
)
def new_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return external_kernel(x, weight)


@new_op.register_fake
def new_op_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)
```

有原地 mutation 时准确声明 schema 和 `mutates_args`。在 `apply()` 中分派：

```python
if torch.compiler.is_compiling():
    return new_op(x, weight)
return external_kernel(x, weight)
```

用最小输入验证数值、dtype、shape 和真实 kernel。custom op 不会减少调用次数；需要减少 launch 时应设计更大粒度的融合算子。

边界应覆盖包含 data-dependent 中间结果的最小完整语义单元。当中间结果会迫使 Dynamo 追踪其数据依赖时，block map、增量 LUT 和消费该 LUT 的 attention kernel 应共同留在边界内，不能只把最后一个 kernel 设为叶子；fake 实现只描述该单元对外可见的输出元数据。

## 自定义算子内层编译与 Triton 动态标量

custom op 让外层 Dynamo 只看到一个算子，但函数体仍会正常执行。其内部的 Triton、CUDA extension 和第三方 wrapper 有各自的 JIT 与缓存，因此外层没有 recompile 日志，不代表正式请求没有发生内层编译。服务 ready 前后分别记录内层编译日志或缓存产物，再用冷缓存完整生命周期复测；不要仅凭 `TORCH_LOGS` 判断。

多模态路径按实际执行顺序比较签名：

```text
原始模态 token
→ packing / padding / alignment
→ TP / SP 切分
→ 各 rank 的实际 q/k 长度
→ sparse block 数和 leaf kernel 标量参数
```

prompt 长度可任意变化时，不要靠枚举 prompt 预编译所有长度。先判断 Triton 标量是否必须静态：

- 标量只参与指针偏移、mask 或运行时算术，且 launch grid 可在 JIT 外计算时，可以作为运行时参数；block size 等静态元参数仍保留 `tl.constexpr`。
- 标量决定 `tl.arange` 范围、静态 tensor shape、constexpr 分支、unroll、layout 或 `num_warps` 时，不能直接动态化；应重新划分参数，或使用明确且有限的 bucket/padding。
- 对确认安全的长度参数，使用 `do_not_specialize` 禁止按运行时值和对齐关系特化，避免每个长度或整除关系产生新变体：

```python
@triton.jit(do_not_specialize=("seq_len",))
def kernel(x, seq_len, block: tl.constexpr):
    ...
```

不要批量套用该装饰器。至少用两个不同长度（包含非整除 tail）验证数值，并确认 ready 后不再产生新的内层编译产物，稳态性能也没有回退。

## 测试骨架

mock `torch.compile` 为原函数，测试 Python cache 语义：

```python
with patch.object(common_module.torch, "compile", side_effect=lambda fn, dynamic=None: fn) as compile_mock:
    infer.run_block(0, block, *args)
    infer.run_block(0, block, *args)

assert compile_mock.call_count == 1
```

至少覆盖默认逻辑层键、staging identity、同键换对象、eager bypass、phase cache、guider 分支和 lazy prefetch/swap 顺序。单测验证分派，真实模型测试验证 Dynamo、kernel、数值和性能。

---
name: model-enable-intel-xpu
description: 对照当前 LightX2V 平台、模型、权重和内核实现，接入或诊断 Intel XPU 推理；核对算子契约、offload 支持、量化、构建与运行证据。
---

# Intel XPU 新模型接入 & 崩溃排查

LightX2V Intel XPU 模型接入与推理诊断。本文的 Arc 140V 故障经验只作排查线索，不能替代当前设备、驱动和模型的复现。仓库路径以根目录为基准。

先看 [XPU 使用说明](../../../scripts/platforms/intel_xpu/README_zh.md)、[执行入口](xpu-exec/SKILL.md) 和 `lightx2v_platform/base/intel_xpu.py`。已有模型优先复用其 runner/weights/infer，仅补实际缺失的平台路径。

---

## Quick Start

**接入新模型**：
```
"Use model-enable-intel-xpu skill. 接入新模型 {ModelName}，DiT ~XX GB BF16，XX transformer blocks。
Model path: /path/to/{ModelName}"
```

**排查崩溃**：
```
"Use model-enable-intel-xpu skill. 进程静默消失，无任何输出。Stage: 权重加载。文件大小: ~XX GB"
"Use model-enable-intel-xpu skill. OOM at second infer. Error: Tried to allocate XX MiB."
"Use model-enable-intel-xpu skill. 推理中途日志截断，Stage: 文本编码器 forward"
```

---

# 常见场景

## Case 1: 接入新模型到 Intel XPU

**关键决策**：已实现的 `cpu_offload` 可降低权重驻留；block 双缓冲通常保留两个 block，但还需非 block 权重、激活、workspace、编码器和 VAE 的内存。只有模型实现了磁盘加载链才可启用 `lazy_load`。量化不是跑起来的必要条件，而是"缩小体积 + 减少每步搬运量"的可选优化。真正决定要不要量化的是**内存 profile**，不是单看 DiT 体积：

| 硬件 profile | 内存特点 | 建议 |
|---|---|---|
| 独立显卡 + 大内存服务器（如 Arc Pro B60） | 显存和系统内存分离，系统内存通常远大于模型体积 | 先评估 CPU 常驻权重和设备峰值；MiniMax H3 有 BF16 block offload 配置（`configs/platforms/intel_xpu/minimax_h3_t2av.json`，无 `dit_quantized`） |
| Arc 140V 等统一内存设备 | CPU/XPU 与 OS 共享物理内存，容量以实际机器为准 | 同时测量组件驻留和系统峰值；仅在模型支持时评估量化或 lazy loading |

**说明**：量化是可选的性能/内存优化，不是运行的先决条件。INT8 与 FP8 的精度、吞吐和额外激活量化开销应在同一模型与设备上比较。MiniMax H3 当前拒绝 `lazy_load`，仅有 model/block offload，不能套用下方 Wan 的 lazy/phase 经验。

→ 完整 8 步流程见[接入新模型：完整流程](#接入新模型完整流程)

---

## Case 2: 进程静默消失，无任何输出

先检查 XPU 环境、动态库和退出前的最后日志，按实际失败阶段定位。

详见 [Case 2: 进程静默消失，无任何输出](references/troubleshooting.md#case-2-进程静默消失无任何输出)。

## Case 3: 推理中途静默 kill，日志截断

区分主机 OOM kill 与设备/扩展崩溃；用内存和退出证据定位，不凭日志截断猜测。

详见 [Case 3: 推理中途静默 kill，日志截断](references/troubleshooting.md#case-3-推理中途静默-kill日志截断)。

## Case 4: OOM（内存不足）

分别检查编码器加载、连续请求状态和量化中间态的峰值内存。

详见 [Case 4: OOM（内存不足）](references/troubleshooting.md#case-4-oom内存不足)。

## Case 5: KeyError / AttributeError 权重加载失败

核对真实 checkpoint key、转换映射和权重类继承；修正契约所在层。

详见 [Case 5: KeyError / AttributeError 权重加载失败](references/troubleshooting.md#case-5-keyerror--attributeerror-权重加载失败)。

## Case 6: NoneType / Stream / 注意力限制崩溃

核对平台能力、注意力语义和 stream 接口；配置字段不能替代缺失实现。

详见 [Case 6: NoneType / Stream / 注意力限制崩溃](references/troubleshooting.md#case-6-nonetype--stream--注意力限制崩溃)。

## Case 7: TP 报错 `torch.addmm` 维度不匹配（列并行 bias 未切分）

检查列并行 weight 与 bias 是否一致切分，并对照实际 TP 形状。

详见 [Case 7: TP 报错 `torch.addmm` 维度不匹配（列并行 bias 未切分）](references/troubleshooting.md#case-7-tp-报错-torchaddmm-维度不匹配列并行-bias-未切分)。

# XPU 平台约束

## 统一内存、Stream 与 pinned memory

统一内存设备的 CPU/XPU 会竞争整机物理内存。以实际设备属性、进程和系统峰值为准，offload 不保证任意大小的模型可运行。

- 复用 `WeightAsyncStreamManager` 和模型 XPU 分支。当前 `_sync()`、`swap_blocks()`、`swap_phases()` 对 XPU 使用设备同步；不要重复增加同步或直接套 CUDA compute-stream 循环。
- 异步 H2D 要检查 pinned source、`non_blocking`、stream 依赖与源存活期。实际重叠和带宽必须测量，不能从两个开关直接推导。
- 组件卸载使用 runner/encoder/VAE 的现有路径；不要假定 `unload_modules` 与 warmup、lazy 或所有模型均兼容。

## INT8 DiT 权重

INT8 与 FP8 的收益需分别验证。**不是**反量化成 float 后走普通 matmul（那样等于退化成 bf16 matmul，没有性能收益）——`MMWeightInt8IntelXpu`（`lightx2v_platform/ops/mm/intel_xpu/mm_weight.py`）走的是动态 per-token W8A8：激活先在线量化成 int8，再调用真正的 INT8×INT8 GEMM 内核：

```python
def apply(self, input_tensor):
    # 激活 + 权重均为 int8，走真正的 INT8 GEMM kernel，而非反量化后 matmul
    output = sycl_kernels.onednn_w8a8_int8(
        input_tensor.reshape(-1, input_tensor.shape[-1]).contiguous(),
        self.weight.contiguous(),                        # int8, [out_dim, in_dim]
        self.weight_scale.reshape(-1).float().contiguous(),
        self.bias,
    )
    return output.reshape(*input_tensor.shape[:-1], self.weight.shape[0])
```

**关键点**：INT8 scale 为每行（per-channel）保存（shape `(out_dim, 1)`），与 FP8 一致；`int8-intel-xpu` 要求激活为 FP16/BF16，且需要 `lightx2v_kernel/intel_xpu` 编译出 `sycl_kernels.onednn_w8a8_int8`（见下方"前置条件"）。

---

## torch.compile 与融合算子（MiniMax H3 XPU 配置）

在 CUTE attention / INT8 GEMM 之上，MiniMax-H3 XPU 配置又叠加了三项可独立开关的性能优化，新模型稳定跑通后可评估是否适用：

**DiT torch.compile**：`"use_compile": true`。首次调用可能包含 JIT 编译；结合日志和 trace 确认进度，不按固定等待时间判定正常。

**融合 QKV Norm + RoPE**（省掉 QKV norm/RoPE 之间的中间 tensor 往返）：
```json
{
  "use_fused_qkv": true,
  "use_fused_qkv_norm_rope": true,
  "qkv_norm_rope_type": "intel_xpu"
}
```
XPU 专用后端在 `lightx2v_platform/ops/norm/intel_xpu/xpu_qkv_norm_rope.py`，注册进 `QKV_NORM_ROPE_REGISTER`；没有 XPU 专用实现前可先用通用 Triton fallback（`qkv_norm_rope_type: "triton"`，实现见 `lightx2v/models/networks/minimax_h3/infer/fused_qkv.py`），语义上与逐步计算的 RMSNorm+RoPE 等价（FP32 中间量保精度）。

**VAE CUTE Attention + Compile**：
```json
{
  "vae_cpu_offload": false,
  "vae_use_compile": true,
  "vae_attn_type": "minimax_h3_xpu_cute"
}
```
`vae_attn_type` 除通用的 `torch_sdpa`/`sage_attn2` 外新增 XPU 专用的 `minimax_h3_xpu_cute`（D64 CUTE kernel，`lightx2v_platform/ops/attn/intel_xpu/minimax_h3_xpu_cute.py`）。用它时通常把 VAE 从 `vae_cpu_offload: true` 切回 `false`（常驻 XPU）+ `vae_use_compile: true` 一起开，前提是显存够用。**注意**：VAE 的 `shared`/`block` 权重 offload 模式与 `vae_use_compile` 互斥，两者不能同时开。

---

## 多卡分布式推理

当前 `IntelXpuDevice.init_parallel_env()` 使用 `backend="xccl"`。非 Linux 系统仅在 PyTorch `< 2.10.0+xpu` 时被代码拒绝；通过版本检查不等于目标 Windows/驱动组合已验证。以下是 Linux 示例。

**进程数和并行大小关系**：
```
进程数 = tensor_p_size × seq_p_size × cfg_p_size
```

**启动命令（Linux）**：
```bash
export ZE_AFFINITY_MASK=0,1,2,3          # 选择卡 0-3
export PLATFORM=intel_xpu
export PYTHONFAULTHANDLER=1

# TP（张量并行）4卡
torchrun --standalone --nproc_per_node=4 -m lightx2v.infer \
    --config_json configs/platforms/intel_xpu/dist_infer/minimax_h3_t2av_tp.json ...

# SP+TP（序列+张量并行）8卡
export ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7
export CCL_SYCL_ALLTOALL_ARC_LL=1
export CCL_SYCL_ALLTOALL_TMP_BUF=1
export CCL_SYCL_CCL_BARRIER=1
export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296

torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
    --config_json configs/platforms/intel_xpu/dist_infer/minimax_h3_t2av_sp_tp.json ...
```

**并行策略**：
- **TP**：模型不适合单卡或需要分割 linear layer
- **SP**：长视频或高分辨率，使用 Ulysses all-to-all 通信
- **CFG**：两组设备分别计算正负条件分支（通常 `cfg_p_size=2`）
- **SP+TP**：先单独验证 TP 和 SP，再组合

**要点**：
- `ZE_AFFINITY_MASK` 选择卡（不用 `CUDA_VISIBLE_DEVICES`）
- CCL 变量来自现有示例的调优设置，不是所有 XCCL 环境的必需值；按实际 backend、版本和通信测试确认
- 多卡不一定线性加速，小模型/低分辨率可能通信开销更大
- 文本编码器/VAE 可能仍在所有卡上复制，不能简单按卡数除以内存

---

## 长序列稀疏注意力（Turbo-SLA / `dynamic_sparse_attn`）

长视频/长序列场景下，MiniMax-H3 在 XPU 上支持稀疏注意力路由（SLA）：只计算 top-k block，比全量 CUTE attention 更快，通常配合专门蒸馏的 Turbo-SLA LoRA 权重、少数几步（如 4-5 步）推理使用：

```json
{
  "attn_type": "dynamic_sparse_attn",
  "refiner_attn_type": "intel_xpu_cute_attn",
  "dynamic_sparse_attn_setting": {
    "sparsity_ratio": 0.85,
    "operator": "intel_xpu_cute_attn"
  }
}
```

- **主 transformer**：`attn_type: dynamic_sparse_attn`，`operator: intel_xpu_cute_attn` 时走 XPU 版 SLA block-map + sparse-block attention；`sparsity_ratio` 越高跳过的 block 越多、越快，可能损失精度（`0.85` 是 Turbo-SLA 4-step 配置验证过的值，不是通用默认）。
- **refiner**（早期少数精度敏感的 step/block）单独用 `refiner_attn_type: intel_xpu_cute_attn` 走稠密 CUTE attention，不做稀疏化。
- XPU 实现依赖 `sycl_kernels` 新增的 `sla_block_map` / `sparse_block_attention` / `sla_sparse_attention` 三个 API（封装在 `lightx2v_platform/ops/attn/intel_xpu/xpu_sla_attn.py`），编译 `sycl_kernels` 时需要打开 SLA/CUTE sparse attention 支持，否则运行时报 `RuntimeError: sycl_kernels.xxx is unavailable`。
- 该路径是长序列专用优化，不是常规接入新模型的必需项。

---

## INT8 文本编码器

仅对已实现该方案的文本编码器使用。按模型 wrapper 核对量化格式、加载与 forward；不存在适用于所有编码器的自动 INT8 阈值。设计在线量化时可参考：

1. `init_empty_weights` 建模型骨架（0 RAM，全是 meta tensor）
2. `_replace_linear_int8` 递归替换所有 `nn.Linear`（仍 0 RAM）
3. 逐 shard `struct+readinto` 读取，fp16 → int8 即时量化赋值，peak RAM ≈ 单 shard

`DequantLinearInt8.forward` 关键（两步 cast，绕过 oneDNN SIGABRT）：
```python
def forward(self, x):
    w = self.weight.to(torch.float32) * self.weight_scale  # int8→fp32
    return F.linear(x, w.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)
```

**non-persistent buffer**：`init_empty_weights` 后 `register_buffer(..., persistent=False)`（如 RoPE `inv_freq`）仍是 meta tensor，加载权重后需手动重算。

---

# 接入新模型：完整流程

## 第一步：读取原始权重，确定架构参数

读取源权重、配置与前向代码，确定层数、维度、head 和关键张量布局。

详见 [第一步：读取原始权重，确定架构参数](references/model-integration.md#第一步读取原始权重确定架构参数)。

## 第二步：INT8/FP8 转换

优先扩展 tools/convert/converter.py，保留模型所需 FP32 等非量化精度；save_by_block 不等于流式加载。内存受限的原始模板见 [流式转换参考](references/streaming-conversion.md)。

详见 [第二步：INT8/FP8 转换](references/model-integration.md#第二步int8fp8-转换)。

## 第三步：补齐模型文件骨架

沿用最近模型族的 runner/model/weights/infer/scheduler 骨架，明确每层职责。

详见 [第三步：补齐模型文件骨架](references/model-integration.md#第三步补齐模型文件骨架)。

## 第四步：实现 `weights/transformer_weights.py`

按真实 key 注册权重算子，核对构造签名、量化参数及已有 offload 约束。

详见 [第四步：实现 `weights/transformer_weights.py`](references/model-integration.md#第四步实现-weightstransformer_weightspy)。

## 前置条件：编译 sycl_kernels

使用仓库根目录 lightx2v_kernel/intel_xpu；历史 Windows 包仅供比较。按目标 GPU、oneAPI、oneDNN 与 PyTorch ABI 构建并验证。

详见 [前置条件：编译 sycl_kernels](references/model-integration.md#前置条件编译-sycl_kernels)。

## 第五步：实现推理类

复用平台算子并保持注意力、RoPE、归一化及 CFG 的真实计算语义。

详见 [第五步：实现推理类](references/model-integration.md#第五步实现推理类)。

## 第六步：实现 `model.py`

由模型组织权重与 infer，保持模型族的组件加载和生命周期。

详见 [第六步：实现 `model.py`](references/model-integration.md#第六步实现-modelpy)。

## 第七步：创建 Runner 并注册

注册 runner 并加入 runner_factory.RUNNER_MODULES，声明 task 对应的请求字段和 InputInfo。

详见 [第七步：创建 Runner 并注册](references/model-integration.md#第七步创建-runner-并注册)。

## 第八步：选择 XPU Config 并验证

选用模型匹配的 XPU 配置，先验证单卡，再验证任务要求的多卡和连续请求；报告数值及环境证据。

详见 [第八步：选择 XPU Config 并验证](references/model-integration.md#第八步选择-xpu-config-并验证)。

# 快速参考

## 多卡快速检查

```bash
# 1. 验证设备可见
python -c 'import torch; print(torch.xpu.is_available(), torch.xpu.device_count())'

# 2. 计算进程数 = TP_SIZE × SP_SIZE × CFG_SIZE
# 3. 设置 ZE_AFFINITY_MASK，按当前通信环境评估示例 CCL 调优变量
# 4. 检查配置文件中的 parallel 部分
# 5. 启动 torchrun --nproc_per_node=<进程数>
```

---

## 移植检查清单

- [ ] runner 注册、任务、请求字段与 `InputInfo` 对齐。
- [ ] checkpoint/key/dtype/scale 与目标加载器一致；转换与 lazy 支持分开确认。
- [ ] 每个目标 offload 模式的 CPU 源、设备 slots、prefetch/swap 与 cleanup 闭合。
- [ ] attention 的 layout、head_dim、causal/mask 与 RoPE 数学契约匹配。
- [ ] sycl_kernels 的实际导出、构建目标与依赖版本正确；最小算子和目标模型分别验证。
- [ ] 连续请求无状态残留；内存不足按组件峰值定位，不用固定 GB 阈值判断。
- [ ] TP 列并行的 1-D bias/scale 随权重切分，行并行 bias 保持正确合并语义。
- [ ] compile、融合算子、稀疏 attention 按需逐项验证，不由开关存在推断支持。

## 调试工具

```python
# 定位退出前的最后位置（stderr + flush）
import sys
print("[debug] before op", file=sys.stderr, flush=True)

# XPU 内存状态
print(f"alloc: {torch.xpu.memory_allocated()/1e9:.2f} GB  "
      f"reserved: {torch.xpu.memory_reserved()/1e9:.2f} GB")
torch.xpu.empty_cache()
```

## 关键文件

| 文件 | 用途 |
|------|------|
| `lightx2v/common/offload/manager.py` | `WeightAsyncStreamManager`：双缓冲核心，Stream 配置 |
| `lightx2v/common/modules/weight_module.py` | `WeightModule` / `WeightModuleList` 基类 |
| `lightx2v/common/ops/utils.py` | `create_cuda_buffers`、`create_cpu_buffers` 和权重复制 |
| `tools/convert/converter.py` | 通用转换 CLI；新模型加 `model_type`：`get_key_mapping_rules()` + `model_type_keys_map` |
| `tools/convert/quant/quant.py` | `CONVERT_WEIGHT_REGISTER`；`QuantWeightINT8`/`QuantWeightFP8`（per-channel 对称量化） |
| `lightx2v_platform/ops/mm/intel_xpu/mm_weight.py` | `MM_WEIGHT_REGISTER`；`MMWeightInt8IntelXpu`；`MMWeightFp8IntelXpu` |
| `lightx2v_platform/ops/norm/intel_xpu/xpu_rms_norm.py` | `PLATFORM_RMS_WEIGHT_REGISTER`（合并进 `RMS_WEIGHT_REGISTER`）；`IntelXpuRMSWeight` |
| `lightx2v_platform/ops/rope/intel_xpu/minimax_h3_rope.py` | `PLATFORM_ROPE_REGISTER`（合并进 `ROPE_REGISTER`）；`MiniMaxH3XpuRope` |
| `lightx2v/models/networks/base_model.py` | `BaseTransformerModel`；`_init_offload_manager` |
| `lightx2v/utils/registry_factory.py` | 所有 Register 注册表 |
| `lightx2v_platform/ops/attn/template.py` | 平台 attention 模板；当前没有 `load_state_dict_from_disk` |
| `lightx2v_platform/ops/attn/intel_xpu/xpu_cute_attn.py` | `IntelXpuCuteAttnWeight`（CUTE 优化注意力） |
| `lightx2v_platform/ops/attn/intel_xpu/xpu_sla_attn.py` | `sla_block_map`/`sparse_block_attention`/`sla_sparse_attention` XPU 封装，供 `dynamic_sparse_attn` 调用 |
| `lightx2v_platform/ops/attn/intel_xpu/minimax_h3_xpu_cute.py` | VAE 专用 D64 CUTE attention，`vae_attn_type: minimax_h3_xpu_cute` |
| `lightx2v_platform/ops/norm/intel_xpu/xpu_qkv_norm_rope.py` | `"intel_xpu"` 融合 QKV Norm+RoPE 后端，注册进 `QKV_NORM_ROPE_REGISTER` |
| `lightx2v/models/networks/minimax_h3/infer/fused_qkv.py` | 融合 QKV Norm+RoPE 的通用 Triton fallback（`qkv_norm_rope_type: "triton"`） |
| `lightx2v/models/networks/minimax_h3/model.py` | TP 权重切分 `_tp_split`：0-D 标量复制，列并行 1-D bias/scale 切分，行并行 bias 复制 |

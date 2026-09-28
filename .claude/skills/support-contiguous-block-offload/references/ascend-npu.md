# Ascend NPU

简体中文 | [English](ascend-npu_en.md)

## 入口与完整推理链

在导入项目之前通过环境选择 `PLATFORM=ascend_npu`，用 `ASCEND_RT_VISIBLE_DEVICES` 指定目标单卡。分别提供 baseline 和 contiguous 的脚本、配置；复用公共 loader、manager 和模型计算流程。

先确认设备运行环境可用，再区分模型实现问题。容器中的设备节点和驱动库、torch_npu/PyTorch/CANN 兼容性属于环境前提；缺失驱动不能靠切换权重布局修复。不要以关闭设备后端自动加载或 CPU 执行替代 NPU 推理验证。

从最终配置检查整个 pipeline：

| 范围 | 核对内容 |
|---|---|
| DiT | attention、RoPE、RMSNorm、LayerNorm、调制、矩阵算子及 checkpoint 精度 |
| 文本编码器 | 自身算子选择、加载精度和组件 offload，不能只看 DiT 配置 |
| scheduler | device、dtype、位置编码及可能的 CPU fallback |
| VAE | 模型加载、解码算子、输入输出精度及组件迁移 |

优先复用既有 NPU 或设备可用的 PyTorch 实现，选择权放在 config。PyTorch 运算在 NPU tensor 上执行不等于 CPU fallback；同样，配置写了 NPU attention 也不能证明其他组件已适配。

## 平台存储接口

入口为 [NpuDevice / NpuBlockOffload](../../../../lightx2v_platform/base/ascend_npu.py)，通过与 CUDA 相同的 [backend 接口](../../../../lightx2v_platform/base/offload.py) 提供内存操作。

- 当前连续布局以字节 buffer 建立 typed views，需要在相关设备 buffer 创建前调用 backend 的 `prepare()`，请求 ND 存储。沿用公共加载流程，不在模型中重复设置设备选项。
- `torch.npu.config.allow_internal_format` 在部分版本只有 setter。可以记录请求设置和实际 tensor 格式，不依赖不存在的 getter；可用接口以目标环境为准。
- NPU backend 的 `validate_checkpoint=True` 让普通 block baseline 也校验原始权重精度，避免 baseline 与 contiguous 使用不同的错误容忍策略。
- 复制、stream、event 与 synchronize 使用实际设备模块。不要在新增通用函数中硬编码 `torch.cuda`。

`NpuDevice.copy_to_cpu()` 已处理非连续 CPU 目标：先得到完成 D2H 的连续 CPU 源，再由 CPU 按目标 stride 写入。不可简化成直接异步写转置 host view；此前该问题会造成 offload 权重往返后损坏。这个修复属于平台复制层，CUDA 路径不应被迫采用相同额外复制。

相关实现：[公共权重工具](../../../../lightx2v/common/ops/utils.py)、[MM 模板](../../../../lightx2v_platform/ops/mm/template.py)、[norm 模板](../../../../lightx2v_platform/ops/norm/norm_template.py)。NPU 算子的特殊绑定留在其 [MM](../../../../lightx2v_platform/ops/mm/ascend_npu/mm_weight.py)、[RMSNorm](../../../../lightx2v_platform/ops/norm/ascend_npu/npu_rms_norm.py)、[LayerNorm](../../../../lightx2v_platform/ops/norm/ascend_npu/npu_layer_norm.py) 实现中。

## 两种现有配置选择

| 算子 | Wan NPU 示例 | Qwen NPU 示例 |
|---|---|---|
| attention | 三个 attention 配置项均为 `npu_flash_attn` | `attn_type=npu_flash_attn` |
| RoPE | `npu_rope` | `torch_real_rope` |
| RMSNorm / LayerNorm | `npu_rms_norm` / `npu_layer_norm` | `torch` / `torch` |
| 调制 | `torch` | `torch` |
| 文本编码器 | `t5_rms_norm_type=torch` | Qwen2.5-VL 自身路径 |

根据当前源码和算子语义选择，不机械替换为所有名字带 `npu_` 的算子。检查 RoPE 的实数/复数表示、layout、旋转维度和计算精度；注册名存在不保证与目标模型输入协议一致。

## 实机验证与结果解释

先验证权重加载和多轮 H2D/D2H 往返，尤其是转置矩阵及保留在 CPU 的源值；再验证连续 steps、CFG 分支和完整生成。两个变体都出错时，先排查共同的权重、计算算子和解码路径，不能直接归因于连续布局。

保持当前源码版本和实际配置可追溯。历史 Wan 910B 成功记录不能自动证明重构后的版本成功；Qwen NPU 入口存在也不能作为生成已验证的依据。

没有 NPU 时可检查配置、导入边界、存储契约和 CUDA 回归，但明确标记模拟/替代设备测试。提供目标机器执行命令、所需权重和期望输出，实机确认前保留“未验证”状态。按 [验证](validation.md) 保存真实结果，不能用 mock 的通过数宣称 910B 支持完成。

# Wan 与 Qwen 接入案例

简体中文 | [English](model-examples_en.md)

这些案例说明当前实现的复用点和差异。精度、模型规模、block 数量及算子选择以所使用版本的源码、checkpoint 和最终配置为准。

## Wan

从 [TransformerWeights](../../../../lightx2v/models/networks/wan/weights/transformer_weights.py) 的 `register_offload_group` 追踪 block 定义，checkpoint 前缀为 `blocks.{i}.`。原有两个 `offload_block_cuda_buffers` 作为设备 slots。

推理沿用 [offload TransformerInfer](../../../../lightx2v/models/networks/wan/infer/offload/transformer_infer.py)，加载、布局和传输由公共层处理。旧的 Wan 专用 `models/networks/wan/weights/block_layout.py` 已由公共 loader 替代，不按旧 IDE 标签或历史路径恢复它。

- [CUDA continuous 配置](../../../../configs/wan/layout/continuous.json)：当前示例用 LightX2V FP8-vLLM 权重及 CUDA attention，T5/CLIP 也有各自量化配置。
- [NPU continuous 配置](../../../../configs/wan/layout/continuous_npu.json)：原始浮点权重路径，显式 NPU 算子和组件 offload。
- [T5 权重实现](../../../../lightx2v/models/input_encoders/hf/wan/t5/model.py) 接收 `rms_norm_type`；其选择从 runner 的 `t5_rms_norm_type` 传入，默认保持 CUDA 原行为。参见 [WanRunner](../../../../lightx2v/models/runners/wan/wan_runner.py)，其他调用者也需保持参数一致。
- [CUDA 入口](../../../../scripts/wan/layout/run_contiguous.sh) 与 [NPU 入口](../../../../scripts/wan/layout/run_contiguous_npu.sh) 各有 baseline 对应脚本，使用前核对路径、环境和实际输出名。

DiT 连续布局不表示 T5、CLIP、VAE 都使用相同布局。当前 GPU 和 NPU 示例的量化及组件驻留策略不同，不能直接用于跨平台的布局收益比较。

## Qwen Image

[QwenImageTransformerWeights](../../../../lightx2v/models/networks/qwen_image/weights/transformer_weights.py) 注册 `transformer_blocks.{i}.`。一个 block 包含 image attention、text attention、joint attention 和 FFN 四个 phase，它们共同进入一个 layout；不能将四个 phase 各自当成一个完整 block。

推理沿用 [QwenImageOffloadTransformerInfer](../../../../lightx2v/models/networks/qwen_image/infer/offload/transformer_infer.py)，使用同一套公共 group 声明、storage contract 和传输逻辑，没有 Qwen 专用布局加载器。

- [CUDA continuous 配置](../../../../configs/qwen_image/layout/continuous.json)：当前 Qwen-Image-2512 原始 BF16 示例，attention 为 `flash_attn3`，其他默认值从 weights/infer 读取。
- [NPU continuous 配置](../../../../configs/qwen_image/layout/continuous_npu.json)：BF16、`npu_flash_attn`、PyTorch real RoPE/norm/调制。
- [TransformerInfer](../../../../lightx2v/models/networks/qwen_image/infer/transformer_infer.py) 按 `modulate_type` 决定是否加载 Qwen Triton 内核，CUDA 默认内核保持原样。
- [文本编码器](../../../../lightx2v/models/input_encoders/hf/qwen25/qwen25_vlforconditionalgeneration.py) 与 [VAE](../../../../lightx2v/models/video_encoders/hf/qwen_image/vae.py) 各自整体 offload；连续 block 仅作用于 DiT。
- [CUDA 入口](../../../../scripts/qwen_image/layout/run_contiguous.sh)、[NPU 入口](../../../../scripts/qwen_image/layout/run_contiguous_npu.sh) 及对应 baseline 均存在。NPU 入口支持 `MODEL_PATH`，自动定位仓库并输出到仓库下；不要假定所有旧脚本都支持相同环境覆盖。

权重目录应包含原始模型的 `transformer/`、`text_encoder/`、`tokenizer/`、`vae/` 等文件。单个量化 DiT 文件不能替代整个 pipeline。现有 `fp8-sgl` 蒸馏入口不代表其算子已具备这里所需的存储契约。

## 新模型的判断顺序

| 发现的情况 | 接入位置 |
|---|---|
| 原生 block offload 和算子契约都齐全 | 模型注册 group，补入口及验证 |
| 多种 block 结构 | 同一 group 只容纳布局兼容的 blocks/slots；异构组件需要独立的权重容器、manager 及对应调度 |
| 算子有额外 scale、bias 或设备标量 | 算子存储描述及必要绑定 |
| 目标平台缺少内存能力 | 平台 backend；同时验证 CUDA 隔离 |
| 文本编码器或 VAE 仍有 CUDA 专用依赖 | 对应组件的配置或平台算子，不能靠 DiT layout 修复 |
| 没有原生 block offload，或使用不兼容的原生 nn.Module 表示 | 先分析 block 权重/slot/调度接口缺口，再确定改动范围 |

已有两个模型的代码用作机制参考，不复制其层号、参数量、精度、shape 或路径为通用常量。参考 [Wan 中文说明](../../../../scripts/wan/layout/README_CN.md) 与 [Qwen 中文说明](../../../../scripts/qwen_image/layout/README_CN.md) 获取现有命令；运行前根据目标机器调整。

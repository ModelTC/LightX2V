# Wan 与 Qwen Image 接入案例

简体中文 | [English](model-adapters_en.md)

这些案例用于定位当前实现。任务和平台的校验范围、脚本示例以及实际验证范围分别说明；脚本存在不等于对应硬件已验收。公共算法与生命周期见 [公共实现](implementation-patterns.md)。

## Wan2.1

### 调用链与估测输入

- [model.py](../../../../lightx2v/models/networks/wan/model.py)：`WanModel` 初始化与 `validate_wan_residency_config()` 调用。
- [transformer_weights.py](../../../../lightx2v/models/networks/wan/weights/transformer_weights.py)：`WanTransformerAttentionBlock` 与设备 slots。
- [offload/transformer_infer.py](../../../../lightx2v/models/networks/wan/infer/offload/transformer_infer.py)：`WanOffloadTransformerInfer`、`infer_with_blocks_offload()`、`release_resident_blocks()`。
- [wan_runner.py](../../../../lightx2v/models/runners/wan/wan_runner.py)：`run_segment()` 与 `_run_warmup()` 的释放入口。
- [估测工具](../../../../tools/estimate_offload_residency.py)：`build_wan_case()`，加载前缀 `blocks.0.`。
- [infer/utils.py](../../../../lightx2v/models/networks/wan/infer/utils.py)：`image_latent_size()`，与正式 I2V 的尺寸规则一致。

当前校验接受 `model_cls="wan2.1"` 的 `t2v` 或 `i2v`，示例脚本使用 I2V。其他 Wan 变体不能仅因继承相同类而宣称支持静态常驻。

I2V 估测读取 `--image-path` 的真实图片比例，结合配置 `size`、VAE stride 和 patch size 构造 latent 网格；直接把 JSON 尺寸当作最终图像尺寸可能不匹配正式推理。T2V 使用配置尺寸。时间维由规范化后的 `num_frames` 推导。

case 使用正式 Wan block、scheduler 状态和 RoPE 构造路径，模拟 hidden state、调制输入与 context。I2V 启用图像编码器时，context 长度还包含图像条件 token。当前工具明确拒绝尚未实现的 Sol-Attn Morton 输入重排，不能静默换算子规避。

量化 checkpoint 路径优先取 `dit_quantized_ckpt`；未量化路径取 `dit_original_ckpt`，否则使用 `model_path`。保持正式加载的 dtype 与敏感层转换规则，尤其不要改变 FP8 scale 的舍入语义。FP8 checkpoint 与计算 dtype 是不同设置。

### 脚本与配置

入口在 [Wan offload_residency](../../../../scripts/wan/inference/offload_residency/)，使用说明见 [中文版](../../../../scripts/wan/inference/offload_residency/README_CN.md) / [英文版](../../../../scripts/wan/inference/offload_residency/README.md)。

| 平台 | 模型权重与基础配置 | 默认生成条件 |
|---|---|---|
| CUDA | `Wan2.1-I2V-14B-720P-Lightx2v` FP8；[baseline.json](../../../../configs/wan/residency/baseline.json) | 81 帧、40 步，配置 `size=[480,832]`，实际 I2V 比例由图片决定 |
| Ascend NPU | 原始 `Wan2.1-I2V-14B-720P` BF16；[baseline_npu.json](../../../../configs/wan/residency/baseline_npu.json) | 同样的帧数、步数与尺寸配置，使用对应 NPU 算子 |

CUDA 配置中的 `dit_quantized_ckpt`、`t5_quantized_ckpt`、`clip_quantized_ckpt` 独立于脚本 `model_path`，须分别检查。NPU 脚本的模型路径是待填写示例，需要完整的模型与编码器/VAE 资产。

估测的 `--image-path` 与推理的 `--image_path` 使用同一图片；两份推理脚本的正负提示词和 seed 保持一致。修改帧数、尺寸或步数后从基础配置重新生成常驻配置。

输出在 `save_results/wan_residency/`：`resident.json`、`resident.profile.json`、`resident.mp4`、`baseline.mp4`；NPU 对应 `resident_npu.json`、`resident_npu.profile.json`、`resident_npu.mp4`、`baseline_npu.mp4`。

## Qwen Image

### 调用链与估测输入

- [model.py](../../../../lightx2v/models/networks/qwen_image/model.py)：`QwenImageTransformerModel` 初始化与配置校验。
- [transformer_weights.py](../../../../lightx2v/models/networks/qwen_image/weights/transformer_weights.py)：`QwenImageTransformerAttentionBlock` 与设备 slots。
- [offload/transformer_infer.py](../../../../lightx2v/models/networks/qwen_image/infer/offload/transformer_infer.py)：`validate_qwen_residency_config()`、`QwenImageOffloadTransformerInfer` 及释放入口。
- [qwen_image_runner.py](../../../../lightx2v/models/runners/qwen_image/qwen_image_runner.py)：`run()` 与 `_run_warmup()` 在 VAE 前释放常驻权重。
- [估测工具](../../../../tools/estimate_offload_residency.py)：`build_qwen_case()`，加载前缀 `transformer_blocks.0.`。
- [infer/utils.py](../../../../lightx2v/models/networks/qwen_image/infer/utils.py)：`get_qwen_image_size()`，由估测与 runner 共同使用。

当前范围为 `model_cls="qwen_image"`、`task="t2i"`，不含 layered、编辑任务、Magi compile、`zero_cond_t` 或 `use_layer3d_rope`；同时遵守公共静态常驻校验。普通 CFG 的两路顺序计算与被拒绝的多设备 CFG parallel 是不同条件。

图像网格由有效尺寸、VAE scale 和 patch 规则推导。`Qwen2Tokenizer` 使用正式 prompt template 计算长度并扣除模板前缀；修改正负提示词须同步估测与推理。CFG 开启时当前实现选较短文本分支作为代表，因为通常计算预算更少；这是一种代表输入选择，不是对任意算子的耗时上下界证明。

case 使用正式 `QwenEmbedRope` 及对应算子准备频率/位置，模拟图像、文本和调制输入。`run_block()` 输入图像/文本，返回文本/图像；接入 `run_blocks()` 时保持状态顺序。检查最终输出之外，也检查文本分支，避免单路正确掩盖另一分支错误。

checkpoint 优先采用对应量化/原始覆盖路径，其次 `transformer_model_path`，否则取 `model_path/transformer`。新量化格式仍需检查正式加载与算子契约，不能把 BF16 示例的验证结论推广到所有格式。

### 脚本与配置

入口在 [Qwen offload_residency](../../../../scripts/qwen_image/inference/offload_residency/)，使用说明见 [中文版](../../../../scripts/qwen_image/inference/offload_residency/README_CN.md) / [英文版](../../../../scripts/qwen_image/inference/offload_residency/README.md)。

| 平台 | 模型权重与基础配置 | 默认生成条件 |
|---|---|---|
| CUDA | `Qwen-Image-2512` BF16；[baseline.json](../../../../configs/qwen_image/residency/baseline.json) | 咖啡店提示词、16:9、50 步、seed 42 |
| Ascend NPU | 同一模型 BF16；[baseline_npu.json](../../../../configs/qwen_image/residency/baseline_npu.json) | 相同生成条件，使用对应 NPU 算子 |

模型目录包含 `transformer`、`text_encoder`、`tokenizer`、`scheduler`、`vae`。配置中的文本编码器/VAE offload 与 DiT 常驻分别生效；不要把整个 pipeline 的内存变化全部归因于 `r`。

输出在 `save_results/qwen_residency/`：`resident.json`、`resident.profile.json`、`resident.png`、`baseline.png`；NPU 对应 `resident_npu.json`、`resident_npu.profile.json`、`resident_npu.png`、`baseline_npu.png`。

## 两个平台的使用约定

两套目录均有 `estimate.sh`、`run.sh`、`run_baseline.sh` 及对应 `_npu.sh`。CUDA 脚本使用 `PLATFORM=cuda` 和 `CUDA_VISIBLE_DEVICES`；Ascend 使用 `PLATFORM=ascend_npu` 和 `ASCEND_RT_VISIBLE_DEVICES`，运行时代码通过 `AI_DEVICE="npu"` 选择设备接口。不要将平台名与设备名混用。

脚本先填写 `lightx2v_path`、`model_path`，source [base.sh](../../../../scripts/base/base.sh)，配置、输入、输出和 `mkdir` 路径直接使用 `"${lightx2v_path}/..."`，保留调用时的工作目录，不添加 `set -e`。`base.sh` 的 `DTYPE` 默认 BF16、`SENSITIVE_LAYER_DTYPE` 默认 None（跟随主 dtype），外部已设置的值会保留；比较时记录最终值。

脚本沿用 `base.sh` 中的 `PROFILING_DEBUG_LEVEL`，当前为 `2`。需要调整时修改该公共设置，baseline 与常驻使用相同级别。在调用脚本前 export 不会覆盖 `base.sh` 中的显式赋值。计时标签与验证范围见 [验证与调试](validation-and-debugging.md)。

现有 CUDA 验证与 NPU 静态检查只能覆盖各自条件。新增 NPU 测试要实际执行估测与推理入口；没有硬件时明确保留“未实测”，不通过删改精度或输入掩盖限制。

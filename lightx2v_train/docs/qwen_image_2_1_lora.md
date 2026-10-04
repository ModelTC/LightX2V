# Qwen-Image-2.1 LoRA 训练

模型名为 `qwen_image_2_1`，支持文生图（T2I）与单图/多图编辑（I2I）混合 LoRA 训练、RGB/RGBA 图片、特征缓存和恢复训练，并接入项目通用的 Flow Matching、DMD 和 Consistency 训练器。使用 2.1 的本地权重，不能复用 2512 的 latent 缓存或 LoRA。

以下命令在 `lightx2v_train` 目录执行。默认权重目录是 `/data/wushuo1/models/Qwen-Image-2.1`；训练和验证都合并 `/data/wushuo1/data/LightX2V_train_data_examples/` 下的 `dataset_v1` 与 `dataset_edit` 对应 JSONL。

## 安装与启动

先准备已安装训练依赖和适配设备的 PyTorch 的环境。Diffusers 需包含 `QwenImage21Pipeline`、`QwenImage21Transformer2DModel` 和 `AutoencoderKLQwenImage21`，Transformers 需支持 Qwen3-VL。以下示例假设环境位于 `.venv-qwen-image-2.1`：

```bash
cd /data/wushuo1/LightX2V/lightx2v_train

PYTHON_BIN=.venv-qwen-image-2.1/bin/python \
    bash scripts/run_qwen_image_2_1_t2i_lora.sh
```

默认自动检测 CUDA。用 `PYTHON_BIN`、`CONFIG`、`NPROC_PER_NODE` 指定解释器、配置和进程数。示例启用 DDP，每个设备容纳完整模型；使用 FSDP2 时需另行配置并关闭 `distributed.dp.enabled`。

默认配置为 `configs/train/flow/qwen_image_2_1_lora.yaml`：BF16、LoRA rank/alpha 16、梯度检查点、1024² 目标面积，每 100 步保存并验证。推理默认 40 步、关闭 CFG。

配置支持环境变量覆盖路径：

```bash
QWEN_IMAGE_21_MODEL_PATH=/path/to/Qwen-Image-2.1 \
QWEN_IMAGE_21_TRAIN_DATA=/path/to/train.jsonl \
QWEN_IMAGE_21_EDIT_TRAIN_DATA=/path/to/edit_train.jsonl \
QWEN_IMAGE_21_VAL_DATA=/path/to/val.jsonl \
QWEN_IMAGE_21_EDIT_VAL_DATA=/path/to/edit_val.jsonl \
QWEN_IMAGE_21_OUTPUT_DIR=/path/to/output \
PYTHON_BIN=.venv-qwen-image-2.1/bin/python \
    bash scripts/run_qwen_image_2_1_t2i_lora.sh
```

## 选择训练算法

`runtime.platform` 或 `PLATFORM` 选择设备，`model.name` 选择模型，`training.method` 选择通用训练器。示例使用 `flow_matching`；DMD 设为 `dmd`，Consistency 设为 `consistency`，并在 `training.consistency` 下选择 `algorithm` 和 `mode`。CM、PCM、TCM、MeanFlow、sCM 按所选算法配置 teacher、阶段 checkpoint 等参数。

DMD 的 LoRA 和优化器分别配置在 `training.student`、`training.fake`，不能使用 SFT 的顶层 `training.train_type`、`training.lora`。它会加载 student、fake、teacher 三份主干，设备内存需求高于 SFT。仅使用文本的训练集可设为 `prompt_dataset`；包含参考图片的训练集以及图片验证集使用 `image_dataset`。

MeanFlow 添加端点时间嵌入，sCM 添加 log-variance head，辅助参数随 checkpoint 保存和恢复。两类算法的 JVP 使用 `training.consistency.jvp.method: finite_difference`；已验证的上游 2.1 RoPE 实现包含 `Tensor.tolist()`，不兼容 `torch.func.jvp`，因此该实现下 `exact` 模式不可用。

## 数据与透明通道

Flow Matching 和 Consistency CT 训练 JSONL 每行至少包含 `prompt` 和 `target_image`。省略 `source_images` 或写 `[]` 表示 T2I；提供有序的图片路径列表表示 I2I，可含多张参考图。图片路径相对各自 JSONL 所在目录解析：

```json
{"prompt":"A red ceramic teapot on a wooden table","target_image":"images/teapot.jpg"}
{"prompt":"Change the teapot to blue","source_images":["images/teapot.jpg"],"target_image":"images/blue_teapot.jpg"}
{"prompt":"Use the style of the second picture to redraw the first","source_images":["images/object.jpg","images/style.jpg"],"target_image":"images/styled_object.jpg"}
{"prompt":"This is an RGBA image with transparency. A dragon sticker. The image has alpha channel and the background is transparent.","target_image":"images/dragon.png"}
```

可以把两类记录放入同一 JSONL，也可以通过 `data.train.data_path` 的多个路径拼接。混合比例由记录数决定，shuffle 后不保证每个优化步的比例；当前没有任务比例 sampler。每卡物理 batch 为 1，不同记录可有不同数量和尺寸的参考图，有效 batch 为梯度累积步数乘数据并行数。只训练其中一类时，修改 YAML 的路径列表；不要将未使用的环境变量设为空路径。prompt dropout 只替换文本，保留参考图。

训练目标优先使用记录中的 `target_height`/`target_width`（必须同时提供），否则按目标图片自身宽高比与 `target_area` 缩放裁剪，不受参考图尺寸影响。所有尺寸对齐到 32 的倍数。每张参考图按自身宽高比缩放，VLM/VAE 两路使用同一尺寸；`data.processor.reference_image_area` 默认与 `target_area` 相同，示例为 1024²，建议保留该值以匹配 VLM 的可接受尺寸。

RGB 图片自动补不透明 alpha。VAE 保留目标和参考图的 RGBA，四通道统一归一化到 `[-1, 1]`；VLM 的参考图副本将透明区域合成到白底。仅目标 latent 加噪并计算训练损失，参考 latent 作为条件；输出使用 PNG 保存透明通道。2.1 使用 64 通道、16 倍压缩的 VAE，latent 不做 2×2 打包；对应 scheduler 的 `patch_size` 必须保持 `[1, 1]`。

## 预缓存训练

以下以 Flow Matching 为例，先用在线配置构建缓存；此阶段只加载 VAE 和文本编码器：

```bash
.venv-qwen-image-2.1/bin/python cache_data.py \
    --config configs/train/flow/qwen_image_2_1_lora.yaml \
    --output_dir ./cache/qwen_image_2_1/train \
    --data_split train --save_dtype bf16
```

构建完成后，在训练配置中将 `data.train.name` 改为 `cache_dataset`、`data.train.data_path` 改为缓存清单 `./cache/qwen_image_2_1/train/cache_data.jsonl` 的列表，并设置 `inference.method: none` 关闭训练中验证；再使用同一启动脚本运行。缓存保存混合样本各自的目标 latent、提示条件及参考 latent。缓存训练阶段只加载 Transformer，可减少冻结组件占用的显存；目标 latent 使用 VAE posterior mode，在线训练使用 posterior sample，参考 latent 均使用 posterior mode。更换目标/参考图片、prompt、目标/参考分辨率或模型后重新构建缓存；覆盖已有输出需显式传 `cache_data.py --overwrite`。

其他算法使用各自的训练配置构建缓存，并保留其 `training` 配置。DMD/Consistency 可能要求额外的条件分支或 generation shape，不能直接复用仅含 SFT 条件的缓存。

## 保存、恢复与推理

默认输出到 `output_train/qwen_image_2_1_lora/`，可通过 `QWEN_IMAGE_21_OUTPUT_DIR` 覆盖。每个 `checkpoint-000000100/` 这类目录包含 `pytorch_lora_weights.safetensors`；完整 checkpoint 还保存优化器等恢复状态。

DMD 根目录的 LoRA 是 student，fake 权重和状态单独保存。MeanFlow/sCM 还会保存 `consistency_auxiliary.safetensors`，恢复训练需要该文件。

`resume.auto_resume: true` 从 `training.output_dir` 中最新 checkpoint 恢复，需要完整目录、原模型和 LoRA 配置以及相同进程数。切换算法或开始独立实验时更换输出目录，或设置 `auto_resume: false`。

使用保存的 Flow Matching LoRA 运行验证集：

```bash
QWEN_IMAGE_21_LORA_PATH=./output_train/qwen_image_2_1_lora/checkpoint-000000100/pytorch_lora_weights.safetensors \
    .venv-qwen-image-2.1/bin/python infer.py \
    --config configs/infer/qwen_image_2_1_lora.yaml
```

默认推理也同时读取两份验证集。T2I 验证记录可只包含 `prompt`，I2I 可使用 `prompt` 加 `source_images`，无需目标图片。可用 `QWEN_IMAGE_21_VAL_DATA`、`QWEN_IMAGE_21_EDIT_VAL_DATA` 覆盖对应验证集，`QWEN_IMAGE_21_INFER_OUTPUT_DIR` 指定图片输出目录。显式 `target_height`/`target_width` 优先；没有目标图片和显式尺寸时，I2I 按最后一张参考图的比例与 `default_height * default_width` 的面积生成，T2I 使用默认宽高。若记录包含目标图片，则使用其处理后的尺寸。

DMD student 也使用该推理路径，将 LoRA 路径指向 DMD checkpoint，并将推理配置的 `num_inference_steps` 改为训练的 rollout 步数、保留 `enable_cfg: false`。Flow Matching/DMD 的导出权重也可在同一版本的 `QwenImage21Pipeline` 中通过 `pipe.load_lora_weights(checkpoint_dir)` 加载。

sCM 对外仍预测 rectified-flow velocity，log-variance head 用于训练损失；其 LoRA 可以通过上述普通 velocity 推理路径加载，采样步数需按训练效果设置。

MeanFlow 采样需要每步传入端点 `r=t_next`，并加载端点时间嵌入。当前 `image_infer` 仅调用 `denoise(..., timestep_or_sigma, condition)`，`infer.py` 不会自动恢复 `consistency_auxiliary.safetensors`。MeanFlow checkpoint 需通过恢复辅助参数、传递端点的 Consistency capability 采样代码运行，不能直接使用上述普通 LoRA 命令。

官方实现参考：[Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1)、[验证使用的 Diffusers pipeline](https://github.com/huggingface/diffusers/blob/578c9b2c6636ab2424a0e56186268b83623656b2/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py)。

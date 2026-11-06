# TFMQ LoRA 重建

[English](RECONSTRUCTION.md) · [工具入口与实验结论](README_zh.md)

本文以 Qwen-Image 2.1 为例，当前重建实现支持 Qwen-Image 2.1。

冻结 base，只优化 LoRA。示例配置选择 232 组 Linear，rank 64、alpha 8。统一入口 `lora_reconstructor.py` 根据 `target_model` 自动识别完整模型目录、导出的 transformer 目录（含分片）或单个 safetensors 文件，使用同一套 TFMQ 流程。

先联合重建时间通路的 modulation 和最终 scale，再冻结该通路。随后按输入投影、block 00–31、输出投影的顺序重建，最小化 `MSE(S(xs, ms), T(xt, mt))`：学生和教师分别使用各自的前缀特征、时间条件，梯度不跨重建单元。

## 启动

先修改配置中的模型与数据路径，再从仓库根目录运行。在具有 CUDA 版 PyTorch 的环境中安装 LightX2V 基础依赖后，安装重建依赖：

```bash
pip install -r tools/extract/requirements-reconstruction.txt
python -c "from diffusers.models.transformers import transformer_qwenimage21; print('Qwen-Image 2.1 backend available')"
```

已验证环境使用 Diffusers `0.41.0.dev0`，对应提交 `62cde2b0115d57492d2f5ecefab54994797df2e3`，以及 Transformers `5.14.1`，版本已固定在上述依赖文件中。无需安装训练框架。

```bash
# target 为完整模型目录：先采集轨迹，再重建
torchrun --standalone --nproc_per_node=8 tools/extract/lora_reconstructor.py \
  --config tools/extract/configs/qwen_image_21_reconstruct_tfmq.json \
  --collect-and-reconstruct

# target 为导出的 transformer 目录或单个 .safetensors 文件
torchrun --standalone --nproc_per_node=8 tools/extract/lora_reconstructor.py \
  --config tools/extract/configs/qwen_image_21_reconstruct_checkpoint_tfmq.json \
  --collect-and-reconstruct
```

### 命令行参数

`lora_reconstructor.py`：

- `--config <路径>`：必填，指定 JSON 或 YAML 重建配置文件。
- `--calibration <路径>`：覆盖配置中的校准文件或目录；执行采集时，该目录用于保存轨迹。
- `--output <路径>`：覆盖最终 LoRA 的 `.safetensors` 输出路径。
- `--work-dir <路径>`：覆盖运行日志和临时重建缓存的保存目录。
- `--device <设备>`：覆盖重建设备，例如 `auto`、`cuda:0` 或 `cpu`。torchrun 下建议使用 `auto`，由各进程选择本地 GPU。此参数不控制采集使用的 GPU。
- `--collect-calibration`：仅准备校准轨迹。所有 target 格式均自动复用已有的匹配轨迹。
- `--collect-and-reconstruct`：准备轨迹后，在同一次启动中进行重建。
- `--reuse-calibration`：要求使用已有校准数据进行重建，不重新采集。所有 target 格式均校验来源和文件清单，并可从较大的缓存中选取子集。数据缺失或不匹配时报错。
- `--dry-run`：检查采集或复用流程，不执行推理或训练；必须与上述三个流程参数之一同时使用。独立 transformer 目录或文件的准备过程仍可能写入推理目录链接和配置。
- `--help`：显示可用参数。

三个流程参数互斥。都不传时，直接使用 `calibration` 重建，不执行采集准备或 checkpoint 子集选择。

### 运行方式

target 为导出的 transformer 目录或文件时，采集器使用其权重和 base 的冻结组件组成推理目录。目录支持单文件及带索引的 safetensors 分片。采样使用 `calibration_inference_config`；权重旁若有 `inference_config.json`，其中的 `sample_sigmas` 和 `scheduler` 会覆盖对应设置。目前采集要求 CFG=1。

新采集清单记录模型文件的实际路径、大小和修改时间、JSON 配置摘要、请求清单路径及摘要、seed、实际推理配置和采集精度，复用前统一比对。旧单文件 checkpoint 缓存仍按原有来源记录校验；没有可验证来源记录的旧缓存，需要另选目录重新采集。模型权重使用文件元数据识别，不计算整份权重的哈希。

采集使用各卡独立的 BF16 推理与 CPU 卸载；重建使用 DDP，每次加载一个单元。单个单元及激活须能放入一张 GPU，各卡需共享模型、数据和输出文件系统。单卡运行时将 torchrun 前缀换成 `python`。

## 配置与产物

提取目标是 **`source_model` + 导出的 LoRA ≈ `target_model`**。

- `source_model`：冻结的原始 base，之后使用 LoRA 时也应加载这个模型。两份示例均使用 `lightx2v_train/models/Qwen-Image-2.1`。
- `target_model`：希望逼近的完整模型权重，作为 teacher 生成校准轨迹和重建监督；这里填写完整模型目录、独立 transformer 目录或权重文件，不是 LoRA 文件，也不要求来自 RL。
- `output`：最终导出的 LoRA `.safetensors` 路径；使用时将它加载到 `source_model` 上。

两份示例的 target 分别为：

| 配置 | `target_model` |
|---|---|
| [qwen_image_21_reconstruct_tfmq.json](configs/qwen_image_21_reconstruct_tfmq.json) | `lightx2v_train/models/Qwen-Image-2.1-Lightning-v2-base` |
| [qwen_image_21_reconstruct_checkpoint_tfmq.json](configs/qwen_image_21_reconstruct_checkpoint_tfmq.json) | `lightx2v_train/models/Qwen-Image-2.1-Lightning-v2-RL/diffusion_pytorch_model.safetensors` |

第一项名称中的 `-base` 是目标模型名称的一部分；本次提取的原始 base 仍由 `source_model` 指定。两份配置使用同一算法，数据量和迭代数只是实验取值。

其他关键参数：

| 参数 | 含义 |
|---|---|
| `layers` | 要提取并优化 LoRA 的 Linear 层名称，支持 `*` 通配符；示例覆盖 232 组 Linear。 |
| `rank` / `alpha` | LoRA 的秩和缩放参数，示例为 64 / 8；权重增量为 `(alpha / rank) × up @ down`。SVD 初始化会补偿该缩放，alpha=8 不表示只拟合目标差分的 1/8。 |
| `initialization` | `svd`：对 target 与 source 的权重差做 SVD 初始化，并在训练前保存；`lora`：从 `initial_lora` 加载已有 LoRA；也支持 `random`。 |
| `svd_output` / `initial_lora` | 前者是 SVD 初始化的独立保存路径，默认将 `output` 的 `.safetensors` 后缀换为 `.svd.safetensors`；后者仅在 `initialization: lora` 时填写。 |
| `calibration_manifest` | 采集请求清单，包含提示词、参考图等，用于选择请求。 |
| `calibration_counts` | 按 `t2i`、`i2i_single`、`i2i_multi` 指定请求数；使用全部采集时间步，每个请求的每个时间步产生一条重建数据，例如 1024 个请求 × 4 步 = 4096 条。 |
| `calibration_inference_config` | teacher 采集时的推理配置，包含采样步数、分辨率等；独立权重旁的采样配置覆盖规则见上文。 |
| `calibration` | 原始校准轨迹的保存或读取路径，与请求清单分开；采集后保留。 |
| `iterations` / `temporal_iterations` | 分别为每个普通重建单元、时间通路的优化器更新次数；不是遍历数据集的 epoch 数。 |
| `batch_size` / `learning_rate` | 全局 batch 和 LoRA 学习率，示例为 8 / `5e-5`。每卡每次前向一个样本，累积 `batch_size / WORLD_SIZE` 步，batch 必须能被卡数整除。 |
| `compute_dtype` / `dtype` | 前者控制重建前向精度（`bf16` 或 `fp32`），LoRA 主参数和 MSE 保持 FP32；后者控制导出权重精度。 |
| `checkpointing` | 开启重建单元内部的梯度检查点，通过反向时重算降低激活显存；不是训练断点保存。 |
| `work_dir` | 存放运行记录和临时单元缓存。单元推进时清理上一单元缓存，正常结束时清理剩余临时缓存；原始校准轨迹和导出权重保留。 |
| `log_every` / `save_iteration_losses` | 前者控制每隔多少 iter 显示 minibatch MSE；后者开启逐 iter 的 `iteration_losses.jsonl` 保存。 |

`work_dir/<run>/` 保存配置及各单元训练前后的 `metrics.jsonl`。MSE 在该单元全量校准缓存上计算，变差时恢复该单元的初始 LoRA。重新启动会从初始化开始重建，不恢复优化器状态。

图片效果验证使用 inference 框架。

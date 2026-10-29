# LoRA 模型部署与相关工具

LoRA (Low-Rank Adaptation) 是一种高效的模型微调技术，通过低秩矩阵分解显著减少可训练参数数量。LightX2V 全面支持 LoRA 技术，包括 LoRA 推理、LoRA 提取和 LoRA 合并等功能。

## 🎯 LoRA 技术特性

- **灵活部署**：支持动态加载和移除 LoRA 权重
- **多种格式**：支持多种 LoRA 权重格式和命名约定
- **工具完善**：提供完整的 LoRA 提取、合并工具链

## 📜 LoRA 推理部署

### 配置文件方式

在配置文件中指定 LoRA 路径：

```json
{
  "lora_configs": [
    {
      "path": "/path/to/your/lora.safetensors",
      "strength": 1.0
    }
  ]
}
```

**配置参数说明：**

- `lora_path`: LoRA 权重文件路径列表，支持多个 LoRA 同时加载
- `strength_model`: LoRA 强度系数 (alpha)，控制 LoRA 对原模型的影响程度
- `lora_merge_force_fp32`: 默认为 `false`，保持原有合并精度。设为 `true` 时，以 FP32 加载 LoRA 权重、计算增量并与基础权重相加，随后转回各层原来的 dtype。仅影响静态合并；使用动态 LoRA（`lora_dynamic_apply=true`）时不生效。逐层合并需要额外的临时内存，多份 LoRA 按配置顺序分别合并并转回原 dtype。

### 命令行方式

直接在命令行中指定 LoRA 路径（仅支持加载单个 LoRA）：

```bash
python -m lightx2v.infer \
  --model_cls wan2.1 \
  --task t2v \
  --model_path /path/to/model \
  --config_json /path/to/config.json \
  --lora_path /path/to/your/lora.safetensors \
  --lora_strength 0.8 \
  --prompt "Your prompt here"
```

### 多LoRA配置

要使用多个具有不同强度的LoRA，请在配置JSON文件中指定：

```json
{
  "lora_configs": [
    {
      "path": "/path/to/first_lora.safetensors",
      "strength": 0.8
    },
    {
      "path": "/path/to/second_lora.safetensors",
      "strength": 0.5
    }
  ]
}
```

### 支持的 LoRA 格式

LightX2V 支持多种 LoRA 权重命名约定：

| 格式类型 | 权重命名 | 说明 |
|----------|----------|------|
| **标准 LoRA** | `lora_A.weight`, `lora_B.weight` | 标准的 LoRA 矩阵分解格式 |
| **Down/Up 格式** | `lora_down.weight`, `lora_up.weight` | 另一种常见的命名约定 |
| **差值格式** | `diff` | `weight` 权重差值 |
| **偏置差值** | `diff_b` | `bias` 权重差值 |
| **调制差值** | `diff_m` | `modulation` 权重差值 |

### 推理脚本示例

**步数蒸馏 LoRA 推理：**

```bash
# T2V LoRA 推理
bash scripts/wan/inference/distill/run_wan_t2v_distill_lora.sh

# I2V LoRA 推理
bash scripts/wan/inference/distill/run_wan_i2v_distill_lora.sh
```

### API 服务中使用 LoRA

在 API 服务中通过 [config 文件](https://github.com/ModelTC/lightx2v/blob/main/configs/distill/wan21/wan_t2v_distill_lora_4step_cfg.json) 指定，对 [scripts/server/start_server.sh](https://github.com/ModelTC/lightx2v/blob/main/scripts/server/start_server.sh) 中的启动命令进行修改：

```bash
python -m lightx2v.api_server \
  --model_cls wan2.1 \
  --task t2v \
  --model_path $model_path \
  --config_json ${lightx2v_path}/configs/distill/wan21/wan_t2v_distill_lora_4step_cfg.json \
  --port 8000 \
  --nproc_per_node 1
```

## 🔧 LoRA 提取工具

使用 `tools/extract/lora_extractor.py`，按 JSON/YAML 配置对选定 Linear 的权重差做 FP32 SVD。运行以下命令前，先修改配置中的模型与输出路径。相对路径以运行命令的工作目录为准，支持 `~` 和环境变量。

需要基于校准数据优化 Qwen-Image 2.1 LoRA 时，使用独立入口 `tools/extract/lora_reconstructor.py`。支持 Q-Diffusion 风格的逐块重建，以及 TFMQ 风格的时间通路联合重建后再逐块重建；支持单卡和 `torchrun` DDP。示例配置使用 rank=64、alpha=8，并单独保存 SVD 初始化 LoRA。校准数据采集、算法细节和运行命令见[重建说明](../../../../tools/extract/RECONSTRUCTION_zh.md)。

### 基本用法

```bash
python tools/extract/lora_extractor.py \
  --config tools/extract/configs/wan.json

python tools/extract/lora_extractor.py \
  --config tools/extract/configs/qwen_image_21.json
```

### 参数说明

| 配置字段 | 默认值 | 说明 |
|----------|--------|------|
| `source_model` | 必填 | 基础模型本地路径 |
| `target_model` | 必填 | 微调模型、导出学生权重或原始学生 checkpoint 路径 |
| `output` | 必填 | 输出 `.safetensors` 路径 |
| `layers` | 必填 | 不含 `.weight` 的模块名，支持 `*` 等 glob 通配符 |
| `rank` | `32` | LoRA rank，超过矩阵维度时逐层截断 |
| `alpha` | 与 rank 相同 | LoRA alpha；因子补偿 `alpha / 实际rank`，不会改变重建增量 |
| `dtype` | `bf16` | 因子 dtype：`bf16` / `fp16` / `fp32` 或完整名称；alpha 标量固定 FP32 |
| `device` | `auto` | 优先 CUDA，否则 CPU；也可指定 `cuda:0` 等 |
| `key_prefix` | `auto` | `transformer` / `diffusion_model`；自动根据模型键名选择 |

路径自动识别完整模型目录中的 `transformer/`、独立 Transformer 目录、safetensors 文件/分片、PyTorch state dict，以及下文的 train/RL checkpoint。无需 `--source-type` 或 `--target-type`。输入为本地文件，不自动下载模型。

Wan 配置覆盖 T2V 的 Attention、FFN、文本/时间嵌入及输出 Linear，兼容 native Wan 和 Diffusers Wan 命名；I2V 特有层可自行加入 `layers`。Qwen 配置覆盖 32 个 block 的 224 个 Linear 和 8 个全局 Linear，不输出归一化参数差分。Qwen 合并的 `gate_up` 会按 `gate_layer` / `proj` 分开提取，使用 `transformer.` 前缀兼容 ComfyUI。

所有选中的矩阵均做低秩分解，包括小矩阵；不再提供 `--diff-only`，也不自动输出 bias、norm 或 modulation 参数的直接差分。完全相同的矩阵会跳过。层名无匹配、目标缺层或形状不一致时直接报错，不保存不完整 LoRA。

### 高级用法示例

**提取高秩 LoRA：**

```bash
python tools/extract/lora_extractor.py \
  --config tools/extract/configs/qwen_image_21.json \
  --target-model /path/to/exported/student \
  --output /path/to/high_rank_lora.safetensors \
  --rank 64 \
  --alpha 64 \
  --output-dtype fp16
```

命令行参数覆盖文件中的同名字段；只覆盖 rank 时，文件中显式设置的 alpha 保持不变。

### 导出 train / RL 学生模型

`export_student.py` 在 CPU 上导出全参数学生模型，输出独立 Transformer 目录及分片索引，不复制文本编码器、VAE 或采样配置。导出结果可直接作为提取器的 `target_model`。

```bash
python tools/extract/export_student.py \
  --checkpoint /path/to/train/checkpoint-000001000 \
  --base-model /path/to/base/model \
  --output /path/to/exported/student

python tools/extract/export_student.py \
  --checkpoint /path/to/rl/global_step_100 \
  --base-model /path/to/base/model \
  --output /path/to/exported/rl_student \
  --dtype bf16
```

支持 train 的 `model_state.pt`、`student_consolidated/`、`dist_state/student` 中的 `model`，以及 `dist_state` 中的 `student_model` 或 `model`；仅读取学生参数，不加载 optimizer/fake。RL 支持 `actor/` 下的一维 FSDP2 DTensor 分片和完整副本；运行时使用与训练 checkpoint 兼容的现代 PyTorch。LoRA 训练 checkpoint 需要先合并成全参数权重。

默认保留保存精度，`--max-shard-size-gb` 默认 2；单个张量不拆分，可能超过此上限。`--base-model` 用于校验参数名称/形状并提供 `config.json`，源目录已有配置时可以省略。DCP 若有刻意不保存的冻结参数，可显式加 `--fill-missing-from-base` 补齐；默认缺层报错。导出目录必须不存在，失败不会留下半成品。safetensors/DCP 逐层读取，现代 PyTorch 下的 RL 分片通过 CPU mmap 读取，无需启动分布式训练进程。

## 🔀 LoRA 合并工具

使用 `tools/extract/lora_merger.py` 将 LoRA 权重合并到基础模型中，以进行后续量化等操作。

合并工具支持 `diffusion_model.` 和 `transformer.` 前缀，并应用文件内每层的 `alpha / rank`。命令行 `--alpha` 仍表示整体合并强度，默认 1；旧 LoRA 未保存 alpha 时保持单位缩放。

### 基本用法

```bash
python tools/extract/lora_merger.py \
  --source-model /path/to/base/model \
  --lora-model /path/to/lora.safetensors \
  --output /path/to/merged/model.safetensors \
  --alpha 1.0
```

### 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--source-model` | str | ✅ | 无 | 基础模型路径 |
| `--lora-model` | str | ✅ | 无 | LoRA 权重路径 |
| `--output` | str | ✅ | 无 | 输出合并模型路径 |
| `--source-type` | str | ❌ | `safetensors` | 基础模型格式 |
| `--lora-type` | str | ❌ | `safetensors` | LoRA 权重格式 |
| `--output-format` | str | ❌ | `safetensors` | 输出格式 |
| `--alpha` | float | ❌ | `1.0` | LoRA 合并强度 |
| `--output-dtype` | str | ❌ | `bf16` | 输出数据类型 |

### 高级用法示例

**部分强度合并：**

```bash
python tools/extract/lora_merger.py \
  --source-model /path/to/base/model \
  --lora-model /path/to/lora.safetensors \
  --output /path/to/merged_model.safetensors \
  --alpha 0.7 \
  --output-dtype fp32
```

**多格式支持：**

```bash
python tools/extract/lora_merger.py \
  --source-model /path/to/base/model.pt \
  --source-type pytorch \
  --lora-model /path/to/lora.safetensors \
  --lora-type safetensors \
  --output /path/to/merged_model.safetensors \
  --output-format safetensors \
  --alpha 1.0
```

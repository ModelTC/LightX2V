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
bash scripts/wan/distill/run_wan_t2v_distill_lora_4step_cfg.sh

# I2V LoRA 推理
bash scripts/wan/distill/run_wan_i2v_distill_lora_4step_cfg.sh
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

使用 `tools/extract/lora_extractor.py` 从两个模型的差异中提取 LoRA 权重。

### 基本用法

```bash
python tools/extract/lora_extractor.py \
  --source-model /path/to/base/model \
  --target-model /path/to/finetuned/model \
  --output /path/to/extracted/lora.safetensors \
  --rank 32
```

### 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--source-model` | str | ✅ | - | 基础模型路径 |
| `--target-model` | str | ✅ | - | 微调后模型路径 |
| `--output` | str | ✅ | - | 输出 LoRA 文件路径 |
| `--source-type` | str | ❌ | `safetensors` | 基础模型格式 (`safetensors`/`pytorch`) |
| `--target-type` | str | ❌ | `safetensors` | 微调模型格式 (`safetensors`/`pytorch`) |
| `--output-format` | str | ❌ | `safetensors` | 输出格式 (`safetensors`/`pytorch`) |
| `--rank` | int | ❌ | `32` | LoRA 秩值 |
| `--output-dtype` | str | ❌ | `bf16` | 输出数据类型 |
| `--diff-only` | bool | ❌ | `False` | 仅保存权重差值，不进行 LoRA 分解 |

### 高级用法示例

**提取高秩 LoRA：**

```bash
python tools/extract/lora_extractor.py \
  --source-model /path/to/base/model \
  --target-model /path/to/finetuned/model \
  --output /path/to/high_rank_lora.safetensors \
  --rank 64 \
  --output-dtype fp16
```

**仅保存权重差值：**

```bash
python tools/extract/lora_extractor.py \
  --source-model /path/to/base/model \
  --target-model /path/to/finetuned/model \
  --output /path/to/weight_diff.safetensors \
  --diff-only
```

## 🔀 LoRA 合并工具

使用 `tools/extract/lora_merger.py` 将 LoRA 权重合并到基础模型中，以进行后续量化等操作。

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

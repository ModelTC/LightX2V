# LoRA Model Deployment and Related Tools

LoRA (Low-Rank Adaptation) is an efficient model fine-tuning technique that significantly reduces the number of trainable parameters through low-rank matrix decomposition. LightX2V fully supports LoRA technology, including LoRA inference, LoRA extraction, and LoRA merging functions.

## 🎯 LoRA Technical Features

- **Efficient Fine-tuning**: Dramatically reduces training parameters through low-rank adaptation
- **Flexible Deployment**: Supports dynamic loading and removal of LoRA weights
- **Multiple Formats**: Supports various LoRA weight formats and naming conventions
- **Comprehensive Tools**: Provides complete LoRA extraction and merging toolchain

## 📜 LoRA Inference Deployment

### Configuration File Method

Specify LoRA path in configuration file:

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

**Configuration Parameter Description:**
- `lora_path`: LoRA weight file path list, supports loading multiple LoRAs simultaneously
- `strength_model`: LoRA strength coefficient (alpha), controls LoRA's influence on the original model

### Command Line Method

Specify LoRA path directly in command line (supports loading single LoRA only):

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

### Multiple LoRAs Configuration

To use multiple LoRAs with different strengths, specify them in the config JSON file:

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

### Supported LoRA Formats

LightX2V supports multiple LoRA weight naming conventions:

| Format Type | Weight Naming | Description |
|-------------|---------------|-------------|
| **Standard LoRA** | `lora_A.weight`, `lora_B.weight` | Standard LoRA matrix decomposition format |
| **Down/Up Format** | `lora_down.weight`, `lora_up.weight` | Another common naming convention |
| **Diff Format** | `diff` | `weight` difference values |
| **Bias Diff** | `diff_b` | `bias` weight difference values |
| **Modulation Diff** | `diff_m` | `modulation` weight difference values |

### Inference Script Examples

**Step Distillation LoRA Inference:**

```bash
# T2V LoRA Inference
bash scripts/wan/run_wan_t2v_distill_4step_cfg_lora.sh

# I2V LoRA Inference
bash scripts/wan/run_wan_i2v_distill_4step_cfg_lora.sh
```

**Audio-Driven LoRA Inference:**

```bash
bash scripts/wan/run_wan_i2v_audio.sh
```

### Using LoRA in API Service

Specify through [config file](wan_t2v_distill_4step_cfg_lora.json), modify the startup command in [scripts/server/start_server.sh](https://github.com/ModelTC/lightx2v/blob/main/scripts/server/start_server.sh):

```bash
python -m lightx2v.api_server \
  --model_cls wan2.1_distill \
  --task t2v \
  --model_path $model_path \
  --config_json ${lightx2v_path}/configs/distill/wan_t2v_distill_4step_cfg_lora.json \
  --port 8000 \
  --nproc_per_node 1
```

## 🔧 LoRA Extraction Tool

Use `tools/extract/lora_extractor.py` to extract LoRA weights from the difference between two models.

### Basic Usage

```bash
python tools/extract/lora_extractor.py \
  --source-model /path/to/base/model \
  --target-model /path/to/finetuned/model \
  --output /path/to/extracted/lora.safetensors \
  --rank 32
```

### Parameter Description

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `--source-model` | str | ✅ | - | Base model path |
| `--target-model` | str | ✅ | - | Fine-tuned model path |
| `--output` | str | ✅ | - | Output LoRA file path |
| `--source-type` | str | ❌ | `safetensors` | Base model format (`safetensors`/`pytorch`) |
| `--target-type` | str | ❌ | `safetensors` | Fine-tuned model format (`safetensors`/`pytorch`) |
| `--output-format` | str | ❌ | `safetensors` | Output format (`safetensors`/`pytorch`) |
| `--rank` | int | ❌ | `32` | LoRA rank value |
| `--output-dtype` | str | ❌ | `bf16` | Output data type |
| `--diff-only` | bool | ❌ | `False` | Save weight differences only, without LoRA decomposition |

### Advanced Usage Examples

**Extract High-Rank LoRA:**
```bash
python tools/extract/lora_extractor.py \
  --source-model /path/to/base/model \
  --target-model /path/to/finetuned/model \
  --output /path/to/high_rank_lora.safetensors \
  --rank 64 \
  --output-dtype fp16
```

**Save Weight Differences Only:**
```bash
python tools/extract/lora_extractor.py \
  --source-model /path/to/base/model \
  --target-model /path/to/finetuned/model \
  --output /path/to/weight_diff.safetensors \
  --diff-only
```

## 🔀 LoRA Merging Tool

Use `tools/extract/lora_merger.py` to merge LoRA weights into the base model for subsequent quantization and other operations.

### Basic Usage

```bash
python tools/extract/lora_merger.py \
  --source-model /path/to/base/model \
  --lora-model /path/to/lora.safetensors \
  --output /path/to/merged/model.safetensors \
  --alpha 1.0
```

### Parameter Description

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `--source-model` | str | ✅ | - | Base model path |
| `--lora-model` | str | ✅ | - | LoRA weights path |
| `--output` | str | ✅ | - | Output merged model path |
| `--source-type` | str | ❌ | `safetensors` | Base model format |
| `--lora-type` | str | ❌ | `safetensors` | LoRA weights format |
| `--output-format` | str | ❌ | `safetensors` | Output format |
| `--alpha` | float | ❌ | `1.0` | LoRA merge strength |
| `--output-dtype` | str | ❌ | `bf16` | Output data type |

### Advanced Usage Examples

**Partial Strength Merging:**
```bash
python tools/extract/lora_merger.py \
  --source-model /path/to/base/model \
  --lora-model /path/to/lora.safetensors \
  --output /path/to/merged_model.safetensors \
  --alpha 0.7 \
  --output-dtype fp32
```

**Multi-Format Support:**
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

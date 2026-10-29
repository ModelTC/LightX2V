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
- `lora_merge_force_fp32`: Defaults to `false`, preserving the existing merge precision. When `true`, load LoRA weights, compute deltas, and add them to base weights in FP32, then cast back to each layer's original dtype. This only affects static merging; dynamic LoRA (`lora_dynamic_apply=true`) is unchanged. Merging uses additional temporary memory per layer. Multiple LoRAs are merged and cast back individually in configuration order.

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
bash scripts/wan/inference/distill/run_wan_t2v_distill_lora.sh

# I2V LoRA Inference
bash scripts/wan/inference/distill/run_wan_i2v_distill_lora.sh
```

### Using LoRA in API Service

Specify through the [config file](https://github.com/ModelTC/lightx2v/blob/main/configs/distill/wan21/wan_t2v_distill_lora_4step_cfg.json), then modify the startup command in [scripts/server/start_server.sh](https://github.com/ModelTC/lightx2v/blob/main/scripts/server/start_server.sh):

```bash
python -m lightx2v.api_server \
  --model_cls wan2.1 \
  --task t2v \
  --model_path $model_path \
  --config_json ${lightx2v_path}/configs/distill/wan21/wan_t2v_distill_lora_4step_cfg.json \
  --port 8000 \
  --nproc_per_node 1
```

## 🔧 LoRA Extraction Tool

Use `tools/extract/lora_extractor.py` to extract selected Linear weight deltas with FP32 SVD, configured through JSON or YAML. Edit the model/output paths before running. Relative paths use the command's working directory; `~` and environment variables are expanded.

For calibration-based Qwen-Image 2.1 reconstruction, use the separate `tools/extract/lora_reconstructor.py` entry point. It supports Q-Diffusion-style block reconstruction and TFMQ-style temporal reconstruction followed by block reconstruction, with single-GPU or torchrun DDP execution. The supplied configs use rank 64 / alpha 8 and preserve a separate SVD initialization LoRA. See [the reconstruction guide](../../../../tools/extract/RECONSTRUCTION.md) for calibration collection, algorithm details and commands.

### Basic Usage

```bash
python tools/extract/lora_extractor.py \
  --config tools/extract/configs/wan.json

python tools/extract/lora_extractor.py \
  --config tools/extract/configs/qwen_image_21.json
```

### Parameter Description

| Config field | Default | Description |
|--------------|---------|-------------|
| `source_model` | Required | Local base model path |
| `target_model` | Required | Fine-tuned model, exported student, or raw student checkpoint |
| `output` | Required | Output `.safetensors` file |
| `layers` | Required | Module names without `.weight`; glob patterns such as `*` are supported |
| `rank` | `32` | LoRA rank, capped to each matrix's dimensions |
| `alpha` | rank | Adapter alpha; factors compensate for `alpha / effective_rank` to preserve the reconstructed delta |
| `dtype` | `bf16` | Factor dtype: `bf16` / `fp16` / `fp32` or full names; alpha scalars stay FP32 |
| `device` | `auto` | CUDA when available, otherwise CPU; also accepts `cuda:0`, etc. |
| `key_prefix` | `auto` | `transformer` / `diffusion_model`; auto selects from model keys |

Inputs automatically recognize pipeline `transformer/` directories, standalone transformers, single/sharded safetensors, PyTorch state dicts, and the train/RL checkpoints described below. No `--source-type` or `--target-type` is needed. Model paths are local; the tool does not download models.

The Wan T2V config selects Attention, FFN, text/time embedding and output Linear layers, handling both native and Diffusers Wan names. Add I2V-specific layers to `layers` when needed. The Qwen config selects 224 block Linear layers and 8 global Linear layers, without normalization differences. Fused Qwen `gate_up` weights are exposed as separate `gate_layer` / `proj` adapters, using the ComfyUI-compatible `transformer.` prefix.

All selected matrices, including small ones, use low-rank decomposition. `--diff-only` has been removed; bias, normalization and modulation parameter differences are not automatically exported. Identical matrices are skipped. Unmatched patterns, missing target layers and shape mismatches fail without saving a partial LoRA.

### Advanced Usage Examples

**Extract High-Rank LoRA:**

```bash
python tools/extract/lora_extractor.py \
  --config tools/extract/configs/qwen_image_21.json \
  --target-model /path/to/exported/student \
  --output /path/to/high_rank_lora.safetensors \
  --rank 64 \
  --alpha 64 \
  --output-dtype fp16
```

CLI arguments override config fields. Overriding only rank preserves an explicitly configured alpha.

### Export train / RL students

`export_student.py` exports full student weights on CPU into a standalone transformer directory with a shard index. It does not copy text encoders, VAEs or sampling configurations. The result can be passed directly as `target_model`.

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

Train inputs include `model_state.pt`, `student_consolidated/`, `model` in `dist_state/student`, and `student_model` or `model` in `dist_state`. Optimizer and fake-model states are not loaded. RL supports one-dimensional FSDP2 DTensor shards and full replicas under `actor/`; use a modern PyTorch version compatible with the training checkpoint. Adapter-only training checkpoints must first be merged into full weights.

Export preserves dtype by default. `--max-shard-size-gb` defaults to 2; individual tensors are not split and may exceed this limit. `--base-model` validates keys/shapes and supplies `config.json`; it can be omitted if the input already contains that config. Use `--fill-missing-from-base` explicitly for intentionally frozen parameters omitted from DCP; missing parameters otherwise fail. The output directory must not exist, and failed exports leave no partial result. Safetensors/DCP weights are read per tensor; modern PyTorch memory-maps RL rank files on CPU. No distributed training launch is required for export.

## 🔀 LoRA Merging Tool

Use `tools/extract/lora_merger.py` to merge LoRA weights into the base model for subsequent quantization and other operations.

The merger supports `diffusion_model.` and `transformer.` prefixes and applies each adapter's saved `alpha / rank`. CLI `--alpha` remains the overall merge strength, defaulting to 1. Legacy files without alpha retain unit scaling.

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

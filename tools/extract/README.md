# LoRA extraction tools

[中文说明](README_zh.md)

Extract a LoRA so that **base + LoRA approximates the full target model**. Run the commands below from the repository root.

## Export student weights

`export_student.py` exports full student/actor weights from `lightx2v_train` or `lightx2v_rl` checkpoints on CPU. It supports DCP, RL FSDP, PyTorch state dicts and safetensors.

```bash
python tools/extract/export_student.py \
  --checkpoint /path/to/training_checkpoint \
  --base-model lightx2v_train/models/Qwen-Image-2.1 \
  --output outputs/student_transformer
```

- `--checkpoint`: training checkpoint or full model weights to export.
- `--base-model`: matching base model, used to obtain the transformer config and validate weight names/shapes.
- `--output`: a new transformer directory; it must not already exist. Contains safetensors weights, their index, `config.json` and an export manifest.
- Optional: `--dtype bf16` converts export precision (default: preserve original precision); `--max-shard-size-gb` controls shard size (default: 2). Use `--fill-missing-from-base` only to restore frozen parameters intentionally omitted from the checkpoint.

The output contains full transformer weights, not a LoRA or a complete inference pipeline. Skip this step if the required full weights are already available.

## Extract a LoRA with SVD

`lora_extractor.py` decomposes the target-minus-base weight differences with FP32 SVD; no calibration data or training is required.

```bash
python tools/extract/lora_extractor.py \
  --config tools/extract/configs/qwen_image_21.json
```

Edit these config fields before running:

- `source_model`, `target_model`, `output`: original base, full target weights to approximate, and output LoRA `.safetensors` path.
- `rank`, `alpha`, `dtype`: LoRA rank, scaling and export precision. The Qwen-Image 2.1 example uses rank 64 and alpha 8.
- `layers`: Linear layer names or wildcard patterns to extract.

To use the weights exported above, append `--target-model outputs/student_transformer`. A Wan example is available at [configs/wan.json](configs/wan.json). For `lora_merger.py` usage, see the [deployment guide](../../docs/EN/source/deploy_guides/lora_deploy.md).

## Reconstruct a LoRA with TFMQ

For `lora_reconstructor.py` usage, calibration data and configuration, see [RECONSTRUCTION.md](RECONSTRUCTION.md).

## Qwen-Image 2.1 RL comparison

Same base/target, rank 64, alpha 8; 512 T2I + 256 single-reference + 256 multi-reference held-out requests, four teacher steps each. MSE compares the final target-image velocity output, averaging each request/step equally.

| Method | Overall MSE |
|---|---:|
| SVD | 0.0097887147 |
| **TFMQ** | **0.0018547516** |
| Online joint fine-tuning | 0.0038252176 |

TFMQ reduced MSE by **81.1% versus SVD** and **51.5% versus online joint fine-tuning** in this experiment. These are output-fitting metrics, not image-quality scores.

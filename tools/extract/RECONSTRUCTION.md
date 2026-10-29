# Qwen-Image 2.1 LoRA reconstruction

[中文版](RECONSTRUCTION_zh.md)

Extract a LoRA so that **base + LoRA approximates the full target model**. The base stays frozen; reconstruction trains only LoRA factors using target sampling trajectories. The supplied configs select 232 Linear groups with rank 64 and alpha 8, without exporting normalization weight differences.

## Configurations

| Config | Time-condition reconstruction | Input mixing |
|---|---|---|
| [qdiff](configs/qwen_image_21_reconstruct_qdiff.json) | Fit four time-path Linear layers sequentially | Off |
| [tfmq](configs/qwen_image_21_reconstruct_tfmq.json) | Jointly fit modulation and final-scale outputs | Off |
| [tfmq_qdrop](configs/qwen_image_21_reconstruct_tfmq_qdrop.json) | Same as `tfmq` | 50% student / 50% teacher |

These adapt ideas from [Q-Diffusion](https://github.com/Xiuyu-Li/q-diffusion), [TFMQ-DM](https://github.com/ModelTC/TFMQ-DM) and [QDrop](https://github.com/wimh966/QDrop) to LoRA reconstruction, without quantization. All use this order:

```text
Time-condition modules → input projections → block_00 … block_31 → output projection
```

The time path is frozen after its reconstruction. Teacher and student features advance through separate caches; gradients never cross unit boundaries.

## Run

Run commands from the repository root. Use an environment with `diffusers.models.transformers.transformer_qwenimage21`; collection also requires the LightX2V inference environment. Set model and data paths in the selected config first. Relative paths resolve from the working directory.

**Reuse existing calibration traces and reconstruct on eight GPUs:**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 tools/extract/lora_reconstructor.py \
  --config tools/extract/configs/qwen_image_21_reconstruct_tfmq_qdrop.json
```

Append an option when needed:

| Option | Action |
|---|---|
| `--collect-calibration` | Collect target trajectories only |
| `--collect-and-reconstruct` | Collect a new dataset, then reconstruct after collection finishes |
| `--collect-calibration --dry-run` | Check sample selection and reference paths without inference |

For one GPU, replace `torchrun --standalone --nproc_per_node=8` with `python` and set `CUDA_VISIBLE_DEVICES=0`. To switch algorithms, change the config path.

Collection runs independent inference workers with block CPU offload. Reconstruction uses DDP for the current unit's LoRA parameters and loads teacher/student units sequentially; it does not require FSDP. Workers need shared access to models, data and output directories. Each active unit and its activations must fit on one GPU.

## Calibration data and settings

The configs collect **512 T2I + 256 single-reference + 256 multi-reference requests**. Four denoising steps per request produce **4,096 reconstruction samples**. Each `.pt` trajectory stores noisy latents, conditions and timesteps; block targets are computed separately during reconstruction.

| Field | Meaning / supplied value |
|---|---|
| `source_model`, `target_model` | Base and full target weights; exported student weights are also supported |
| `output`, `layers`, `rank`, `alpha` | LoRA output, selected layers, rank 64, alpha 8 |
| `calibration` | Existing trace file/directory; also the collection destination |
| `calibration_manifest` | Request manifest, default `lightx2v_train/data/opengpt4o_qwen_image21/train.jsonl` |
| `calibration_counts` | Request counts before timestep expansion |
| `calibration_inference_config` | Native inference settings; supplied config uses four steps |
| `iterations`, `temporal_iterations` | 1,000 optimizer steps per ordinary/time reconstruction unit |
| `batch_size` | Global batch 8: one sample per GPU on eight GPUs |
| `learning_rate`, `gradient_clip` | `5e-5`, `1.0` |
| `compute_dtype` | BF16 computation; LoRA master parameters, optimizer state and MSE arithmetic remain FP32 |
| `dtype` | Saved LoRA precision, default FP32 |
| `log_every` | Update the current iteration's batch MSE every 25 iterations |
| `seed` | Seed for request selection and reconstruction sampling |
| `work_dir`, `keep_cache` | Temporary reconstruction directory; remove intermediate caches by default |

On fewer GPUs, gradient accumulation is `batch_size / world_size`, which must be an integer. Training samples cached records randomly rather than running fixed epochs. Time reconstruction uses the observed timesteps plus condition `t=0`.

Collection requires `target_model` to be a complete inference pipeline. For a transformer-only exported target, supply traces collected separately. Match the collection schedule to the target model. `calibration_selection.json` records selected files and completion status; reconstruction rejects incomplete collections.

## QDrop input mixing

Both algorithms support these fields; only the `tfmq_qdrop` config enables mixing:

```json
{
  "qdrop_input_mix": true,
  "qdrop_student_prob": 0.5
}
```

During block/output training, each hidden-state element is independently taken from the student prefix with probability `qdrop_student_prob`, otherwise from the teacher prefix. A new mask is sampled each microstep. `0` selects only teacher features; `1` selects only student features.

Targets always remain teacher outputs. Time conditioning and the input/time units are unchanged. **Full evaluation and propagation to the next block always use pure student inputs.** Mixing needs extra temporary disk space for teacher inputs.

## MSE, outputs and cache cleanup

Each unit is evaluated on **all its calibration samples before and after training**. This is the training calibration set, not a separate validation set. Teacher targets and student-prefix inputs stay fixed during that unit's training. If final MSE worsens, the initial factors are restored. `MSE before → after` reports the retained result; progress-bar MSE is the current training batch's loss. Evaluate generated images on held-out requests separately.

For `tfmq_qdrop`, the default paths are:

```text
outputs/qwen_image_21_calibration/                         # retained raw trajectories
    calibration_selection.json
    collection_<id>/rank_<rank>/*.pt                       # parallel collection
outputs/qwen_image_21_tfmq_qdrop_r64_a8.svd.safetensors     # retained SVD baseline
outputs/qwen_image_21_tfmq_qdrop_r64_a8.safetensors         # final LoRA
outputs/reconstruction_tfmq_qdrop/<id>/
    config.json
    calibration.json
    metrics.jsonl
    rank_<rank>/                                          # temporary feature caches
```

With `initialization: "svd"`, the initial LoRA is saved before training; `svd_output` can override its path. With `keep_cache: false`, previous-stage feature caches are removed as reconstruction advances, and remaining caches and per-rank initialization shards are removed after success. **Raw trajectories, the SVD baseline, final LoRA and run metrics are retained.**

Interrupted runs restart reconstruction; optimizer/stage resume is not supported. To reuse an initialization, set `initialization: "lora"` and `initial_lora` to a matching exported LoRA. Use distinct output paths for different experiments.

# TFMQ LoRA reconstruction

[中文说明](RECONSTRUCTION_zh.md) · [Tools and experiment results](README.md)

This guide uses Qwen-Image 2.1 as an example. The current reconstruction implementation supports Qwen-Image 2.1.

Freeze the base and optimize only LoRA factors. The example configs select 232 Linear groups with rank 64 and alpha 8. `lora_reconstructor.py` detects whether `target_model` is a complete model directory, an exported transformer directory (including shards), or a single safetensors file and uses the same TFMQ workflow for all three.

First jointly reconstruct the time path's modulation and final-scale outputs, then freeze it. Reconstruct the input projections, blocks 00–31 and output projection in order, minimizing `MSE(S(xs, ms), T(xt, mt))`: each model uses its own prefix features and time conditions. Gradients do not cross reconstruction units.

## Run

Run from the repository root after editing the model and data paths in the config. Install the LightX2V base dependencies in a CUDA-enabled PyTorch environment, then install the reconstruction dependencies:

```bash
pip install -r tools/extract/requirements-reconstruction.txt
python -c "from diffusers.models.transformers import transformer_qwenimage21; print('Qwen-Image 2.1 backend available')"
```

The verified environment uses Diffusers `0.41.0.dev0` at commit `62cde2b0115d57492d2f5ecefab54994797df2e3` and Transformers `5.14.1`, pinned in the requirements file. No training-framework installation is required.

```bash
# Target is a complete model directory: collect trajectories, then reconstruct.
torchrun --standalone --nproc_per_node=8 tools/extract/lora_reconstructor.py \
  --config tools/extract/configs/qwen_image_21_reconstruct_tfmq.json \
  --collect-and-reconstruct

# Target is an exported transformer directory or a single .safetensors file.
torchrun --standalone --nproc_per_node=8 tools/extract/lora_reconstructor.py \
  --config tools/extract/configs/qwen_image_21_reconstruct_checkpoint_tfmq.json \
  --collect-and-reconstruct
```

### Command arguments

`lora_reconstructor.py`:

- `--config <path>`: required reconstruction config, in JSON or YAML.
- `--calibration <path>`: override the config's calibration file or directory; during collection, this is the trajectory output directory.
- `--output <path>`: override the final LoRA `.safetensors` output path.
- `--work-dir <path>`: override the directory for run logs and temporary reconstruction caches.
- `--device <device>`: override the reconstruction device, such as `auto`, `cuda:0` or `cpu`. With torchrun, use `auto` so each worker uses its local GPU. This option does not select collection GPUs.
- `--collect-calibration`: prepare calibration trajectories only. All target formats automatically reuse matching trajectories when available.
- `--collect-and-reconstruct`: prepare trajectories, then reconstruct in the same launch.
- `--reuse-calibration`: require existing calibration data and reconstruct without collecting. All target formats validate provenance and the file inventory, and can select a smaller subset. Missing or incompatible data raises an error.
- `--dry-run`: check collection or reuse without inference or training; requires one of the three workflow flags above. Checkpoint preparation may still write pipeline links/config.
- `--help`: display available arguments.

The three workflow flags are mutually exclusive. With none set, reconstruct directly from `calibration`, without collection preparation or checkpoint subset selection.

### Runtime behavior

For an exported transformer directory or file, the collector combines its weights with the base's frozen components. Directory inputs support single-file and indexed/sharded safetensors exports. Sampling uses `calibration_inference_config`; optional `sample_sigmas` and `scheduler` in the checkpoint's adjacent `inference_config.json` override those settings. Collection currently requires CFG=1.

New collections record the resolved model file paths, sizes/mtimes, JSON configuration digests, request manifest path/digest, seed, effective inference settings and collection dtype in the selection inventory. Reuse checks these fields before reading trajectories. Old single-file checkpoint inventories remain supported through their original provenance checks; old collections without verifiable provenance must be recollected in a separate directory. Model weights are identified by file metadata, not full-file hashes.

Collection uses independent BF16 inference workers with CPU offload; reconstruction uses DDP, loading one unit at a time. Each unit and its activations must fit on one GPU, and workers must share the model/data/output filesystem. For one GPU, replace the torchrun prefix with `python`.

## Configuration and outputs

The extraction objective is **`source_model` + exported LoRA ≈ `target_model`**.

- `source_model`: the frozen original base; load the exported LoRA onto this model for inference. Both examples use `lightx2v_train/models/Qwen-Image-2.1`.
- `target_model`: the full model weights to approximate. This teacher supplies calibration trajectories and reconstruction targets. Specify a complete model directory, an exported transformer directory or a weight file, not a LoRA file; RL training is not required.
- `output`: the final LoRA `.safetensors` path, to be loaded onto `source_model`.

The two example targets are:

| Config | `target_model` |
|---|---|
| [qwen_image_21_reconstruct_tfmq.json](configs/qwen_image_21_reconstruct_tfmq.json) | `lightx2v_train/models/Qwen-Image-2.1-Lightning-v2-base` |
| [qwen_image_21_reconstruct_checkpoint_tfmq.json](configs/qwen_image_21_reconstruct_checkpoint_tfmq.json) | `lightx2v_train/models/Qwen-Image-2.1-Lightning-v2-RL/diffusion_pytorch_model.safetensors` |

The first target's `-base` suffix is part of its model name; the original base for extraction is still specified by `source_model`. Both configs use the same algorithm; dataset sizes and iteration counts are experiment settings.

Other key parameters:

| Parameter | Meaning |
|---|---|
| `layers` | Linear layer names to extract and optimize, supporting `*` wildcards. The examples cover 232 Linear groups. |
| `rank` / `alpha` | LoRA rank and scaling, set to 64 / 8 in the examples. The weight delta is `(alpha / rank) × up @ down`. SVD initialization compensates for this scaling; alpha=8 does not mean fitting only 1/8 of the target delta. |
| `initialization` | `svd`: initialize from the target-minus-source weight delta and save it before training; `lora`: load an existing adapter from `initial_lora`. `random` is also supported. |
| `svd_output` / `initial_lora` | The former is the separate SVD initialization output, defaulting to `output` with its `.safetensors` suffix replaced by `.svd.safetensors`. Set the latter only with `initialization: lora`. |
| `calibration_manifest` | Request manifest containing prompts, reference images and other request fields, used to select collection requests. |
| `calibration_counts` | Request counts for `t2i`, `i2i_single` and `i2i_multi`. All captured timesteps are used, each contributing a reconstruction record: for example, 1024 requests × 4 steps = 4096 records. |
| `calibration_inference_config` | Teacher inference settings for collection, including sampling steps and resolution. See above for sampling overrides from metadata beside a standalone checkpoint. |
| `calibration` | Where raw calibration trajectories are saved or read, separate from the request manifest. These trajectories are retained. |
| `iterations` / `temporal_iterations` | Optimizer updates per ordinary reconstruction unit and for the time path, respectively; these are not dataset epochs. |
| `batch_size` / `learning_rate` | Global batch and LoRA learning rate, set to 8 / `5e-5` in the examples. Each GPU processes one sample per forward and accumulates `batch_size / WORLD_SIZE` microsteps; batch must be divisible by GPU count. |
| `compute_dtype` / `dtype` | The former controls reconstruction forward precision (`bf16` or `fp32`), while LoRA master parameters and MSE remain FP32. The latter controls export precision. |
| `checkpointing` | Enable gradient checkpointing inside reconstruction units, recomputing activations during backward to reduce memory. This does not save training checkpoints. |
| `work_dir` | Stores run records and temporary unit caches. Previous-unit caches are deleted as reconstruction advances; remaining temporary caches are deleted on successful completion. Raw calibration trajectories and exported weights are retained. |
| `log_every` / `save_iteration_losses` | The former controls how often minibatch MSE is displayed; the latter enables per-iteration `iteration_losses.jsonl` output. |

`work_dir/<run>/` stores the config and per-unit before/after MSE in `metrics.jsonl`. This MSE uses the full calibration cache for that unit; a worse result restores its initial LoRA. Restarting begins reconstruction afresh, without resuming optimizer state.

Use the inference framework to validate generated images.

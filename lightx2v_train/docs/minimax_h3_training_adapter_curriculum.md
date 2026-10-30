# H3 TrainingAdapter experiments and new-then-old data passes

The DiffSynth TrainingAdapter is a DeCFG LoRA for fine-tuning an already
CFG-distilled H3 base. Its official SFT workflow adds the preset during
training and removes it for inference. It is not a four/eight-step speed LoRA,
and the author does not provide a DMD/PDMD recipe for it.
[Official model card](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-TrainingAdapter)
and [official SFT example](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-Ref2VA.sh).

The new DMD/PDMD configurations leave this experiment **disabled by default**.
When explicitly enabled, DeCFG is applied only to the **fake and teacher score
models**. The teacher remains frozen; the full fake starts from the preset
base and continues training. The student uses the original H3 base plus its
learned LoRA for rollout, gradient-enabled prediction and inference. The
student LoRA keeps its own rank/alpha and its export does not include DeCFG.
**Do not load the DeCFG preset for student inference.** Export metadata records
the score-preset context separately from the student base dependency.

This score-side adaptation keeps the deployed generator identical to the
student used during distillation. It is our dCM/DMD/PDMD experiment, not a
published DiffSynth recipe: DiffSynth's documented SFT procedure loads DeCFG
while training a new LoRA and discards the preset at inference. No official
dCM/PDMD recipe establishes where to put it. Checkpoints from the previous
student-side-preset experiment must not be resumed under this score-side
recipe. Enabled launches now use a `ta1_scoredecfg` output tag.

## Download on the new machine

Sync the updated code first, then run once:

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate

H3_PYTHON="$VIRTUAL_ENV/bin/python" \
H3_TRAINING_ADAPTER_DIR=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3-TrainingAdapter \
bash lightx2v_train/scripts/download_minimax_h3_training_adapter.sh
```

This downloads the two Comfy-QKV-layout files required by these recipes,
approximately 1.24 GB total:

- FL2VA: `model_for_comfy_dit.safetensors`.
- Ref2VA: `model_ref2va_for_comfy_dit.safetensors`.

The helper reuses ModelScope's completed/partial download cache, installs the
SDK only when missing, writes `download.log`, and validates both files before
publishing `download.complete.json` with SHA256 hashes. Downloading does not
enable the adapter in training.

## Data order

Each data cycle exhausts the new source, then exhausts the old source, then
repeats with a deterministic shuffle. Sources are not mixed within a phase.
The last global batch of a phase repeats boundary samples if needed so every
rank follows the same phase and every original row is visited. Checkpoints
save the absolute data cursor and source fingerprints for exact continuation.

- FL2V new source: `minimax_h3_self_generated_latents_124_bf16/t2av/condition_metadata.jsonl`.
  It contains all successfully processed native-resolution T2AV samples.
  The old source is `minimax_h3_dmd_cache_5s_768p/task_aware_view_3task_portrait`,
  with T2AV/I2AV/FL2AV task queues. Queues rotate while available, then consume
  their remaining rows instead of dropping a large task's tail.
- Ref2V new source: `minimax_h3_self_generated_latents_124_bf16/ref2av/condition_metadata.jsonl`.
  It retains images, silent videos and audio-video references at native
  resolution. The old source remains the Omni image-only cache. Old Ref rows
  are grouped by reference count and ordered by compute cost, with full row
  coverage.

These are online DMD/PDMD conditioning streams. The source dataset's clean
target video/audio latent files are not needed by these losses. The earlier
filtered root CM manifest remains available for the separate CM objective.

## Four-node, 32-GPU launch

ACP provides the same `MASTER_ADDR` and `MASTER_PORT` on four workers. Export
the following on each worker, then choose **one** of the launch commands.

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate

export H3_CODE_ROOT="$PWD"
export H3_PYTHON="$VIRTUAL_ENV/bin/python"
export H3_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3
export H3_SELFGEN_CACHE=/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_self_generated_latents_124_bf16
export H3_FL2AV_CACHE=/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_dmd_cache_5s_768p/task_aware_view_3task_portrait
export H3_REF2AV_CACHE=/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/latent_match124_bf16/metadata.jsonl
export KERNELS_CACHE=/mnt/lm_data_afs/gushiqiao/cache/kernels
export H3_TRAINING_ADAPTER_ENABLE=0
```

```bash
# Choose one; execute that same line on all four ACP workers.
bash lightx2v_train/scripts/run_minimax_h3_new_then_old_fsdp32_32gpu_acp.sh fl2av pdmd
bash lightx2v_train/scripts/run_minimax_h3_new_then_old_fsdp32_32gpu_acp.sh fl2av dmd
bash lightx2v_train/scripts/run_minimax_h3_new_then_old_fsdp32_32gpu_acp.sh ref2av pdmd
bash lightx2v_train/scripts/run_minimax_h3_new_then_old_fsdp32_32gpu_acp.sh ref2av dmd
```

All four recipes use eight rollout steps, five fake updates per student update,
student regular LoRA rank/alpha 128/128, block/output AdaLN 16/16, FSDP32,
student LoRA FP32, full fake FP32, base/teacher BF16, BF16 FSDP computation and
FP32 reduction. FL2V video/audio shifts are 6/3; Ref2V shifts are 12/3.

Official PDMD counts each role optimizer update: `max_train_iters=15000`
means 2500 student plus 12500 fake updates, saving every 300 role updates.
Ordinary DMD counts outer iterations: `max_train_iters=2500` means the same
2500 student plus 12500 fake updates, saving every 50 outer iterations.

The default output/RDZV names distinguish task, method and adapter on/off
(`ta0`/`ta1_scoredecfg`). A changed adapter file, scale or data recipe requires a new output
directory. Set `H3_CURRICULUM_OUTPUT` and `H3_RDZV_ID` for additional trials.

An optional student-only warm start is selected with `H3_STUDENT_INIT`. It
defaults to empty, meaning a fresh H3 base. For the earlier FL2V dCM EMA:

```bash
export H3_STUDENT_INIT=/mnt/lm_data_afs/gushiqiao/outputs/h3_fl2av_dcm1700_ema_init
```

Use a Ref2V-trained checkpoint for a Ref2V warm start, with the exact same LoRA
architecture. This option transfers only student weights, not optimizer/data
cursors from the source experiment.

To run an explicit adapter experiment after downloading it:

```bash
export H3_TRAINING_ADAPTER_ENABLE=1
export H3_TRAINING_ADAPTER_SCALE=1.0
# The launcher selects the corresponding FL2VA/Ref2VA Comfy-layout file.
```

`H3_TRAINING_ADAPTER_PATH` overrides that file; `H3_NEW_CACHE` overrides the new
source manifest. `H3_CURRICULUM_CONFIG` is the dedicated config override, so
stale older `H3_CONFIG_PATH`/PDMD launcher variables cannot select another
recipe. Append `--dry-run` for CPU-only metadata and command checks. The new
cache must have a valid completed preprocessing receipt before launch.

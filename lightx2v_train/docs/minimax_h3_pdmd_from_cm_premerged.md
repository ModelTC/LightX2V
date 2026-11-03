# Continue H3 PDMD from completed CM with offline DeCFG score checkpoints

This route skips CM entirely. It initializes the live student from the completed
CM1500 EMA LoRA on the **original H3 base**. CM's `student_ema` does not contain
DeCFG: its `training_adapter.json` declares `enabled: false` and
`export_semantics: learned_lora_only`. The nested score adapter is audit metadata.

Teacher and fake initialize from a separately materialized **original H3 +
DeCFG** checkpoint. Online adapter loading is disabled for all three roles, so
DeCFG is applied once. Final inference uses the original base + learned student
LoRA, without DeCFG. Fake does not initialize from the CM student LoRA.

The recipe remains official PDMD8, 4 ACP workers x8 GPUs, video/audio shifts12/3,
fake-first5:1, student/fake LR2e-5/4e-6. Regular LoRA rank/alpha128/128 and
block/output AdaLN16/16 are unchanged. PDMD runs1500000 single-role optimizer
updates; saves every100 role updates, retaining the latest2 checkpoints. A100
update interval includes16 or17 student updates. Full fake state is saved too.
PDMD counters, optimizer states, fake and curriculum cursor start fresh; later
relaunches with the same new output resume PDMD normally.

## Merge once on one worker

First sync these scripts, new configs and model loader changes to the new
machine. Run the merge once, outside distributed training, on a worker with
one available GPU. CPU mode is also supported with `--device cpu`.

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate

mkdir -p /mnt/lm_data_afs/gushiqiao/models/logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0 "$VIRTUAL_ENV/bin/python" -u \
  lightx2v_train/scripts/merge_minimax_h3_decfg.py \
  --model-path /mnt/lm_data_afs/gushiqiao/models/MiniMax-H3 \
  --adapter-dir /mnt/lm_data_afs/gushiqiao/models/MiniMax-H3-TrainingAdapter \
  --output-dir /mnt/lm_data_afs/gushiqiao/models/MiniMax-H3-DeCFG-score \
  --scale 1.0 --device cuda:0 \
  2>&1 | tee /mnt/lm_data_afs/gushiqiao/models/logs/h3_merge_decfg_score.log
```

Both FL2AV (`transformer/`) and Ref2AV (`transformer_ref/`) are merged by default,
using their corresponding adapter files. Adapted weights are stored FP32 and
other tensor dtypes remain unchanged. Original adapted targets must be BF16:
this preserves the old online FP32 fake update and BF16 teacher casting order.
The merger writes row tiles directly to safetensors and never constructs a
complete model or shard in GPU/CPU RAM. It publishes `decfg_merged.json` only
after all output shards are complete. It refuses existing outputs, malformed
adapter coverage and double merges. Shared encoders/VAEs use symlinks to the
original model root; keep that root available.
For the current converted H3 checkpoints, the two materialized partitions total
approximately198 GiB; shared components are not copied again.

## Common environment on every worker

Allocate a separate 4-node x8-GPU ACP job for each task. Each worker in one job
executes the same task command. ACP supplies `MASTER_ADDR` and `MASTER_PORT`.

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate
export H3_CODE_ROOT="$PWD"
export H3_PYTHON="$VIRTUAL_ENV/bin/python"
export H3_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3
export H3_SCORE_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3-DeCFG-score
export H3_SELFGEN_CACHE=/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_self_generated_latents_124_bf16
export H3_FL2AV_CACHE=/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_dmd_cache_5s_768p/task_aware_view_3task_portrait
export H3_REF2AV_CACHE=/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/latent_match124_bf16/metadata.jsonl
export KERNELS_CACHE=/mnt/lm_data_afs/gushiqiao/cache/kernels
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset H3_NEW_CACHE
```

## FL2AV: full new T2AV -> old three tasks -> repeat

The source path below uses the previous restarted experiment's output. If its
actual directory differs, change only `H3_STUDENT_INIT` to that task's completed
`checkpoint-000001500/student_ema`; never substitute the interrupted PDMD path.

```bash
H3_STUDENT_INIT=/mnt/lm_data_afs/gushiqiao/outputs/h3_fl2av_dcm1500_pdmd1500000_restart_ema999_lr2e5_4e6_20261030/cm/train/checkpoint-000001500/student_ema \
H3_PDMD_OUTPUT=/mnt/lm_data_afs/gushiqiao/outputs/h3_fl2av_pdmd8_from_cm1500_premerged_decfg_save100_20261101/train \
H3_RDZV_ID=h3_fl2av_pdmd8_from_cm1500_premerged_decfg_save100_20261101 \
bash lightx2v_train/scripts/run_minimax_h3_pdmd_from_cm_premerged_fsdp32_32gpu_acp.sh fl2av
```

Config: `configs/train/dmd/minimax_h3_fl2av_pdmd1500000_from_cm_premerged_decfg8_save100_fsdp32.yaml`.

## Ref2AV: new image-only -> old image-only -> repeat

```bash
H3_STUDENT_INIT=/mnt/lm_data_afs/gushiqiao/outputs/h3_ref2av_imgonly_dcm1500_pdmd1500000_restart_ema999_lr2e5_4e6_20261030/cm/train/checkpoint-000001500/student_ema \
H3_PDMD_OUTPUT=/mnt/lm_data_afs/gushiqiao/outputs/h3_ref2av_imgonly_pdmd8_from_cm1500_premerged_decfg_save100_20261101/train \
H3_RDZV_ID=h3_ref2av_imgonly_pdmd8_from_cm1500_premerged_decfg_save100_20261101 \
bash lightx2v_train/scripts/run_minimax_h3_pdmd_from_cm_premerged_fsdp32_32gpu_acp.sh ref2av
```

Config: `configs/train/dmd/minimax_h3_ref2av_pdmd1500000_from_cm_premerged_decfg8_save100_fsdp32.yaml`.

The new image-only condition view should already exist from CM. This launcher
does not re-encode data or need the CM clean AV payloads: it validates the
completed CM marker, all32 recorded rank recipes/cursors, student export
components, task/base/LoRA identity, premerged score adapter identity, and both
condition curriculum sources. It checks the actual recorded CM recipe instead
of requiring a byte-identical current CM YAML or restarting CM. This also allows
a completed historical CM EMA decay while retaining its exact provenance.

Appending `--dry-run` performs read-only checks without starting Torch workers.
Actual launch saves immutable `cm_premerged_handoff.json` in the new PDMD output.
It binds source LoRA and rank metadata hashes, config and merged score manifest;
changing any of these on a subsequent launch requires another output directory.
An output containing earlier PDMD checkpoints without this identity is rejected.

Logs are under each `H3_PDMD_OUTPUT/train.log`.


## Ref CM1500 continuation with fake10, shifts12/3

This variant retains the completed Ref CM1500 EMA and image-only NEW -> OLD ->
repeat curriculum and its video/audio shifts12/3. It aligns optimizer settings
with the historical FL2AV run: fake10 -> student1, student/fake LR2e-5/4e-6, Adam betas0/0.9,
weight decay0, epsilon1e-8, clipping1, gradient accumulation1, and8 rollout steps.
Regular LoRA stays128/128 and block/output AdaLN stays16/16. Student uses the
original Ref base; teacher/fake use the already-merged DeCFG score base.

The budget is1500000 role updates (136363 student and1363637 fake). Save every100
role updates and retain2 checkpoints; each interval contains9 or10 student
updates. Use a fresh output for this different recipe. The default fake5 configs
remain unchanged. Run this same command on all four ACP workers:

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate

H3_CODE_ROOT="$PWD" \
H3_PYTHON="$VIRTUAL_ENV/bin/python" \
H3_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3 \
H3_SCORE_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3-DeCFG-score \
H3_STUDENT_INIT=/mnt/lm_data_afs/gushiqiao/outputs/h3_ref2av_imgonly_dcm1500_pdmd1500000_restart_ema999_lr2e5_4e6_20261030/cm/train/checkpoint-000001500/student_ema \
H3_NEW_CACHE=/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_self_generated_latents_124_bf16/ref2av_imageonly/condition_metadata.jsonl \
H3_REF2AV_CACHE=/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/latent_match124_bf16/metadata.jsonl \
H3_PDMD_CONFIG="$PWD/lightx2v_train/configs/train/dmd/minimax_h3_ref2av_pdmd1500000_from_cm_premerged_decfg8_shift12_3_fake10_save100_fsdp32.yaml" \
H3_PDMD_OUTPUT=/mnt/devsft_afs_2/gushiqiao/outputs/h3_ref2av_pdmd8_from_cm1500_decfg_shift12_3_fake10_save100_32gpu_20261103/train \
H3_RDZV_ID=h3_ref2av_pdmd8_from_cm1500_decfg_shift12_3_fake10_save100_32gpu_20261103 \
KERNELS_CACHE=/mnt/lm_data_afs/gushiqiao/cache/kernels \
bash lightx2v_train/scripts/run_minimax_h3_pdmd_from_cm_premerged_fsdp32_32gpu_acp.sh ref2av
```

`H3_STUDENT_INIT` above is the previously recorded Ref CM1500 output path; if the
completed export lives elsewhere, point it at that Ref `student_ema` directory.
The launcher checks all32 recorded CM ranks and the matching DeCFG provenance
before launching training. `H3_PDMD_CONFIG` can also be a path relative to
`H3_CODE_ROOT`. Append `--dry-run` for read-only preflight checks.

# H3 Turbo student EMA to PDMD8

This starts a new PDMD stage from the exported student adapter. It does not
resume Turbo optimizer state, data cursors, fake weights, or the CM curriculum.
The source `checkpoint-000001700` is still in dCM (the original dCM stage ends
at 2000 student updates), and is usable as an initialization checkpoint.

Recipe:

- FL2AV base transformer, T2AV/I2AV/FL2AV three-task conditioning cache.
- 4 ACP workers x 8 GPUs, FSDP2 size 32, no sequence parallel.
- 8 Euler rollout steps; video/audio shifts 6/3; teacher guidance 1.
- Student adapter rank/alpha 128/128; block `adaln_proj.linear` and output
  `norm_out.linear` rank/alpha 16/16. Exactly matches the Turbo adapter layout.
- Turbo EMA initializes the live student. Official PDMD does not maintain a new
  student EMA. Frozen teacher and full trainable fake both load the H3 base.
- Student LR 2e-5, fake LR 4e-6; PDMD Adam betas 0/0.9, weight decay 0.
- 27500 role optimizer updates = 2500 student + 25000 fake updates. Ten fake
  updates precede each student update; student reuses the last fake trajectory.
- Save every 550 role updates (50 student updates), retain two checkpoints.
- Base/teacher BF16, student LoRA master weights FP32, full fake FP32;
  FSDP computation BF16, gradient reduction FP32.

## Preserve the initialization once

Run once before launching the four ACP workers. A running Turbo job retains
only two checkpoints and can prune checkpoint 1700, so keep the initialization
outside that output directory. Copy the entire `student_ema` export to retain
`adapter_config.json`, which records per-layer alpha values.

```bash
mkdir -p /mnt/lm_data_afs/gushiqiao/outputs/h3_fl2av_dcm1700_ema_init
cp -a /mnt/lm_data_afs/gushiqiao/outputs/h3_fl2av_turbo8_cm_t2av_9424_32gpu_20261028/checkpoint-000001700/student_ema/. \
  /mnt/lm_data_afs/gushiqiao/outputs/h3_fl2av_dcm1700_ema_init/
```

## Launch on each ACP worker

Sync the changed code and new configuration/launcher before running this.
ACP supplies the same `MASTER_ADDR` and `MASTER_PORT` to all four workers.
Use a new PDMD output directory. Do not point it at the Turbo output directory.

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate

H3_CODE_ROOT="$PWD" \
H3_PYTHON="$VIRTUAL_ENV/bin/python" \
H3_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3 \
H3_STUDENT_INIT=/mnt/lm_data_afs/gushiqiao/outputs/h3_fl2av_dcm1700_ema_init \
H3_FL2AV_CACHE=/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_dmd_cache_5s_768p/task_aware_view_3task_portrait \
H3_FL2AV_PDMD_OUTPUT=/mnt/lm_data_afs/gushiqiao/outputs/h3_fl2av_pdmd8_from_dcm1700_ema_fake10_32gpu_20261029 \
H3_RDZV_ID=h3_fl2av_pdmd8_from_dcm1700_ema_fake10_32gpu_20261029 \
KERNELS_CACHE=/mnt/lm_data_afs/gushiqiao/cache/kernels \
bash lightx2v_train/scripts/run_minimax_h3_fl2av_pdmd8_from_turbo_ema_fsdp32_32gpu_acp.sh
```

Appending `--dry-run` checks the export header and conditioning metadata and
prints the torchrun command. It does not initialize a GPU model or train.

Startup logs should show `student LoRA warm-start ... weights_only=true`, then
`[pdmd_official]`. A fresh stage starts its PDMD counters at zero. Relaunching
with the same PDMD output resumes a complete PDMD checkpoint; it does not
reapply the initial Turbo export or reset the optimizer.

The adapter loader runs after H3's streamed FSDP initialization so the streamed
base loader cannot overwrite the transferred adapters. It checks adapter keys,
matrix shapes, and saved rank/alpha scaling before loading. Only rank zero reads
the full adapter tensors and broadcasts them into the destination shards; it
does not gather or replace the frozen base.

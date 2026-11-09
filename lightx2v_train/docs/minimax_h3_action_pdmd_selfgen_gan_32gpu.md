# H3 action PDMD -> self-generated PDMD + GAN, 32 GPUs

The launcher runs three stages automatically on four ACP nodes with eight GPUs
per node:

1. Encode the external 2,000 T2AV action prompts into text conditions with 32
   independent GPU shards, validate complete coverage, then release the encoder.
2. Run plain official PDMD for **100 student updates and 500 fake updates**.
   `max_train_iters: 600` counts individual optimizer-role updates. GAN is off.
3. Load that stage's final student and fake weights into PDMD + DMD2-style GAN.
   Both generator conditioning and GAN positive AV samples now come from the
   same filtered self-generated T2AV dataset, with independent sampling streams.

Both training stages use 8-step rollout, video/audio shifts 12/3, fake-first
5:1 updates, FSDP32, BF16 compute with FP32 reductions, and no sequence
parallelism. The student LoRA stays rank/alpha 128/128, with block AdaLN and
`norm_out.linear` at 16/16. Student/fake learning rates stay `2e-5` / `4e-6`;
AdamW betas are `(0, 0.9)`, weight decay 0, and gradient clipping 1. Student EMA
is disabled. Checkpoints are saved every 100 role updates, retaining two per
stage.

Configurations:

- `configs/train/dmd/minimax_h3_t2av_action_pdmd8_shift12_3_r128_a128_adaln16_fsdp32.yaml`
- `configs/train/dmd/minimax_h3_t2av_selfgen_pdmd8_gan_shift12_3_r128_a128_adaln16_fsdp32.yaml`

The first stage reads 500 × 32 = 16,000 fresh prompt samples: eight complete
passes over the 2,000-row pool. Each student update reuses the immediately
preceding fake batch, as in official PDMD. Thus 100 student updates do not
guarantee a student-gradient update for every distinct prompt.

The GAN stage defaults to 15,000 role iterations (2,500 student and 12,500 fake
updates), adjustable through `H3_GAN_MAX_ITERS`. Its generator/discriminator GAN
weights are 0.005/0.01. The classifier shares the fake-score backbone, and the
generator loss backpropagates through frozen discriminator parameters. See
`minimax_h3_action_pdmd_gan_32gpu.md` for the DMD2 objective and H3 adaptations.

## Launch

Run the same command on all four ACP workers. ACP supplies common `MASTER_ADDR`
and `MASTER_PORT`. Preprocessing uses `MASTER_PORT + 1`, phase 1 uses
`MASTER_PORT`, and phase 2 uses `MASTER_PORT + 2`. Override
`H3_GAN_PREPROCESS_PORT` and `H3_GAN_PHASE2_PORT` if necessary.

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate

H3_CODE_ROOT="$PWD" \
H3_PYTHON="$VIRTUAL_ENV/bin/python" \
H3_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3 \
H3_GAN_PROMPTS=/mnt/lm_data_afs/gushiqiao/datasets/data.json \
H3_GAN_INIT_CHECKPOINT=/mnt/lm_data_afs/gushiqiao/outputs/h3_fl2av_pdmd8_from_dcm1700_ema_32gpu_20261029/checkpoint-000005800 \
H3_SELFGEN_CACHE=/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_self_generated_latents_124_bf16/t2av/metadata.jsonl \
H3_STAGED_OUTPUT=/mnt/lm_data_afs/gushiqiao/outputs/h3_action_pdmd100_selfgen_gan_shift12_3_fake5_from5800_32gpu_20261109 \
H3_GAN_MAX_ITERS=15000 \
H3_RDZV_ID=h3_action_pdmd100_selfgen_gan_shift12_3_fake5_from5800_32gpu_20261109 \
KERNELS_CACHE=/mnt/lm_data_afs/gushiqiao/cache/kernels \
bash lightx2v_train/scripts/run_minimax_h3_action_pdmd_selfgen_gan_fsdp32_32gpu_acp.sh
```

Append `--dry-run` to print the three distributed commands without reading model
weights, encoding data, or launching training.

## Data and handoff

The external 2k prompt input defaults to
`/mnt/lm_data_afs/gushiqiao/datasets/data.json`; override `H3_GAN_PROMPTS` to
change it. JSON arrays, `data`/`records`/`examples` wrappers and JSONL content
are supported, including JSONL saved with a `.json` suffix. Prompt preprocessing
does not generate video/audio targets. `H3_GAN_PROMPT_CACHE` can override its
shared cache directory.

Self-generated rows are filtered to T2AV, 124 frames, 768×1344, 24 fps, with
their own caption and complete condition/video/audio latent paths. Ref2AV,
portrait and other resolutions are excluded from this fixed-geometry recipe.
At least 64 eligible rows are required by the FSDP32 stratified sampler. The
launcher publishes immutable task/bucket views under the staged output's
`data/` directory; it does not rewrite the original dataset. Both phase-2
conditioning and GAN positives reference the same eligible rows. These positives
are teacher-generated examples, rather than real-camera videos.

Phase 1 initializes student and fake from checkpoint 5800, with fresh optimizers
and cursors. Phase 2 initializes both weights from
`phase1_action_pdmd/checkpoint-000000600`, with fresh optimizers, cursors and GAN
head. Its teacher remains the original H3 base. The handoff checks all 32 rank
states, the expected recipe, a 500-microbatch cursor, and an empty trajectory
cache at the completed update-cycle boundary.

Each phase supports full automatic resume from its own output. A completed
phase 1 is verified and skipped on restart. Once phase 2 has its own complete
checkpoint, restarting goes directly to phase 2 and no longer needs the initial
5800 checkpoint, action cache, or phase-1 weights. It still needs the phase-2
view and referenced AV/condition files. Keep the original initialization path
strings and output root unchanged when resuming.

Logs and outputs:

```text
$H3_STAGED_OUTPUT/
├── phase1_action_pdmd/train.log
├── phase1_action_pdmd/checkpoint-000000600/
├── phase2_selfgen_gan/train.log
├── phase2_selfgen_gan/checkpoint-*/
├── data/action_prompts/t2av/landscape/metadata.jsonl
├── data/self_generated_t2av/t2av/landscape/metadata.jsonl
├── stages/plan.json
├── stages/phase1_handoff.complete.json
└── logs/launch.*.log
```

Detailed encoding logs are under `$H3_GAN_PROMPT_CACHE/logs/rank-*`.

Cache namespace initialization runs on global rank 0 and broadcasts its result
before encoding. It uses atomic directory publication protection rather than
`flock`, since shared AFS mounts may return `EAGAIN` for blocking file locks.
Workers reuse a matching namespace through a read-only check. Legacy
`.preprocess.lock` files are ignored; existing valid condition caches can be
reused after restarting a failed launch.

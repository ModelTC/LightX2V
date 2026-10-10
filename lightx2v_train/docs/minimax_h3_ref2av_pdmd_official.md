# MiniMax-H3 Ref2AV: source-aligned 4-step PDMD on 32 ACP GPUs

This is an opt-in training path, separate from the existing eight-step PDMD,
legacy DMD, HEAD and DMAD recipes. It follows the released
[ZeamoxWang/pdmd training implementation](https://github.com/ZeamoxWang/pdmd/tree/6b6e10635495a2d9b989fee393235d406d3840c4)
at commit `6b6e10635495a2d9b989fee393235d406d3840c4`, specifically
`training/configs/pdmd_4nfe_544p.json`, `training/train.py`, and
`src/pdmd/{engine,schedule,objectives}.py`. It adapts those equations to
LightX2V's opposite, clean-ward velocity convention and ordered reference
conditions. It is not a reproduction of the published T2AV quality score.

## Recipe and retained overrides

The new configuration is
`configs/train/dmd/minimax_h3_ref2av_omni_imageonly_pdmd4_official_fsdp32.yaml`.
Both `training.dmd.official_pdmd` and
`model.capabilities.distribution_matching.official_pdmd` are enabled, so the
source-aligned update loop and losses are selected together.

| Setting | This recipe |
| --- | --- |
| Update order | Five critic updates, then one student update |
| Training rollout | Full four-step, no-grad Euler rollout; video/audio grid `[1, .75, .5, .25, 0]` |
| Student update | Reuse the immediately preceding critic rollout; one grad-enabled student forward at a randomly selected rollout state |
| Student score query | Shared random ratio inside that step's video/audio intervals; query follows the student's predicted trajectory, without fresh Gaussian re-noising |
| Critic score query | Full rollout endpoint plus fresh Gaussian noise; independent uniform video/audio sigmas; if video sigma > .95, audio sigma is drawn from [.85, 1] |
| Loss | Per-modality MSE without the old 0.5 multiplier, clamped to 5 before weighting by .8 |
| Projection / normalization | Projection denominator epsilon `1e-8`, normalizer floor `1e-5`; nonfinite normalized updates become zero |
| Optimizers | AdamW; student LR `5e-5`, critic LR `1e-5`; betas `(0, .9)`, epsilon `1e-8`, no weight decay; gradient norm clip 1 |
| LoRA initialization | Kaiming-uniform A, zero B in the new opt-in path; old recipes keep Gaussian A |
| Inference shifts | Video/audio 12/3; these are **not** applied to the explicit physical sigmas of this training loop |
| Duration | 250000 single-role optimizer updates; checkpoint every 50 (user overrides) |

The following are intentional user/task overrides, not upstream defaults:

- Student LoRA rank **128**, alpha **8**; critic remains **full-parameter**.
- Student LoRA parameters use **FP32** via
  `training.student.lora.param_dtype: fp32`; BF16 FSDP compute is unchanged.
- Existing Omni image-only reference cache and 124-frame 768p geometry.
  Image-only refers to the conditioning; H3 still generates joint video/audio.
- FSDP2 over **32 ranks**, SP disabled, batch 1/rank and accumulation 1.
  Upstream uses global batch 16 and a shared frozen base with two LoRA roles;
  this port retains separate student, critic and teacher models.
- Existing base precision settings: student transformer parameter setting BF16,
  critic FP32, teacher BF16; running dtype and FSDP compute BF16, gradient
  reduction FP32, autocast disabled. Existing loader-specific FP32 modules
  remain unchanged; these settings do not assert that every tensor is BF16.
- Existing `count_random` data sampler: a global 32-row microbatch has one
  reference-image count; observed counts 1–6 are balanced across a sampler
  epoch. Orientations are shuffled together without a fixed 16/16 quota.
- Only the latest five checkpoints are retained to limit full-critic storage.

## Iteration and data accounting

An iteration means **one optimizer update of one role**, not an outer round
containing six updates. The configured 250000 iterations comprise **208334
critic updates and 41666 student updates**. This duration is a user override,
not the upstream 2500-update default.

Only critic updates consume new dataloader batches. Student updates reuse the
preceding critic conditions, noise and rollout states. With 32 ranks and
accumulation 1, the configured run therefore consumes 208334 × 32 = **6666688
new row occurrences**; 41666 × 32 = **1333312 reused conditions** enter student updates.
These are not unique-row counts. Count balancing rotates majority-count
subsets across epochs, so neither a sampler epoch nor a fixed update count guarantees
coverage of every row in the dataset.

## ACP launch: four workers, eight GPUs each

Pull the new code into the shared checkout first. ACP must supply the same
reachable `MASTER_ADDR` and `MASTER_PORT` on all four workers. Run the command
below once on **each worker**, not four copies on one machine. Use a new output
directory and rendezvous ID; do not resume old outer-iteration checkpoints.

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate

H3_CODE_ROOT="$PWD" \
H3_PYTHON="$VIRTUAL_ENV/bin/python" \
H3_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3 \
H3_REF2AV_CACHE=/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/latent_match124_bf16/metadata.jsonl \
H3_REF2AV_DMD_OUTPUT=/mnt/lm_data_afs/gushiqiao/outputs/h3_omni_pdmd4_official_lorafp32_32gpu_250k_run01 \
H3_RDZV_ID=h3_omni_pdmd4_official_lorafp32_32gpu_250k_run01 \
KERNELS_CACHE=/mnt/lm_data_afs/gushiqiao/cache/kernels \
bash lightx2v_train/scripts/run_minimax_h3_ref2av_omni_imageonly_pdmd4_official_fsdp32_32gpu_acp.sh
```

Append `--dry-run` to perform read-only cache-receipt/checksum and configuration
checks and print the torchrun command. It does not import torch, open tensor
payloads, initialize CUDA or launch training. A successful preflight is not a
GPU memory or training-correctness guarantee.

Set `H3_PDMD_OFFICIAL_CONFIG` to a separate YAML to customize this recipe.
Stale `H3_CONFIG_PATH`, `H3_PDMD`, `H3_PDMD_COUNT_RANDOM_CONFIG` and
`H3_REF2AV_EXPECTED_ROWS` from older runs cannot select another recipe here.

The existing condition cache is sufficient; this PDMD path does not require
DMAD real/teacher target latents or an offline teacher rollout dataset.

## Resume and verification

New checkpoints include per-rank official-PDMD state, the consumed dataloader
microbatch cursor and, where needed, the cached preceding-critic trajectory.
Recipe and world-size changes are checked before resuming. Old DMD/PDMD
checkpoints or incomplete official state must not be treated as compatible.

CPU configuration/preflight regression tests cover the retained data/precision
settings, 4-step recipe, single-role update accounting, old-recipe isolation,
cache receipts and read-only launch behavior. They do not constitute a
32-GPU smoke test; actual H3/FSDP2 memory and throughput need validation on
the allocated ACP workers.

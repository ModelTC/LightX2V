# MiniMax-H3 Ref2AV: 8-step DMAD on 32 ACP GPUs

This is a new `training.method: dmad` implementation for the current
capability-based training tree. It is separate from DMD, PDMD and residual
HEAD. The student generates joint video **and audio**; “image-only” describes
its reference inputs, not its generated output or its real training targets.

The algorithm follows [DMAD: Distribution Matching as Adversarial Distillation
for Fast Visual Generation](https://arxiv.org/abs/2610.02188), Zhengming Yu et al.,
and the Apache-2.0 [upstream H3 training code](https://github.com/Yzmblog/DMAD/tree/8067c05f74a8cfc818d6e21c2b49405b49ba9cbc/train/h3)
at commit `8067c05f74a8cfc818d6e21c2b49405b49ba9cbc`. This port adapts that
T2AV work to ordered Ref2AV conditions and an eight-step student. Upstream's
H3 recipe uses four steps; this eight-step variant is an adaptation, not a
claimed reproduction of that recipe or a published Ref2AV result.
Source credit, modification notes and the upstream notice are preserved in
[THIRD_PARTY_DMAD.md](THIRD_PARTY_DMAD.md).

## Data preparation is required

An existing Omni `condition_path` cache is **not sufficient**. Every exact
condition must be paired with both:

1. Real video **and audio** VAE latents for its target clip, using H3's
   normalized latent convention and matching output geometry.
2. An offline teacher-generated video **and audio** latent sample produced
   with that same prompt, ordered references and output geometry.

Reference image latents are conditioning, not the real target video. A T2AV
teacher sample with the same caption is not a valid substitute for a Ref2AV
teacher sample. Do not fabricate missing audio or copy the real latent into
the teacher field. Upstream's public T2AV latent cache is not a drop-in
replacement for the Omni reference-conditioned targets.

The two offline tools below prepare targets; the subsequent manifest join
only checks and joins the resulting artifacts. Finish both preprocessing jobs
before launching training. No online teacher score model is constructed
during DMAD training.
This does not omit the teacher branch: every critic update still consumes
its identity-matched offline teacher target. Missing teacher targets are an
error, not a reason to silently skip that term.

### Pairing contract

Create three manifests: conditions, real targets and teacher targets. All
three require `condition_path`, a non-empty `cache_fingerprint` identifying
the exact condition, and `target_height`, `target_width`, `target_num_frames`
(`num_frames` is an accepted alias). Target geometry and fingerprints must
match exactly. If the condition row has `source_id`, `source_row_uid` or
`sample_id`, both target manifests must retain the same values.

The real manifest adds `real_latent_path` and `normalized: true`; the teacher
manifest adds `teacher_latent_path` and `normalized: true`. `latent_path` is
an accepted alias within either target manifest. Paths are resolved relative
to the manifest that contains them. Each condition has one unique pair;
missing, extra or duplicate conditions are rejected.

Each target `.pt` payload repeats `normalized: true`, `cache_fingerprint`,
target geometry and `video`/`audio` tensors (`video_latents`/`audio_latents`
aliases are accepted). The dataset accepts packed video `[V, 96]` and audio
`[A, 32]` rows, or canonical video `[24, F, H/16, W/16]` and stereo audio
`[2, T, 32]` (native VAE `[2, 32, T]` is also accepted), with an optional
singleton batch dimension. This contract is
stricter than a bare upstream `.pt`: tensor normalization, identity, finite
values and shape are checked when each sample is loaded.

Preserve the condition metadata needed by the reference cost sampler:
`reference_image_count` (or `ref_image_count`), `reference_video_count`,
`reference_audio_count`, `packed_sequence_tokens_124` and output geometry.
The new joined manifest retains this metadata; it does not overwrite or
modify the existing condition cache.

### Encode real targets and generate offline teacher targets

The raw real-source manifest needs a unique `source_id` and actual
`target_video_path` (or `video_path`), plus optional separate `audio_path`.
For example: `{"source_id": "clip_001", "video_path": "/data/clip_001.mp4"}`.
The utility matches this ID to the condition row's `source_id` and attaches
the exact condition fingerprint and geometry to the produced targets. Use
`--identity-field sample_id` if the raw source manifest uses that column
instead; the condition cache still supplies `source_id`. Duplicate IDs or
multiple condition variants for one source are rejected rather than guessed.
Optional raw-source `condition_path`, `cache_fingerprint` and geometry fields
are validated if present; they are not required at this raw-media stage.

When `audio_path` is absent, the source video's audio is used. Missing or
short audio fails; no silence is substituted. Reference-only source data
without the corresponding real target clip/audio cannot supply this stage.

These are GPU preprocessing commands, separate from the CPU join/preflight
and from 32-GPU training. Choose new output directories and run them only on
the GPUs allocated for preprocessing. The real encoder uses one device; the
teacher generator below uses one eight-GPU node with FSDP. They need the same
H3-compatible environment as training.

```bash
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
export PYTHONPATH="$PWD/lightx2v_train${PYTHONPATH:+:$PYTHONPATH}"
export H3_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3
export H3_DMAD_CONDITIONS=/path/to/condition-cache/metadata.jsonl
export H3_DMAD_SOURCE_AV=/path/to/identity-matched-source-video-audio.jsonl
export H3_DMAD_REAL_DIR=/path/to/new-real-targets
export H3_DMAD_TEACHER_DIR=/path/to/new-ref2av-teacher-targets

python lightx2v_train/data_process/minimax_h3/prepare_minimax_h3_dmad_latents.py \
    --conditions "$H3_DMAD_CONDITIONS" --targets "$H3_DMAD_SOURCE_AV" \
    --model-path "$H3_MODEL_PATH" --output-dir "$H3_DMAD_REAL_DIR" \
    --device cuda:0

# Use the real stage's exact selected condition subset, including for a smoke test.
torchrun --standalone --nproc_per_node=8 \
    lightx2v_train/data_process/minimax_h3/generate_minimax_h3_dmad_teacher_latents.py \
    --conditions "$H3_DMAD_REAL_DIR/condition_metadata.jsonl" --model-path "$H3_MODEL_PATH" \
    --output-dir "$H3_DMAD_TEACHER_DIR" \
    --steps 31 --video-shift 12 --audio-shift 3
```

The real encoder supports `--dry-run` for metadata/path validation without
loading torch, media or VAEs. Actual encoding writes `metadata.jsonl` (real
targets) and `condition_metadata.jsonl` (the exact selected condition subset).
Its `--max-samples` selects a global prefix; `--rank`/`--world-size` can split
independent VAE workers into new `shard_NNNNN` output directories. The real
output directory must be fresh, including after an interrupted run.

For a preparation smoke test, add `--max-samples N` to the real stage and
pass its resulting `condition_metadata.jsonl` to both teacher generation and
the join. The teacher also supports `--max-samples`, but do not further
truncate only that stage: a partial teacher manifest cannot be joined to a
larger condition/real manifest. A small preparation test is not necessarily
large or balanced enough for the default 32-rank training sampler.
The teacher uses reference conditions, not the upstream text-only generator;
its final `metadata.jsonl` is assembled from the successful rank shards.

### Join the completed targets

From the new checkout on an ACP worker, after both jobs complete:

```bash
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V

# Replace these with COMPLETED, identity-matched preprocessing artifacts.
export H3_DMAD_CONDITIONS=/path/to/new-real-targets/condition_metadata.jsonl
export H3_DMAD_REAL=/path/to/new-real-targets/metadata.jsonl
export H3_DMAD_TEACHER=/path/to/new-ref2av-teacher-targets/metadata.jsonl
export H3_DMAD_CACHE=/path/to/new-dmad-cache/paired.jsonl

python lightx2v_train/lightx2v_train/data/minimax_h3_dmad_manifest.py \
    --conditions "$H3_DMAD_CONDITIONS" \
    --real "$H3_DMAD_REAL" --teacher "$H3_DMAD_TEACHER" \
    --output "$H3_DMAD_CACHE"
```

The join is CPU-only and refuses to overwrite an existing output. Repeat
`--conditions`, `--real` or `--teacher` to supply shards. The result has
`condition_path`, `real_latent_path`, `teacher_latent_path`, identity/geometry
metadata and `dmad_schema_version: 1`.

## Launch: four workers, eight GPUs per worker

Use the same new checkout, environment, model, joined manifest and output
directory on all four workers. ACP supplies a common `MASTER_ADDR` and
`MASTER_PORT`; do not launch four copies on one worker. Start with a new
DMAD output directory, not an old DMD/PDMD/HEAD checkpoint directory.

```bash
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
export H3_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3
export H3_DMAD_CACHE=/path/to/new-dmad-cache/paired.jsonl
export H3_DMAD_OUTPUT=/mnt/lm_data_afs/gushiqiao/codes/news/LightX2V/outputs/ref2av_dmad8_run01
export H3_RDZV_ID=h3_ref2av_dmad8_run01
export H3_DMAD_MAX_ITERS=800
# Optional when the worker image supplies an offline FA3 snapshot:
# export H3_KERNEL_SNAPSHOT=/path/to/flash-attn3/snapshot

bash lightx2v_train/scripts/run_minimax_h3_ref2av_dmad8_fsdp32_32gpu_acp.sh --dry-run
# After preflight succeeds, run this on EACH of the four ACP workers:
bash lightx2v_train/scripts/run_minimax_h3_ref2av_dmad8_fsdp32_32gpu_acp.sh
```

`--dry-run` resolves the actual config, validates manifest identity/geometry
and all three artifact paths per row, checks the default sampler's minimum
cell sizes, and prints the 4×8 launch command. It does not import torch,
deserialize tensor files, initialize CUDA or launch training. It is not a
replacement for dataset tensor validation or an actual multi-GPU smoke test.
`H3_PYTHON` can select an explicit interpreter. Preflight also rejects
manifest geometry incompatible with `fixed_num_frames`/`allowed_resolutions`
and DMAD settings that would select a different schedule, update order or EMA.

The dedicated config is
`configs/train/dmd/minimax_h3_ref2av_dmad8_fsdp32.yaml`. Set `H3_DMAD_CONFIG`
to a separate YAML to customize the recipe. Stale `H3_CONFIG_PATH` and
`H3_PDMD` values cannot select an old algorithm in this launcher.

## Recipe and deliberate adaptations

| Setting | This Ref2AV recipe |
| --- | --- |
| Parallelism | FSDP2 over 32 ranks, SP off, batch 1/rank, accumulation 1 |
| Student / critic | Independent LoRAs, rank 128, alpha 128 |
| Optimizers | Both AdamW, LR 4e-5, betas 0/0.99, weight decay 0.01 |
| Updates | One student update, then one critic update; no gradient clipping |
| Student rollout | At most 8 evaluations, random decreasing noise levels, fresh re-noise between evaluations; gradient only at the selected exit step |
| Critic noise | Continuous base U[0.02, 0.98], then video/audio shifts 12/2 |
| Targets | 124 frames, 768×1344 or 1344×768, joint stereo audio/video |
| Critic | Two adversarial heads on block 49; real-vs-generated and teacher-vs-generated |
| Teacher routing | Gap temperature 2; globally synchronized band statistics |
| Student EMA | Power-function EMA, gammas 6.94 and 16.97; separate from standard student EMA |
| Precision | Student and critic FP32 master weights, FSDP BF16 compute, FP32 reduction and generated latents |
| Duration / saving | Default 800 outer iterations; checkpoint every 100, retain 8 |

The upstream LoRA-critic paper checkpoint uses iteration 800 of an 8-GPU
T2AV run. At batch 1/rank, this 32-GPU adaptation sees four times as many
samples per optimizer update; 800 iterations is a convenient starting
duration, not a claim of equivalent data exposure or convergence. The
upstream YAML allows 8000 iterations. `H3_DMAD_MAX_ITERS` changes this run's
duration without editing the YAML.

FP32 master parameters plus BF16 FSDP compute are an intentional precision
portability change from upstream's BF16 transformer storage with no FSDP
parameter casting. Global gap synchronization is also intentional: upstream
H3 maintains per-rank gap statistics, whereas this recipe aggregates across
32 ranks. These changes, eight-step generation, Ref2AV conditioning and the different data sampler
must be disclosed in comparisons to the paper.

The sampler retains the existing Omni image-only policy: one observed image
count from 1–6 per 32-row global microbatch, 16 landscape plus 16 portrait,
balanced counts over the epoch with rotating omission of excess rows. Every
observed count/orientation cell needs at least 16 paired rows. This is not a
full-data pass and is not paper-mandated. Customize `reference_cost_sampler`
in a separate config when another data distribution is intended.

Upstream's offline teacher generator defaults to 31 evaluations and shifts
12/3; its student and critic training use 12/2. Record the actual Ref2AV
teacher-generation settings with your target cache. Teacher sampling settings
and the student training noise schedule are distinct choices.

## Checkpoints and inference

DMAD owns its critic heads, routing statistics and power-function EMA banks.
Use DMAD checkpoints only with a compatible DMAD recipe and paired dataset;
do not resume an old DMD/PDMD/HEAD run by pointing this launcher at its output.
Automatic resume is enabled within the new DMAD output directory.
Completed checkpoints export the live student LoRA and two EMA student
directories, `ema1_student/` (gamma 6.94) and `ema2_student/` (gamma 16.97).

The config disables training-time visual inference by default. Exported live
and EMA student LoRAs require a matching **eight-evaluation stochastic re-noise
sampler** with video/audio shifts 12/2. The old deterministic Euler DMD
inference path is not the DMAD sampling procedure; a config step-count change
alone is insufficient. No visual-quality or 32-GPU throughput result is
implied by CPU unit tests or preflight success.

Validation status for this implementation: CPU tests and launch preflight
have been run; real-media GPU preprocessing, distributed teacher generation
and the full 32-GPU training job have **not** been run as part of this change.

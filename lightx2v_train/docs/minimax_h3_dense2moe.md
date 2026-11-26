# H3 T2AV Dense2MoE

This is an FFN-only adaptation of [Dense2MoE](https://arxiv.org/abs/2510.09094).
It is not timestep distillation, PDMD, or an implementation of MoB depth pruning.
The teacher is the original dense H3 FL2AV checkpoint. Native text, stereo audio
and video tokens are packed together. Reference-image conditioning is absent.

## Data and objective

Use the 8,560 T2AV video/prompt pairs from `DiffSynth-Studio/MiniMax-H3-Self-Generated-Dataset`.
Metadata reports 6,148 videos at 1344×768, 1,239 at 768×1344, and
1,173 at other aspect ratios around the same pixel area. All have 124 frames
and were generated with 30 inference steps. Every training record supplies its
own `width`, `height` and `num_frames` to media decoding and native packing. No
resize, crop, fixed landscape canvas, or aspect-ratio reweighting is applied.
Four cache workers read the existing MP4s and encode them once before training;
there is no teacher rollout.

Each stage shuffles all records without replacement using a private seeded CPU
RNG. FSDP ranks consume disjoint positions in that global order; each complete
epoch exactly preserves the dataset's resolution frequencies. A partial epoch
is a uniformly sampled subset, so its counts need not equal the full-dataset
ratios exactly. The order is reconstructed from stage, seed and optimizer step
on resume. Each rank processes one sample; ranks may have different sequence
lengths and there is no cross-rank sequence padding. The `batch` log event lists
record indices, dimensions and frame counts for every rank.
Text is cached by the original Qwen3-VL conditioner. Each cache worker retains
the frozen BF16 text model on its GPU; training does not load it.
Video and audio use the existing native H3 FP32/SDPA VAE encoders with posterior mode
and checkpoint latent normalization. Video uses the native default spatial tiling
(256 pixels, 64-pixel minimum overlap). Unused VAE decoders stay on CPU; encoders
are frozen. VAE attention is scoped independently of the DiT FA3 backend. FP32 video/audio latents and BF16 prompt embeddings are stored without extra
quantization. Training reads these tensors and samples fresh noise each iteration.

The MP4 is decoded at its native 24 fps; audio is resampled to 32 kHz stereo and
aligned using stream timestamps. The 40 Hz audio grid rounds a 124-frame clip to
207 frames per channel. Only the rounding tail is zero-padded; a missing audio
stream or an uncovered video interval is an error. The resulting tensors are
video `[1, 37296, 96]` at 1344×768 and audio `[1, 414, 32]`, with left-channel
rows followed by right-channel rows, matching native H3 packing.

Draw independent video/audio noise and one shared `u ~ Uniform(0,1)`. Apply
`sigma = shift*u/(1+(shift-1)*u)` separately with video shift 12 and audio shift 3.
Native H3 receives modality-specific `t = 1-sigma` and predicts clean-ward velocity.
Teacher and student see identical noisy inputs and time conditioning. Video and
audio MSEs are computed separately and summed with weights 1 and 1, so the much
smaller audio token count does not dilute its output loss. Block features cover
all packed modalities. This is an H3 AV adaptation of the FFN KD objective.

Stages:

1. `calibrate`: trainable dense FFNs receive an ordinary flow-matching loss
   against `clean-noise`, without optimizer updates. Sum `abs(weight*gradient)`
   over each SwiGLU up/gate/down channel triplet. Rank by cumulative importance.
2. `shared`: keep the most important 25% of each selected FFN's channels. Train
   these shared FFNs against dense output and block-feature targets; freeze
   attention, AdaLN, embeddings and output heads.
3. `moe`: freeze the enhanced shared FFNs. Partition the remaining channels into
   12 disjoint experts; train those experts and a softmax Top-2 router. Retain
   output/feature KD and add load balancing with coefficient 0.01.

For H3 width 14336: shared width 3584, each expert width 896, active width 5376
(37.5% of dense FFN width). Default conversion covers blocks 3–49. This changes
SwiGLU channel capacity; it does not split the input embedding into MoE heads.
The implementation uses PyTorch grouped GEMM, including autograd, on H200.
Each expert receives an extra zero sentinel row so empty expert groups have a
nonzero reduction dimension in backward. The sentinel is excluded from outputs
and contributes zero parameter gradient; token routing is unchanged.

Feature balancing follows equations 7–10 with detached scale coefficients.
Teacher block targets are copied to pinned CPU memory and transferred back one
layer at a time for the feature loss and its backward. This saves about 19 GiB
per rank at 1344×768/124 frames without changing the target values. Student
activation checkpointing remains enabled. AdamW uses `foreach=False` to avoid
an extra parameter-sized tensor list during updates.
Weighted router-loss gradients enter at each FFN's output, allowing FSDP to
release that layer without a separate cross-layer auxiliary backward branch.
The detached router losses are still included in reported total loss. This
trainer uses unscaled BF16 backward without GradScaler or gradient accumulation.
There is no PAD in the native single-sample packing. Frozen non-FFN weights and
FFN-only updates are a deliberate scope choice; quality must be validated after
training. Default 32/1000/2000 stage lengths are experiment defaults, not a claim
of reproducing the paper's compute budget.

## Precision

Matches the existing selfgen PDMD fake-score configuration: all student master
weights are FP32, FSDP2 computes with BF16 parameter views and reduces gradients
in FP32. AdamW's trainable parameters and moment buffers remain FP32. There is
no outer autocast context. This is mixed-precision training, not FP32 arithmetic
for every operator and not an attention-only BF16 cast. The teacher and resident
inference benchmark use BF16. Native operations may internally accumulate in
FP32 (for example normalization and losses).

## Download teacher videos and metadata

```bash
cd /data/nvme6/gushiqiao/codes/latest/LightX2V/lightx2v_train
export H3_PYTHON=/data/nvme6/gushiqiao/models/MiniMax-H3/local_diffusers/.venv/bin/python
"$H3_PYTHON" scripts/download_minimax_h3_dense2moe_prompts.py \
  --output-dir /data/nvme6/gushiqiao/datasets/h3_dense2moe_prompts --videos
```

For the default seed-42, four-rank, two-iteration smoke run, replace `--videos`
with `--smoke-videos`. It downloads the required records from each stage while
retaining the complete metadata and sampling order. Full training requires all videos.

## Four-GPU FSDP2

Use `gsq_x2v_debug1` with the existing environment
`/data/nvme6/gushiqiao/models/MiniMax-H3/local_diffusers/.venv/bin/python`
(PyTorch 2.11, native H3 Diffusers, Transformers and kernels). Teacher and student
both use `_flash_3_hub`, loading the locally cached FA3 revision
`43f0bd269777115d94ff826e0d113ce9c1c9087b` without network access. Its forward and
Q/K/V backward have been checked against SDPA. The older container environment's
`_flash_3` dispatch lacks autograd registration; it is a different integration.

The model path contains converted Diffusers DiT weights. Its text-encoder
symlinks are broken on this machine, so `H3_TEXT_MODEL_PATH` points to the intact
original H3 conditioner weights bundled with Agnes. No Agnes Python code or
MoE weights are used; `local_diffusers/run_h3.py` is not imported either.

```bash
cd /data/nvme6/gushiqiao/codes/latest/LightX2V
export CUDA_VISIBLE_DEVICES=0,1,2,3
export H3_PYTHON=/data/nvme6/gushiqiao/models/MiniMax-H3/local_diffusers/.venv/bin/python
export H3_MODEL_PATH=/data/nvme6/gushiqiao/models/MiniMax-H3
export H3_TEXT_MODEL_PATH=/data/nvme6/gushiqiao/models/Agnes-Video-H3-MoE/FL2VA
export H3_PROMPTS=/data/nvme6/gushiqiao/datasets/h3_dense2moe_prompts/metadata.jsonl
export H3_MOE_OUTPUT=/data/nvme6/gushiqiao/outputs/h3_dense2moe_t2av_cached
bash lightx2v_train/scripts/cache_then_train_minimax_h3_dense2moe_fsdp4.sh
```

After the full cache completes, `run_minimax_h3_dense2moe_fsdp4.sh --smoke`
can run two iterations per stage in a separate output directory. It retains each record's original resolution and frame count. Smoke outputs use the `_smoke` suffix and
must not be described as a converged model or a quality-matched speedup.
The training loop uses the completed cache, including in smoke mode.

The launcher runs separate processes for calibration/shared/MoE. Trainable
weights, optimizer state and progress are saved with distributed checkpointing.
Completed shared/MoE weights are exported as `shared.pt` and `moe.pt` alongside
`partition.pt`. The dense base is still required. `metrics.jsonl` contains the
loss terms, iteration times and peak CUDA memory. `timings_by_rank` reports
cache loading, teacher forward, student
forward/loss, backward and optimizer time in seconds. Each timing synchronizes
CUDA; iteration time excludes startup and checkpoint/export I/O. FSDP timings
include collective waits, so compare rank-local totals rather than summing
the maximum of each stage across different ranks. All four ranks must use the
same model, dataset and recipe. With `save_limit: 1`, `checkpoints/latest.json` points to the only retained
optimizer snapshot across shared/MoE stages. A new checkpoint is fully saved
and published before the old one is removed. `partition.pt`, `shared.pt` and
`moe.pt` are required initialization/export artifacts, not additional optimizer
snapshots. Shared/MoE stages resume optimizer checkpoints;
an interrupted calibration restarts scoring. Run a stage directly:

```bash
cd /data/nvme6/gushiqiao/codes/latest/LightX2V/lightx2v_train
export PYTHONPATH=$PWD
"$H3_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m lightx2v_train.trainers.dense2moe.run \
  --config configs/train/dense2moe/minimax_h3_t2av_fsdp4.yaml --stage moe
```

## Dense versus MoE latency

```bash
CUDA_VISIBLE_DEVICES=4 "$H3_PYTHON" -m lightx2v_train.trainers.dense2moe.run \
  --config configs/train/dense2moe/minimax_h3_t2av_fsdp4.yaml --stage benchmark
```

Add `--smoke` to benchmark the smoke checkpoint. Benchmark geometry defaults
to the selected metadata record. `--benchmark-size HEIGHT WIDTH` and
`--benchmark-frames N` override only the benchmark, never training geometry. Both variants use the same
online-encoded prompt, noisy input, BF16 FFNs, attention backend, geometry,
warmup and repeats. This measures resident single-GPU DiT forwards, excluding
model loading, Qwen encoding, VAE encoding/decoding and FSDP communication. It reports
median latency, peak memory, speed ratio and output relative RMSE. RMSE at one
noise level is a numerical diagnostic, not a video-quality evaluation.

## Historical T2V-only structural latency (2026-11-25)

NVIDIA H200, PyTorch 2.11.0+cu128, BF16 parameter computation with FP32 RoPE
buffers, cached FA3, batch 1, 1344×768, 124 frames. There are 37,296 video tokens
and 236 text tokens. After three warmups, the median of ten DiT forwards was:

| Model | Seconds per forward | Peak allocated GPU memory |
| --- | ---: | ---: |
| Dense H3 | 5.7808 | 67.817 GiB |
| Dense2MoE H3 | 5.3293 | 67.800 GiB |

Speed ratio: **1.0847×**, latency reduction **7.81%**. The checkpoint has only
two calibration samples per rank and two optimizer updates in each distillation
stage at the earlier 160×288/107-frame smoke geometry. It is not converged. Its output relative RMSE against
the teacher at the benchmark's single sigma is 0.4073. These are structural
latency measurements, not evidence of quality-matched acceleration. Both
variants retain all experts in GPU memory, so MoE does not reduce resident
parameter memory in this experiment.

Raw timing report:
`/data/nvme6/gushiqiao/outputs/h3_dense2moe_t2v_768p_fp32_smoke/benchmark.json`.

## Earlier fixed-geometry training validation

All three stages completed two-iteration real-weight smoke runs on four H200s.
With the same FP32-master/BF16-compute policy, a full 1344×768/124-frame MoE
optimizer update passed on four H200s after moving teacher targets to CPU and
placing the router auxiliary gradient locally inside each FFN. Peak allocated
memory on rank 0 was **96.330 GiB**. The update took **95.32 seconds**, including
online text encoding and a **four-step** teacher rollout for this engineering
check. That earlier recipe used **30 teacher steps** by default. These historical
timings do not describe the current online-encoded T2AV recipe.

This full-size check uses the smoke partition/shared initialization and does
not establish convergence or quality. The production-length 32/1000/2000 stages
have not been run. Eight numerical tests pass, including FA3 backward, sparse
dispatch gradients, empty experts, CPU feature targets, and equivalence of the
local router gradient to the explicit weighted auxiliary loss. A 50-layer,
three-rank nested-FSDP/checkpoint test also passes.

Full-size training log:
`/data/nvme6/gushiqiao/outputs/h3_dense2moe_t2v_768p_fp32_fullcheck/local_aux.log`.

## Historical T2V native-resolution validation

Use the new output directory above. Earlier fixed-geometry checkpoints use a
different data order and recipe and must not be resumed in place as the same
experiment. Calibration, shared distillation and MoE distillation all use the
native-record sampler. The dataset's geometry fields are required; global
`height`, `width` and `num_frames` training overrides have been removed.

Validation covers all 8,560 records and 28 geometry buckets: one complete
four-rank sampling epoch exactly preserves the source frequencies, and every
native spatial grid passes packing checks. All 14 numerical and data tests pass.

A real-weight, four-H200 check completed two MoE optimizer updates and checkpoint
export using native 124-frame samples, including 1344×768, 768×1344 and 1184×864.
Rank 0 peak allocated memory reached **111.683 GiB** across these updates. This
engineering check uses four teacher steps and the earlier smoke partition/shared
initialization; it does not establish convergence or measure the speed of the
earlier 30-step teacher recipe.

Native-geometry validation log:
`/data/nvme6/gushiqiao/outputs/h3_dense2moe_native_geometry_check/launch.log`.

## Automated four-GPU cache → train run

The active run uses GPUs 0,1,2,3. Each cache worker handles every fourth metadata
record, downloading missing original videos as needed. Completed samples are
atomically saved and skipped on restart. A cache specification binds the metadata,
encoder files, geometry and precision; incompatible caches are rejected. The
full 8,560-record completion marker is required before training starts. Missing
or invalid records stop the pipeline rather than changing the dataset distribution.

- Script: `scripts/cache_then_train_minimax_h3_dense2moe_fsdp4.sh`
- Config: `configs/train/dense2moe/minimax_h3_t2av_fsdp4.yaml`
- Cache: `/data/nvme6/gushiqiao/datasets/h3_dense2moe_t2av_cache`
- Run: `/data/nvme6/gushiqiao/outputs/h3_dense2moe_t2av_cached`
- Logs: `logs/pipeline.log`, `logs/cache.log`, `logs/train_calibrate.log`,
  `logs/train_shared.log`, `logs/train_moe.log`
- Current stage: `logs/state.txt`; process: `logs/pipeline.pid`
- Per-GPU cache progress/timings: `<cache>/logs/rank_0.jsonl` through `rank_3.jsonl`

The script runs 32 calibration batches, 1,000 shared updates and 2,000 MoE
updates after caching. `save_every: 100`, `save_limit: 1`. Student masters and
optimizer remain FP32, FSDP compute is BF16, and DiT attention uses FA3.
Do not run the standalone trainer concurrently with the automated pipeline.

## Evaluate an intermediate shared/MoE checkpoint

`python -m lightx2v_train.trainers.dense2moe.infer` accepts `--checkpoint` (a DCP
checkpoint directory), `--partition`, `--prompt-file` and `--output`. Add `--dense`
to evaluate the unchanged teacher with the same request. Checkpoint loading reads
only saved model tensors, not optimizer state. Inference uses the training module,
native H3 packing, the native per-modality schedulers and H3 AV decoders.
Default geometry is 1344×768, 124 frames; seed is 42. The native scheduler's
`--steps 30` specifies 30 sigma points including zero, hence 29 model evaluations.
The shared stage retains only the selected shared channels; it is an intermediate
student, not the final routed MoE. Keep an evaluation snapshot outside the rolling
training checkpoint directory when `save_limit=1` is active.

Use `--untrained` to evaluate the partitioned initialization without loading the
trained FFN tensors. This isolates training progress from the initial architecture
change. `--dense` is a separate reference with the full original FFNs. Keep prompt,
seed, geometry and native schedules identical across these comparisons.

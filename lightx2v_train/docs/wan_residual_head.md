# Wan residual-head DMD

An optional small head learns an x0 correction from detached fake tokens and
sigma. It does not consume the clean student sample. The correction is
subtracted from the fake x0 prediction, before the existing DMD loss and its
normalization; it is not combined with projected DMD.

Each iteration holds the student fixed while it:

1. Updates the full fake five times on current student samples.
2. Recomputes features and `fake_x0 - student_x0` targets with the final fake
   snapshot. No fake parameters are copied or changed until the next iteration.
3. Warm-starts the head and takes five AdamW steps on those fit batches.
4. Generates an independent held-out batch with a separate, deterministic RNG
   stream, matching the student-query exit-step and score-sigma distribution.
5. Updates the student on a fresh batch using `fake_x0 - lambda * head - teacher_x0`.

The head is FP32, zero-output initialized, and has a 64-dimensional bottleneck.
It is replicated with DDP outside the FSDP student/fake modules. Only head
fitting updates it. Fake features and residual targets are detached.

Five noise bins maintain globally pooled held-out error improvements, with EMA
decay 0.9. A bin needs at least three independently observed iteration rounds
before its gate can ramp upward by 0.25. Insufficient or unfavorable EMA
evidence sets its gate to zero. These diagnostics are empirical checks, not a
claim that a small head necessarily tracks the student more accurately.

## Three-way comparison

`configs/train/dmd/wan2_1_t2v_1_3b_head_comparison_fsdp2.yaml` uses four rollout
and preview steps, 81 frames at 480 x 832, student LoRA rank 128/alpha 8,
student/fake learning rates 5e-5/4e-7, and eight fixed previews every 100 iterations.
All three configurations use the same fake-first update order. The older
eight-step configuration is unchanged.

Student, fake and teacher transformer parameters are stored in FP32. Forward
computation retains BF16 autocast and the FSDP BF16 mixed-precision policy; this
is not an all-FP32 compute experiment. The VAE remains FP32 and T5 remains BF16.

Set the model, prompts and a separate output directory for every experiment:

```bash
export WAN_DMD_MODEL=/path/to/Wan2.1-T2V-1.3B
export WAN_DMD_PROMPTS=/path/to/prompts.txt
export WAN_DMD_OUTPUT=/path/to/new/experiment

# Baseline: projected=false, residual head=false.
CUDA_VISIBLE_DEVICES=0,1 WAN_DMD_PROJECTED=false WAN_DMD_RESIDUAL_HEAD=false \
  bash scripts/run_wan21_dmd_head_comparison_fsdp2.sh

# PDMD: select a different WAN_DMD_OUTPUT directory first.
CUDA_VISIBLE_DEVICES=2,3 WAN_DMD_PROJECTED=true WAN_DMD_RESIDUAL_HEAD=false \
  bash scripts/run_wan21_dmd_head_comparison_fsdp2.sh

# Residual head: select a third WAN_DMD_OUTPUT directory first.
CUDA_VISIBLE_DEVICES=5,6 WAN_DMD_PROJECTED=false WAN_DMD_RESIDUAL_HEAD=true \
  bash scripts/run_wan21_dmd_head_comparison_fsdp2.sh
```

Head weights, AdamW state and gate EMA/counts/lambdas are saved with trainer
state. Resuming requires the same head configuration. Do not resume a head run
from a non-head or eight-step checkpoint. The head is not needed for inference.

## Head diagnostics

Rank-zero training logs include JSON records pooled from all ranks:

- `[head][check]` reports per-bin fake, full-correction and pre-gate-policy
  risks, paired improvements, independent rank-query counts and standard
  errors. A single query has undefined uncertainty, not a zero standard error.
  The gate still uses the full `F - C` comparison; its new lambdas are decisions,
  not performance estimates on the same query that selected them.
- `[head][student]` reports the actual student-query sigma, bin and applied
  lambda; fresh-query fake and corrected risks; residual/head dot products and
  head energy; correction-to-DMD-direction norm ratios and direction angles.
  Directions are measured in x0 output space, not parameter-gradient space.
  Per-bin and cumulative counters distinguish positive lambda from a genuinely
  nonzero applied correction. These counters are checkpointed.

Candidate-lambda risks and optimal-lambda estimates are diagnostics only and
never select or override the gate. Fresh student-query risks measure the
already selected policy on samples not used for that gate decision. No extra
model forward or random sample is added for logging.

The head adds fitting and held-out computation and consumes additional fresh
prompts. This comparison is not matched-compute evidence against extra fake
updates. Fake still uses the existing velocity regression; head loss and
held-out error are measured in x0 units.

## 10k comparison: accumulated fitting and calibrated gates

The original 1000-iteration configuration above is unchanged. The separate
`configs/train/dmd/wan2_1_t2v_1_3b_head_comparison_10k_fsdp2.yaml` runs DMD,
projected DMD and the revised head for 10000 student updates each. This changes
two head mechanisms together; it is a combined-recipe comparison, not an
ablation attributing improvement to either mechanism alone.

Head fitting still uses the five distinct batches generated for the current
iteration's fake updates. Their features and `F-X` targets are recomputed after
the final fake update. With `fit_grad_accum_steps: 4` and `fit_steps: 5`, four
microbatches are averaged before one head optimizer step, followed by a correctly
normalized one-microbatch tail step. Two ranks with one sample per microbatch
therefore use global effective batches of eight and two, not twenty fresh
samples or four replays of one sample. Head learning rate and the number of
microbatch passes remain unchanged. No gradient is carried across student or
fake updates; the fake snapshot and student remain fixed during fitting,
calibration and validation. DDP synchronizes only at each accumulation group's
last backward. `[head][fit]` records effective batch sizes, replay counts,
optimizer steps and cross-iteration lag (zero for this recipe).

`gate_mode: calibrated` replaces the full-correction test with two independent
query streams:

1. A fresh calibration rollout updates per-bin EMA sufficient statistics
   `a = mean((F-X)*C)` and `b = mean(C*C)`. It proposes
   `lambda = clip(a/b, 0, 1)` (zero if the energy is zero).
2. A different fresh validation rollout updates separate validation statistics.
   It evaluates only the already frozen calibration scale through
   `2*lambda*a_validation - lambda**2*b_validation`; it never optimizes lambda
   using validation data. Both streams need `min_checks` observed rounds.
3. An observed validation bin accepts exactly that scale only when its EMA
   improvement exceeds the configured relative threshold. Rejected bins use
   zero. Unobserved bins retain their previously validated scale, not a new
   unvalidated candidate. The old `gate_ramp` is not used in calibrated mode.
4. The student uses another fresh query. Its risk diagnostics evaluate the
   actual accepted policy, independently of the current calibration/validation
   queries. These x0 prediction errors are not video-quality scores or
   parameter-gradient norms.

The calibration and validation RNG streams are distinct and restore the main
training RNG; they consume different prompt batches. The head run consequently
consumes extra prompts and computation compared with DMD/PDMD. The validation
stream participates in gate decisions, so the subsequent student fresh-query
statistics remain the primary prospective policy check. Historical EMA evidence
is empirical and can lag a changing head/fake distribution.

`[head][calibration]` and `[head][validation]` log the selected snapshot, paired
risks and per-bin statistics separately. Within-query candidate-grid and oracle
lambda values remain counterfactual diagnostics, never validation-based choices.
Checkpoint state includes both gate streams and head fitting counters. The new
recipe must start in a fresh output directory; legacy full-mode checkpoints can
only resume with the original accumulation and gate settings.

Launch all three groups from an environment with the training dependencies and
six idle GPUs:

```bash
bash lightx2v_train/scripts/run_wan21_dmd_pdmd_head_10k_fsdp2.sh --preflight
bash lightx2v_train/scripts/run_wan21_dmd_pdmd_head_10k_fsdp2.sh
```

The launcher defaults to GPU pairs `0,1`, `2,3` and `5,6`, checks availability
without terminating existing jobs, and creates a fresh timestamped output root.
Override `WAN_DMD_PYTHON`, `WAN_DMD_MODEL`, `WAN_DMD_PROMPTS`, `WAN_DMD_RUN_ROOT`,
`DMD_GPUS`, `PDMD_GPUS` or `HEAD_GPUS` as needed. `--dry-run` only prints the
plan. Each group gets a `launch.log`, `launcher.pid` and eventual `exit_code`;
`run_manifest.json` records resolved configs, source/prompt hashes and the GPU
snapshot. Previews remain every 100 iterations, and only the latest checkpoint
per group is retained within this new run.

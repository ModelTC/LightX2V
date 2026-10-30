# H3 dCM1500 -> official PDMD1500000, score-side DeCFG, eight steps, 32 GPUs

Each task has one launcher command and two independent stage configurations.
The first stage trains **1,500 dCM student updates** on the new self-generated
full-target AV cache. There is no sCM stage, JVP, joint DMD objective or fake
optimizer update during this warm-up. The final dCM **student EMA** initializes
the live PDMD student. PDMD starts fresh fake weights, optimizers and data
cursors, then performs **1,500,000 individual role optimizer updates**:
**1,250,000 fake + 250,000 student**, repeating five fake updates followed by one
student update. It does not maintain a new student EMA.
The dCM student EMA decay is **0.999**, restored to the original Turbo setting.

Both tasks now use **video shift 12 and audio shift 3**, with eight rollout
steps, FSDP across 32 ranks, no sequence parallelism, batch1/rank and accumulation1.
Student base and teacher are BF16, fake master parameters and learned LoRA are
FP32, compute is BF16 and gradient reduction is FP32. Student LoRA stays at
regular rank/alpha **128/128**, with block AdaLN and `norm_out.linear` rank/alpha
**16/16**. PDMD retains the released projection, query and complete Euler
rollout rules; AdamW student/fake LR is `2e-5`/`4e-6`, preserving the original
dCM-EMA -> PDMD continuation rates. Betas are `(0,.9)`, with no weight decay
and gradient norm clip1. These rates, the shifted eight-step schedule and full
fake are explicit H3 experiment adaptations of the official recipe.

## Configurations and data

| Task | dCM config | PDMD config |
| --- | --- | --- |
| FL2AV | `configs/train/turbo/minimax_h3_fl2av_dcm1500_decfg8_fsdp32.yaml` | `configs/train/dmd/minimax_h3_fl2av_pdmd1500000_from_cm_decfg8_fsdp32.yaml` |
| Ref2AV | `configs/train/turbo/minimax_h3_ref2av_dcm1500_decfg8_fsdp32.yaml` | `configs/train/dmd/minimax_h3_ref2av_pdmd1500000_from_cm_decfg8_fsdp32.yaml` |

Under `H3_SELFGEN_CACHE`, dCM uses the complete `t2av/metadata.jsonl` for
FL2AV and `ref2av_imageonly/metadata.jsonl` for Ref2AV. These manifests must contain
`condition_path`, `video_latent_path`, `audio_latent_path`, source identity and
cache fingerprints. Every native 32-aligned resolution and 124-frame example
is retained. Ref keeps only samples with one or more image references and no
video/audio reference. Its target still contains paired video and audio latents.
The old root-level 768p-filtered metadata is not selected by this launcher.

On an actual Ref launch, the driver automatically builds this image-only view
from the verified complete `ref2av/metadata.jsonl` in the existing cache. This
is a standard-library CPU manifest operation: no model or tensor decoding,
no latent re-encoding, and no restriction to the two 768p canvases. Four workers
can safely share its publication. The source full-Ref manifests remain available.

PDMD uses the matching task's `condition_metadata.jsonl` as its new source,
then exhausts the old source before returning to a new pass:

- FL2AV: full new T2AV -> full old T2AV/I2AV/FL2AV -> repeat.
- Ref2AV: new image-only Ref2AV -> old Omni image-only Ref2AV -> repeat.

The launcher checks completion receipts, manifest hashes, every new full AV
payload path, and matching identities between the dCM and PDMD new views.
It neither encodes videos during training nor requires old clean target
latents. PDMD reads only conditioning and generates its own training latents.
Boundary padding repeats rows from the same source to complete a DP/accumulation
window, preserving every source row without mixing new and old in one window.

## DeCFG and the stage boundary

DeCFG is enabled in both stage configurations on **fake and teacher score
models only**, at scale1 by default. The teacher is frozen after the preset is
merged; the full fake starts from that modified base and remains trainable.
The student never receives the preset: rollout, trainable student prediction
and inference all use the original H3 base plus the learned LoRA. During dCM,
the teacher uses DeCFG and the unused fake receives no optimizer updates.
The task-specific score presets are:

- FL2AV: `model_for_comfy_dit.safetensors`.
- Ref2AV: `model_ref2va_for_comfy_dit.safetensors`.

Download both once on the new machine if needed:

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate
H3_PYTHON="$VIRTUAL_ENV/bin/python" \
bash lightx2v_train/scripts/download_minimax_h3_training_adapter.sh
```

The launcher automatically passes
`$H3_STAGED_OUTPUT/cm/train/checkpoint-000001500/student_ema` to PDMD. Only the
learned student adapter transfers. The new phase applies DeCFG once to its
teacher/fake score models and loads the EMA into the unmodified student base.
Base/task, LoRA architecture, score-preset identity and scale must agree. A
stale `H3_STUDENT_INIT` from another run is rejected. Export metadata records
that the **student does not use DeCFG** and keeps the score-preset identity for
training audit. **Student inference loads only the learned LoRA, without DeCFG.**

This score-side use is an explicit distillation adaptation. DiffSynth's public
H3 examples document SFT with a training-only DeCFG preset; they do not provide
a dCM/PDMD DeCFG recipe. Keeping student rollout, training and deployment on
the same base avoids a generator mismatch. The older student-side-preset
checkpoints are a different experiment and must not be resumed here. The
launcher defaults now include `scoredecfg8`, `ema999_lr2e5_4e6`, and for Ref
`imgonly`, in output and rendezvous names. Use fresh output/run IDs when
restarting these restored-EMA, image-only recipes.
With score-side DeCFG and a LoRA student, a weights-only warm-start must use
a learned LoRA export. Loading a full student model as that warm-start is
rejected before tensor loading, so it cannot replace the original student base.
Resuming a complete checkpoint from the same stage still follows the strict
recipe and adapter-metadata checks.

## ACP commands

First synchronize the code and finish merging the self-generated cache. Run
the same task command once on **each of four workers**, with eight GPUs visible
on each. ACP must supply the same reachable `MASTER_ADDR` and `MASTER_PORT` on
all workers. The dCM and PDMD torchrun groups use different rendezvous IDs and
ports. PDMD defaults to `MASTER_PORT + 1`; set `H3_PDMD_MASTER_PORT` to another
reachable port if needed. It must differ from `MASTER_PORT`.

Common setup on all four workers:

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate
export H3_CODE_ROOT="$PWD"
export H3_PYTHON="$VIRTUAL_ENV/bin/python"
export H3_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3
export H3_SELFGEN_CACHE=/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_self_generated_latents_124_bf16
export H3_TRAINING_ADAPTER_DIR=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3-TrainingAdapter
export H3_TRAINING_ADAPTER_SCALE=1.0
export KERNELS_CACHE=/mnt/lm_data_afs/gushiqiao/cache/kernels
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset H3_CM_CACHE H3_NEW_CACHE H3_STUDENT_INIT H3_TRAINING_ADAPTER_PATH
unset H3_STAGED_CM_CONFIG H3_STAGED_PDMD_CONFIG
```

FL2AV, on each worker:

```bash
H3_STAGED_OUTPUT=/mnt/lm_data_afs/gushiqiao/outputs/h3_fl2av_dcm1500_pdmd1500000_restart_ema999_lr2e5_4e6_20261030 \
H3_RDZV_ID=h3_fl2av_dcm1500_pdmd1500000_restart_ema999_lr2e5_4e6_20261030 \
bash lightx2v_train/scripts/run_minimax_h3_dcm_pdmd8_decfg_fsdp32_32gpu_acp.sh fl2av
```

Ref2AV, on each worker:

```bash
H3_STAGED_OUTPUT=/mnt/lm_data_afs/gushiqiao/outputs/h3_ref2av_imgonly_dcm1500_pdmd1500000_restart_ema999_lr2e5_4e6_20261030 \
H3_RDZV_ID=h3_ref2av_imgonly_dcm1500_pdmd1500000_restart_ema999_lr2e5_4e6_20261030 \
bash lightx2v_train/scripts/run_minimax_h3_dcm_pdmd8_decfg_fsdp32_32gpu_acp.sh ref2av
```

Choose one task per 32-GPU allocation. Reusing a shell across tasks should not
reuse task-specific overrides such as `H3_CM_CACHE`, `H3_NEW_CACHE`,
`H3_TRAINING_ADAPTER_PATH` or `H3_STUDENT_INIT`; their launcher defaults choose
the matching task. Existing `H3_TURBO_CONFIG`, `H3_CURRICULUM_CONFIG` and old
task-specific config variables cannot select an older recipe in this entry point.

Append `--dry-run` to validate the complete stage plan and print both torchrun
commands without creating distributed groups or loading model tensors. This
is a CPU metadata check; it does not benchmark a 32-GPU training step.
Ref's read-only dry-run requires the image-only view to exist first; normal
training creates it automatically. To prepare the view for a dry-run explicitly:

```bash
"$H3_PYTHON" lightx2v_train/data_process/minimax_h3/prepare_modelscope_h3_dataset.py \
  imageonly-view --output-dir "$H3_SELFGEN_CACHE"
```

## Logs, checkpoints and resume

Logs are `$H3_STAGED_OUTPUT/cm/train/train.log` and
`$H3_STAGED_OUTPUT/pdmd/train/train.log`. dCM saves every500 student updates;
PDMD saves every6000 role updates, equivalent to1000 student updates. Both
retain the latest two checkpoints. Each worker launches PDMD only after its
dCM process succeeds; PDMD rendezvous waits for all four workers.
Each stage resumes its own complete checkpoint and data cursor. Re-running
the same command continues a partially completed experiment rather than
loading Turbo optimizer state into PDMD. Use a new `H3_STAGED_OUTPUT` when
changing the task, schedule, cache identities, LoRA architecture or DeCFG preset.

## Continue a completed dCM without rerunning it

The restored learning rates affect the PDMD configurations only; dCM remains at
student/fake `2e-5`/`4e-6`. The current EMA decay is 0.999 and Ref uses image-only
new data, so older 0.99/all-Ref checkpoints are separate recipes. A running stage driver has already bound both
configurations to its original plan. If its PDMD YAML changes during dCM,
it finishes dCM and then refuses the changed recipe before starting PDMD.
Its original `stages/plan.json` must remain available for audit.

After dCM finishes and the old driver exits, keep the same task, environment,
`H3_STAGED_OUTPUT` and `H3_RDZV_ID`, and append `--pdmd-only` to the task command
on all four workers. For FL2AV:

```bash
H3_STAGED_OUTPUT=/mnt/lm_data_afs/gushiqiao/outputs/h3_fl2av_dcm1500_pdmd1500000_fullnew_scoredecfg8_32gpu_20261030 \
H3_RDZV_ID=h3_fl2av_dcm1500_pdmd1500000_fullnew_scoredecfg8_32gpu_20261030 \
bash lightx2v_train/scripts/run_minimax_h3_dcm_pdmd8_decfg_fsdp32_32gpu_acp.sh fl2av --pdmd-only
```

For Ref2AV, use `ref2av` and its corresponding output/run ID. This mode checks
the complete dCM1500 checkpoint, all 32 rank receipts, exact CM recipe and
EMA export before starting PDMD. It records the new recipe under
`stages/pdmd_only/` and preserves the original two-stage plan. It launches
no dCM process and does not override PDMD checkpoint resume validation.
With `--pdmd-only --dry-run`, only configuration/cache checks and command
printing run; completed dCM checkpoint validation occurs on the actual launch.

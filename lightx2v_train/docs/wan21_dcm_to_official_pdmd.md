# Wan2.1 1.3B: dCM 100 -> 200 official PDMD optimizer updates

This short experiment starts fresh on GPUs 0 and 1, with FSDP2, FP32
parameters/compute/reductions and TF32 disabled. The same 64 training videos
and four held-out prompts from the earlier Wan Turbo experiment are reused.
VAE and T5 remain resident on each GPU. dCM encodes videos online; PDMD reads
only their prompts and generates fresh training latents, with no latent cache.

## Stages and counters

1. `configs/train/turbo/wan2_1_t2v_1_3b_dcm100_fp32_fsdp01.yaml` runs exactly
   **100 dCM student optimizer updates**. There is no sCM, JVP or fake update.
   At step 100 it saves the student and EMA, and four eight-step CM previews.
2. `configs/train/dmd/wan2_1_t2v_1_3b_official_pdmd200_from_dcm_fp32_fsdp01.yaml`
   loads **only** that `checkpoint-000000100/student_ema` adapter into the live
   PDMD student. Teacher/fake start from the original Wan base, with fresh
   optimizers, schedulers and data cursors. No new student EMA is maintained.
3. PDMD runs **200 individual optimizer updates: 33 student and 167 fake**, in repeated groups
   of **five fake updates followed by one student update**. Its
   `max_train_iters: 200` counts individual role optimizer updates. The run ends
   two fake updates into the next group; its checkpoint can resume that group.

Stage 2 creates fixed-prompt, fixed-seed Euler previews before training and
after role optimizer updates 100 and 200. These correspond to output directories
`iter-000000000`, `iter-000000100` and `iter-000000200`. At 100 iterations,
there have been 16 student and 84 fake updates; at 200, 33 student and 167 fake.
Each directory contains four
MP4s, `manifest.json` and `index.html`. The zero-step PDMD preview is the
dCM-100 EMA model sampled with Euler, allowing a consistent sampler comparison.

## Source alignment

The shared H3/Wan official PDMD path follows
[ZeamoxWang/pdmd at 6b6e10635495a2d9b989fee393235d406d3840c4](https://github.com/ZeamoxWang/pdmd/tree/6b6e10635495a2d9b989fee393235d406d3840c4),
specifically `src/pdmd/{engine,schedule,objectives}.py` and
`training/configs/pdmd_4nfe_544p.json`:

- Every fake update uses a complete no-grad Euler student rollout. The
  following student update reuses the immediately preceding fake trajectory,
  chooses one rollout state and makes one grad-enabled student prediction.
- The student score query follows the predicted interval trajectory, without
  fresh Gaussian re-noising. The fake query instead re-noises the full
  student endpoint at a uniform physical video sigma.
- Projection epsilon `1e-8`, normalizer floor `1e-5`, no `0.5` MSE multiplier,
  loss clamp `5`, and video loss weight `0.8` follow the released objective.
  Nonfinite normalized directions become zero.
- AdamW student/fake learning rates are `5e-5`/`1e-5`, betas `(0, 0.9)`,
  epsilon `1e-8`, weight decay zero, gradient norm clip one. There is no EMA.
- Wan's native velocity is `noise - clean`, matching the upstream sign.
  H3 retains its explicit opposite-sign conversion. Teacher CFG is applied
  only for scoring, not during student rollout or preview generation.

Explicit experiment adaptations are eight steps, training/inference video
shift 5, 81 frames at 480x832, teacher CFG 5, FP32 FSDP on two GPUs,
student LoRA rank/alpha 128/128 warmed from dCM EMA, and a full-parameter
fake matching our H3 experiment. Upstream uses H3, a four-step unshifted
training schedule, and two LoRA roles sharing a frozen base. This experiment
ports its algorithm; it does not reproduce the published training setup.

The older independently implemented projected-DMD loss path is retired.
Ordinary DMD is still available; projected DMD requires `official_pdmd=true`
in both training and model capability options. Older projected-only configs
fail explicitly instead of silently selecting the official method.

## Launch and logs

```bash
cd /data/nvme6/gushiqiao/codes/latest/LightX2V
CUDA_VISIBLE_DEVICES=0,1 WAN_CODE_ROOT="$PWD" WAN_PYTHON=/opt/conda/bin/python \
bash lightx2v_train/scripts/run_wan21_dcm100_official_pdmd200_fp32_fsdp01.sh
```

Use the Python executable from the actual training environment. On this
machine that environment is inside Docker `gsq_x2v_debug1`:

```bash
docker exec -e CUDA_VISIBLE_DEVICES=0,1 \
  -e WAN_CODE_ROOT=/data/nvme6/gushiqiao/codes/latest/LightX2V \
  -e WAN_PYTHON=/opt/conda/bin/python -e TZ=Asia/Shanghai \
  gsq_x2v_debug1 bash \
  /data/nvme6/gushiqiao/codes/latest/LightX2V/lightx2v_train/scripts/run_wan21_dcm100_official_pdmd200_fp32_fsdp01.sh
```

Append `--dry-run` to validate configuration, model/data paths and stage budgets
and print the commands without starting workers or initializing CUDA.
`WAN_DCM_PDMD_ROOT`, `WAN_DCM_PDMD_MODEL` and `WAN_DCM_PDMD_DATA_ROOT` override
the experiment, model and existing raw-data-selection roots respectively.

Default output root:
`/data/nvme6/gushiqiao/outputs/wan21_1_3b_dcm100_official_pdmd200_fp32_fsdp01_20261030`.
The two training logs are `dcm/train/train.log` and `pdmd/train/train.log`.
The launcher automatically starts PDMD after dCM training, checkpointing and
previews succeed. It records the exported adapter SHA256 in
`stage_transition.json`. A stage-1 failure stops the pipeline. Repeating the
same command uses each stage's own strict resume state; it never loads Turbo
optimizer state as an official PDMD resume.

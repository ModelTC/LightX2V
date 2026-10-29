# Wan2.1 T2V 1.3B local Turbo experiment

`wan2_1_t2v_1_3b_turbo8_fp32_gpu0_smoke.yaml` runs a short 300-update
experiment: 100 dCM, 50 sCM, then 150 sCM+DMD student updates. The joint
stage performs four additional full fake-model updates per student update.
This shortened schedule is an implementation/effect check, not the published
full training recipe. Changing its schedule is not an interchangeable resume
recipe for the existing strict Turbo checkpoint format.

The source MP4/prompt pairs are selected from
`/data/nvme6/gushiqiao/datasets/minimax_h3_teacher_api_5s`. There are 64
training pairs and four disjoint fixed validation prompts. Video is decoded
and sampled at 16 fps into 81 frames at 480×832, then encoded with the Wan
VAE on each training iteration. UMT5 conditions are encoded online. No latent
or embedding cache is written. The DiT roles, VAE and T5 remain on GPU0 in
FP32; TF32 and autocast are disabled. LoRA rank/alpha are 128/128. Teacher
guidance is 5, and the student's inference guidance is 1. Video shift is 5.

Run `lightx2v_train/scripts/run_wan21_turbo8_fp32_gpu0_smoke.sh` in the
prepared environment. It invokes the Python launcher with memory-efficient
SDPA selected explicitly so long FP32 DiT attention cannot silently allocate
dense score matrices. The launcher owns `CUDA_VISIBLE_DEVICES=0`.

Before training, iteration zero generates four original-model eight-step
baselines. Iterations 100, 200 and 300 generate the same four prompts and
seeds (42–45) using student EMA and the exact Turbo stochastic x0/re-noise
sampler. Each inference directory contains MP4 files, `manifest.json` with
settings and `index.html` for inspection. The optional 50-step UniPC,
CFG=5 original-model comparison is controlled by
`inference.teacher_baseline_at_zero` and defaults to false to avoid delaying
the short training run.

The two-GPU variant is
`wan2_1_t2v_1_3b_turbo8_fp32_fsdp01_smoke.yaml`, launched with
`scripts/run_wan21_turbo8_fp32_fsdp01_smoke.sh`. It shards all three DiT roles
over GPU0 and GPU1 with FSDP2, FP32 parameters/compute/reduction, while each
rank keeps its online T5 and VAE resident in FP32. Its separate output root
contains `fsdp01`; `WAN_TURBO_DATA_ROOT` points to the existing raw selection.
`inference.run_at_start=false` skips regenerating the completed initial
single-GPU comparisons. At every 100 updates all FSDP ranks execute all four
prompts and denoising forwards in the same order, and only rank zero writes
preview videos and metadata.

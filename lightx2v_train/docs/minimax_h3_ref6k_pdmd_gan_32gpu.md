# Ref CM1500 → PDMD8 + GAN on 6k paired AV samples

This recipe loads only the Ref dCM1500 student EMA. The full fake starts from the original Ref base, and the GAN head, optimizers and data cursors start fresh. All three roles use the original base without DeCFG. The source student's historical score-only DeCFG metadata is provenance, not a requirement to merge DeCFG into this run.

Configuration: `configs/train/dmd/minimax_h3_ref2av_pdmd8_gan_from_cm1500_ref6k_fsdp32.yaml`.

| Setting | Value |
|---|---|
| Training | Official projected DMD + existing H3 adaptation of DMD2 GAN |
| Rollout | 8 steps, video/audio shifts 12/3, no shift curriculum |
| Updates | fake first, 5 fake then 1 student |
| Student LoRA | 128/128; block AdaLN and `norm_out.linear` 16/16; FP32 masters |
| Student LR | 5e-5 |
| Full fake / GAN head LR | 4e-6 / 4e-6 |
| Optimizers | AdamW, betas 0/0.9, epsilon 1e-8, weight decay 0, clip 1 |
| GAN weights | Generator 0.005, discriminator 0.01; enabled from the first applicable update |
| Precision / parallel | BF16 compute, FP32 full-fake masters; FSDP32, SP1 |
| Batch | 1 per GPU, accumulation 1; global batch 32 |
| Initial budget | 3,000 role updates = 500 student + 2,500 fake updates |
| Checkpoints | Every 100 role updates, keep 2, auto-resume own output |

The initial budget is a short tuning run, not a validated optimum. The student LR matches the official PDMD value; keeping a full fake at 4e-6 and adding the H3 GAN head are adaptations, not the upstream PDMD recipe.

The launcher first uses one CPU process per node. Global rank zero imports both completed final-only teacher export indices, verifies the original teacher identity, then selects 1,000 eligible single-image self-generated samples deterministically with seed 42. The resulting dataset has 1,000 examples for each image count from 1 to 6. All clips use 124 frames at 24 FPS, either 768×1344 or 1344×768. Each GAN positive retains its own prompt and reference conditioning. No video encoding, latent copying or full trajectory loading is needed. The shared manifests are published once before starting 32 training workers.

Run this same command on each of the four ACP nodes (8 GPUs per node). ACP supplies `MASTER_ADDR` and `MASTER_PORT`. Activate the existing `lightx2v_h3` training environment on every node.

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate

H3_CODE_ROOT="$PWD" \
H3_PYTHON="$VIRTUAL_ENV/bin/python" \
H3_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3 \
H3_STUDENT_INIT=/mnt/lm_data_afs/gushiqiao/outputs/h3_ref2av_imgonly_dcm1500_pdmd1500000_restart_ema999_lr2e5_4e6_20261030/cm/train/checkpoint-000001500/student_ema \
H3_TEACHER_EXPORT=/mnt/devsft_afs_2/gushiqiao/outputs/data_new2/omni_ref_teacher29_base_5k \
H3_RAW_JSONL=/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/omni_r2v_image_only_100000_ir.jsonl \
H3_MEDIA_ROOT=/mnt/lm_data_afs/gushiqiao/datasets/omini-r2v \
H3_SELFGEN_METADATA=/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_self_generated_latents_124_bf16/ref2av_imageonly_768p/metadata.jsonl \
H3_REF6K_DATA=/mnt/devsft_afs_2/gushiqiao/outputs/data_new2/omni_selfgen_ref6k_base \
H3_REF6K_OUTPUT=/mnt/devsft_afs_2/gushiqiao/outputs/h3_ref2av_pdmd8_gan_cm1500_ref6k_lr5e5_shift12_3_fake5_32gpu_20261116/train \
H3_REF6K_MAX_ITERS=3000 \
H3_RDZV_ID=h3_ref2av_pdmd8_gan_cm1500_ref6k_20261116 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
KERNELS_CACHE=/mnt/lm_data_afs/gushiqiao/cache/kernels \
bash lightx2v_train/scripts/run_minimax_h3_ref6k_pdmd8_gan_fsdp32_32gpu_acp.sh
```

Append `--dry-run` to print both distributed commands without reading data or loading models. Normal startup validates checkpoint and dataset receipts before GPU training. It checks that self-gen has at least 1,000 eligible single-image samples; it never silently substitutes multi-image samples or repeats examples.

Logs: `$H3_REF6K_OUTPUT/logs/launch.<hostname>.<timestamp>.<pid>.log` and `$H3_REF6K_OUTPUT/train.log`. Checkpoints live directly under `$H3_REF6K_OUTPUT/checkpoint-*`. A fresh run may log “Starting from scratch” while resolving auto-resume; source student loading occurs afterward. This means fresh training state, not random student initialization.

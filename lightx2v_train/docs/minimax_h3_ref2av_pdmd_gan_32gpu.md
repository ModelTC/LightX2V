# Ref2AV PDMD + DMD2 GAN continuation

Use the self-generated **image-only Ref** cache, including each sample's own
caption, image references, clean target video latent and audio latent. The old
Ref dataset is not included. Native target dimensions and the cached reference
preprocessing are retained; no encoding or teacher video generation is needed.

Configuration: `configs/train/dmd/minimax_h3_ref2av_pdmd8_gan_from5700_shift12_3_fake5_fsdp32.yaml`.

- Official PDMD, eight steps, physical video/audio shifts **12/3**.
- **Five fake updates then one student update**, batch1/rank, accumulation1,
  four nodes × eight GPUs, FSDP32, no sequence parallelism.
- Student/fake LR **2e-5 / 4e-6**, constant schedule, Adam beta0/0.9,
  no weight decay, gradient norm clipped at1.
- Student LoRA **128/128**; `adaln_proj.linear` and `norm_out.linear` **16/16**.
- BF16 frozen student/teacher and forward compute; FP32 LoRA/fake master
  weights and gradient reduction. No student EMA.
- DMD2 non-saturating logistic GAN, shared full fake-score backbone,
  target-video features from block49, four ordered pooled token bins and an
  FP32 MLP with1024 hidden units. Generator weight **0.005**; each discriminator
  real/fake term weight **0.01**. References condition the backbone but are
  excluded from the classifier's target tokens. Real positives use their own
  paired condition and match the generated target geometry.
- Default **15000 role updates = 2500 student + 12500 fake**. Save every100
  role updates and retain the latest2 checkpoints.

The old checkpoint supplies **student LoRA and trained full fake weights**.
Optimizer states, counters, data cursors and the new GAN head start fresh.
Teacher uses the offline DeCFG score base. Student rollout and inference use
the original base without DeCFG. Fake already contains its trained score
weights; the preset is not merged again. Subsequent restarts restore the
complete new-stage state automatically.

After syncing the code, run the same command on all four ACP nodes:

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate

H3_CODE_ROOT="$PWD" \
H3_PYTHON="$VIRTUAL_ENV/bin/python" \
H3_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3 \
H3_SCORE_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3-DeCFG-score \
H3_SELFGEN_CACHE=/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_self_generated_latents_124_bf16 \
H3_REF_GAN_INIT_CHECKPOINT=/mnt/devsft_afs_2/gushiqiao/outputs/h3_ref2av_pdmd8_from_cm1500_decfg_shift12_3_fake10_save100_32gpu_20261103/train/checkpoint-000005700 \
H3_REF_GAN_OUTPUT=/mnt/devsft_afs_2/gushiqiao/outputs/h3_ref2av_pdmd8_gan_from5700_shift12_3_fake5_32gpu_20261112/train \
H3_REF_GAN_MAX_ITERS=15000 \
H3_RDZV_ID=h3_ref2av_pdmd8_gan_from5700_shift12_3_fake5_32gpu_20261112 \
KERNELS_CACHE=/mnt/lm_data_afs/gushiqiao/cache/kernels \
bash lightx2v_train/scripts/run_minimax_h3_ref2av_pdmd8_gan_fsdp32_32gpu_acp.sh
```

ACP supplies `MASTER_ADDR` and `MASTER_PORT`. `H3_SCORE_MODEL_PATH` must be the
same premerged Ref DeCFG root used by the source5700 run.
`H3_REF_GAN_REAL_CACHE` optionally overrides the default
`$H3_SELFGEN_CACHE/ref2av_imageonly/metadata.jsonl`.
If this default view is missing, the launcher builds its metadata from the
existing complete `ref2av/metadata.jsonl`, reusing all latent files.

Append `--dry-run` to print the preflight/training commands without launching
workers. The actual launch checks the source role weights, merged score-base
provenance, complete cache receipt and paired latent paths before torchrun.
It validates each serialized pair when consumed by training.

Training log: `$H3_REF_GAN_OUTPUT/train.log`.
Per-node launch/error logs: `$H3_REF_GAN_OUTPUT/logs/launch.*.log`.
Checkpoints: `$H3_REF_GAN_OUTPUT/checkpoint-*`.

GAN is applied to conditional target-video features; the discriminator is not
a separately supervised audio-quality model. This follows the existing H3
adaptation of DMD2's GAN, whose original image discriminator is replaced by
the H3 feature head.

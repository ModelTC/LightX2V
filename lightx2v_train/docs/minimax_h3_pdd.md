# H3 PDD training

These recipes implement **parallel decoding distillation (PDD)**. PDD trains
the student against teacher trajectory segments and adds multiple output
heads to predict the segment updates from one backbone evaluation. It has
no fake-score model and no fake/student update ratio. This is separate from
projected distribution matching distillation (PDMD); existing PDMD configs
and their default launcher behavior are retained.

The trajectory objective follows
[NVlabs/FastGen PDD](https://github.com/NVlabs/FastGen/blob/898e757836dc244192533fe6ea797906799dd745/fastgen/methods/trajectory_matching/pdd.py).
Our H3 recipes adapt the backbone through LoRA while training the new output
heads in full. That differs from upstream full-model training.

| Setting | H3-Image | H3 FL2AV / Ref2AV |
| --- | --- | --- |
| GPUs | 1 node × 8 | 4 nodes × 8 |
| Target inference steps | 2 | 8 |
| Teacher grid intervals | 32 | 32 |
| Video / image sigma shift | 12 | 12 |
| Audio sigma shift | unused | 3 |
| Training block range | 16–16 | 4–8 |
| Supervised heads per update | 2 | 2 |
| Teacher solver / backward simulation | Euler / staggered | Euler / staggered |
| Objective | image MSE | video MSE + audio MSE, equal modality weights |
| Learning rate / Adam betas | 5e-6 / (0.9, 0.99) | same |
| Backbone LoRA | rank/alpha 128/128 | same |
| AdaLN and final modulation LoRA | rank/alpha 16/16 | same |
| Parameter / compute precision | BF16 backbone, FP32 LoRA, BF16 FSDP compute and FP32 reduction | same |
| Gradient accumulation / clipping | 1 / norm 1 | same |
| Default student updates | 30,000 | 30,000 |
| Checkpoints | every 100 updates, retain 2 | same |
| Guidance / DeCFG | guidance 1, DeCFG disabled | same |

Every iteration updates the student. The 30,000-iteration budget is not a
PDMD single-role budget. Staggered backward simulation can carry a sample
across updates, so iteration count also does not equal unique prompt count.

## Image: eight-GPU ACP pipeline

Config: `configs/train/pdd/minimax_h3_image_pdd2_2048_fsdp8.yaml`.

The image recipe uses `model.image_head_mode: base`, restoring the original
single output projection from the native transformer shard. The preserved
head is **an experimental continuous teacher**: the official checkpoint is
documented for six-step PDD inference, and MiniMax does not separately
validate its original output head as a continuous-score teacher. See
[the adapter transfer audit](minimax_h3_image_pdmd_transfer_audit.md).

Use the environment and model downloads from
[the image preprocessing guide](minimax_h3_image_pdmd2.md#environment-and-weights).
PDD also needs the native `transformer/model.safetensors.index.json` and the
shard containing `final_layer.video_out.weight` and `.bias`; the full model
download already includes them. The pretrained body still loads from
`transformer_diffusers/`.

Run once on one eight-GPU ACP worker:

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate

H3_CODE_ROOT="$PWD" \
H3_PYTHON="$VIRTUAL_ENV/bin/python" \
H3_IMAGE_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3-Image \
H3_IMAGE_PE_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3-Image-PE \
H3_IMAGE_DATASET_DIR=/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_image_finet2i_10k_2048 \
H3_PDD_OUTPUT=/mnt/devsft_afs_2/gushiqiao/outputs/h3_image_pdd2_10k_2048 \
H3_PDD_MAX_ITERS=30000 \
bash lightx2v_train/scripts/run_minimax_h3_image_pdd2_fsdp8_acp.sh
```

The same 10,000-prompt pipeline runs selection → eight PE shards → eight
text-encoding shards → eight-way FSDP training. It reuses completed prompt
and condition files. Training starts only after every preprocessing stage
succeeds. The cache contains text conditions and the 2048×2048 latent
geometry; paired clean image latents are not required for teacher-trajectory
training. Append `--dry-run` to inspect the commands without launching.

## FL2AV: 32-GPU ACP

Config: `configs/train/pdd/minimax_h3_fl2av_pdd8_fsdp32.yaml`.
This uses the existing T2AV/I2AV/FL2AV cache and its stratified three-task
sampler. Metadata determines latent geometry; portrait samples retain their
orientation. Since ranks carry trajectories across updates, every individual
optimizer step is not required to contain all three tasks. The sampler's
route weight stays with its trajectory. This launcher does not recache or
combine separate new/old datasets.

Run the same command on all four ACP workers. ACP must supply a common
`MASTER_ADDR` and `MASTER_PORT`:

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate

H3_CODE_ROOT="$PWD" \
H3_PYTHON="$VIRTUAL_ENV/bin/python" \
H3_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3 \
H3_PDD_CACHE=/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_dmd_cache_5s_768p/task_aware_view_3task_portrait \
H3_PDD_OUTPUT=/mnt/devsft_afs_2/gushiqiao/outputs/h3_fl2av_pdd8_32gpu \
H3_PDD_MAX_ITERS=30000 \
H3_RDZV_ID=h3_fl2av_pdd8_32gpu \
KERNELS_CACHE=/mnt/lm_data_afs/gushiqiao/cache/kernels \
bash lightx2v_train/scripts/run_minimax_h3_fl2av_pdd8_fsdp32_32gpu_acp.sh
```

## Ref2AV: 32-GPU ACP

Config: `configs/train/pdd/minimax_h3_ref2av_pdd8_fsdp32.yaml`.
This loads the Ref backbone and retains the image-only reference cache's
`count_random` sampler, including its one-to-six reference-image range.

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate

H3_CODE_ROOT="$PWD" \
H3_PYTHON="$VIRTUAL_ENV/bin/python" \
H3_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3 \
H3_PDD_CACHE=/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/latent_match124_bf16/metadata.jsonl \
H3_PDD_OUTPUT=/mnt/devsft_afs_2/gushiqiao/outputs/h3_ref2av_pdd8_32gpu \
H3_PDD_MAX_ITERS=30000 \
H3_RDZV_ID=h3_ref2av_pdd8_32gpu \
KERNELS_CACHE=/mnt/lm_data_afs/gushiqiao/cache/kernels \
bash lightx2v_train/scripts/run_minimax_h3_ref2av_pdd8_fsdp32_32gpu_acp.sh
```

Use a distinct output directory and rendezvous ID for each experiment.
`resume.auto_resume` resumes checkpoints from the same PDD experiment;
existing PDMD optimizer states are not interchangeable with these runs.
Logs are under `$H3_PDD_OUTPUT/logs/`, and training outputs under
`$H3_PDD_OUTPUT/train/`. PDD checkpoints include trained output heads as well
as the backbone adapter. Inference must load both the LoRA and the complete
`pdd_heads.safetensors`, then use `sample_pdd` in
`lightx2v_train/trainers/pdd/sampler.py` for head grouping and sampling.
A generic adapter-only Euler call does not load or use the new heads.

## Validation

The implementation was checked against FastGen at the commit linked above:
losses, gradients, teacher integration and audio/video sigma mapping have
numerical reference tests. The combined PDD, image-model and launcher suite
passed 38 tests and six subtests.

Two-H200 checks also passed:

- A small PDD model trained for three updates, saved the optimizer and
  per-rank trajectory state, then resumed through update five.
- The real 33B H3-Image student and teacher completed one 2048×2048 update
  using a real cached text condition, 32 heads, and the configured LoRA ranks.
  Peak allocated memory was 70.34 GiB per GPU. The update took about 5.3 s
  after loading and initial rollout; this is a single-update check, not a
  steady-state throughput measurement.
- Small native T2AV and Ref2AV models each completed two distributed updates
  and fused-head sampling with different trajectory positions on each rank.

These checks validate execution and checkpoint recovery, not convergence or
image/video quality. No long PDD training run was started. Local smoke logs
are in `/data/nvme3/gushiqiao/verification/h3_image_pdmd_20261111/`.

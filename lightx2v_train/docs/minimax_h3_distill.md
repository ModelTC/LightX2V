# MiniMax-H3 dCM and PDMD training

Use a converted Diffusers H3 model root with `transformer/` for FL2AV and `transformer_ref/` for Ref2AV. The training environment must provide MiniMax-H3 in Diffusers and the `_flash_3_hub` attention backend.

| Task | 4-step PDMD | 8-step PDMD | dCM video/audio shift | PDMD video/audio shift |
| --- | --- | --- | --- | --- |
| FL2AV | [Config](../configs/train/pdmd/minimax_h3_fl2av_pdmd4_gan_fsdp32.yaml) | [Config](../configs/train/pdmd/minimax_h3_fl2av_pdmd8_gan_fsdp32.yaml) | 6/3 | 6/3 for updates 1–1,500, 12/3 for updates 1,501–3,000 |
| Ref2AV | [Config](../configs/train/pdmd/minimax_h3_ref2av_pdmd4_gan_fsdp32.yaml) | [Config](../configs/train/pdmd/minimax_h3_ref2av_pdmd8_gan_fsdp32.yaml) | 12/3 | 12/3 |

Both tasks train dCM for 1,500 student updates and use the final student EMA to initialize PDMD. PDMD runs five fake updates followed by one student update. Each role update counts as one iteration, so 3,000 iterations contain 2,500 fake and 500 student updates. Checkpoints are saved every 500 dCM updates and 100 PDMD updates, keeping two per stage.

PDMD uses student/fake learning rates of `5e-5`/`4e-6`, AdamW betas `(0, 0.9)`, and zero weight decay. The GAN head uses the fake optimizer settings. Generator and discriminator loss weights are `0.005` and `0.01`. The student uses rank/alpha 128/128, with AdaLN and the output normalization projection at 16/16.

## Data

Use `data.train.name: cache_dataset` and point `data.train.data_path` to a `meta.json` file. The supplied training configs read the path from `H3_TRAIN_META`. The file contains a JSON array of cached samples.

### FL2AV

Set `model.name: minimax_h3_fl2av` for `t2av`, `i2av`, or `fl2av`. Each record carries one task. The condition cache contains the prompt embeddings and, for I2AV or FL2AV, the encoded first frame or first/last frames.

```json
[
  {
    "task": "t2av",
    "prompt": "A boat crosses the lake.",
    "training_cache": "conditions/boat.pt",
    "num_frames": 124,
    "target_height": 768,
    "target_width": 1344,
    "video_latent_path": "video_latents/boat.pt",
    "audio_latent_path": "audio_latents/boat.pt"
  },
  {
    "task": "i2av",
    "prompt": "The camera follows the boat.",
    "training_cache": "conditions/follow_boat.pt",
    "num_frames": 124,
    "target_height": 768,
    "target_width": 1344
  },
  {
    "task": "fl2av",
    "prompt": "The boat pulls into the dock.",
    "training_cache": "conditions/boat_at_dock.pt",
    "num_frames": 124,
    "target_height": 768,
    "target_width": 1344
  }
]
```

Each global batch covers every active task across the data-parallel ranks. Each rank loads one sample. With three tasks on 32 ranks, the sampler assigns 10, 11, and 11 samples, rotating the extra slots across batches. Smaller task pools are reused as they run out. The default uses all FL2AV-compatible tasks in the file. Use `H3_TASKS='[t2av]'` to select one task from a mixed file.

### Ref2AV

Set `model.name: minimax_h3_ref2av`. `ref_image_count` gives the number of encoded reference images in the cache. This example uses two images.

```json
{
  "task": "ref2av",
  "prompt": "The two referenced people walk through a garden.",
  "training_cache": "conditions/garden.pt",
  "ref_image_count": 2,
  "num_frames": 124,
  "target_height": 1344,
  "target_width": 768,
  "video_latent_path": "video_latents/garden.pt",
  "audio_latent_path": "audio_latents/garden.pt"
}
```

Use `H3_REF_IMAGE_COUNTS='[2]'` for a two-image run or the default `[1,2,3,4,5,6]` for mixed counts. The sampler uses `ref_image_count` to spread each count across global batches while preserving its frequency in the data as closely as full batches allow. The shipped Ref configs drop the final incomplete batch.

### Cache paths and training methods

Paths can be absolute or relative to `meta.json`. For existing H3 condition caches, set `training_cache` to the old `condition_path`. Use the prompt that was encoded for that cache. The target geometry describes the rollout. Visual conditions must match it. The H3 cache uses 24 fps.

- **dCM** selects samples with paired target latents. Provide both `video_latent_path` and `audio_latent_path` for external targets. The video latents use H3 VAE normalization. For a full training cache that already contains paired `inputs.video_latents` and `inputs.audio_latents`, mark the record with `"has_targets": true`.
- **DMD and PDMD** can also sample condition-only records.
- **GAN** is enabled with `training.dmd.gan.enabled: true` in either algorithm. Each rollout geometry needs a paired target for the discriminator. The positive sample carries its own conditioning.

## Launch with torchrun

These configurations use 32 GPUs. The example uses four nodes with eight GPUs each. On every node, activate your training environment and run the following from the repository root. Set `MASTER_ADDR` to the hostname or IP address of node 0, reachable from every node, and set `NODE_RANK` to `0`, `1`, `2`, or `3` on the corresponding node. Use shared paths for the data and output directories.

```bash
cd lightx2v_train
export PLATFORM=nvidia
export PYTHONPATH="$PWD/..${PYTHONPATH:+:$PYTHONPATH}"

export H3_MODEL_PATH=/path/to/MiniMax-H3
export H3_TRAIN_META=/path/to/data/meta.json
export H3_DCM_OUTPUT=/path/to/output/dcm

MASTER_ADDR=node0.example.com
NODE_RANK=0
TASK=ref2av          # Use fl2av for T2AV, I2AV, or FL2AV training.
PDMD_STEPS=8         # Use 4 for four-step distillation.
export H3_PDMD_OUTPUT="/path/to/output/pdmd${PDMD_STEPS}"
```

### dCM

```bash
torchrun --nnodes=4 --nproc-per-node=8 \
    --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port=29500 \
    train.py --config "configs/train/consistency/minimax_h3_${TASK}_dcm_fsdp32.yaml"
```

Set `H3_DCM_TEACHER_PATH` to use a separate teacher model directory.

### PDMD with GAN

After dCM finishes, point `H3_STUDENT_INIT` to its final EMA and start PDMD on every node.

```bash
printf -v dcm_step '%09d' "${H3_DCM_TRAIN_ITERS:-1500}"
export H3_STUDENT_INIT="$H3_DCM_OUTPUT/checkpoint-$dcm_step/student_ema"

torchrun --nnodes=4 --nproc-per-node=8 \
    --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port=29501 \
    train.py --config "configs/train/pdmd/minimax_h3_${TASK}_pdmd${PDMD_STEPS}_gan_fsdp32.yaml"
```

Each stage writes `train.log` and checkpoints to its output directory.

The [FL2AV DMD config](../configs/train/dmd/minimax_h3_fl2av_dmd_lora.yaml) uses the same `cache_dataset` loader. Set `training.dmd.gan.enabled` to `true` and specify the fake-model block in `training.dmd.gan.feature_block`. GAN training requires paired targets.

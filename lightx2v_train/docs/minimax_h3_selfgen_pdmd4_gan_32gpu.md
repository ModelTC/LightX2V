# H3 selfgen PDMD4 + GAN: shift 6 for 1000 updates, shift 12 for 2000

This experiment uses the FL2AV base transformer with **T2AV conditioning and
paired video/audio targets**. It does not train first/last-frame-only FL2V.
Only the existing self-generated cache is used; there is no action-prompt
preprocessing stage, old-data curriculum, or new latent generation.

| Optimizer update, one-based | Video shift | Audio shift | Rollout steps |
| --- | --- | --- | --- |
| 1–1000 | 6 | 3 | 4 |
| 1001–3000 | 12 | 3 | 4 |

Iterations count optimizer-role updates, not student updates. With five fake
updates followed by one student update, the total is **2500 fake + 500 student**.
The first phase has 834 fake + 166 student updates; the second has 1666 fake +
334 student updates. GAN is enabled from the first update in both phases.

The DCM1700 EMA LoRA export initializes the live student only. The fake and
teacher start from the original H3 base, and the GAN classifier starts fresh.
The student uses rank/alpha 128/128, with AdaLN/output norm projections 16/16.
Student/fake learning rates are 2e-5/4e-6. GAN G/D weights are 0.005/0.01.
FSDP32, batch1/rank, accumulation1 and the previous precision settings remain.

## Launch on four nodes

Synchronize the updated training code to the cluster first. Run the following
same command once on each of four nodes with eight GPUs. ACP must supply the
same reachable `MASTER_ADDR` and `MASTER_PORT` to all nodes.

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate

export H3_CODE_ROOT="$PWD"
export H3_PYTHON="$VIRTUAL_ENV/bin/python"
export H3_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3
export H3_STUDENT_INIT=/mnt/lm_data_afs/gushiqiao/outputs/h3_fl2av_dcm1700_ema_init
export H3_SELFGEN_CACHE=/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_self_generated_latents_124_bf16/t2av/metadata.jsonl
export H3_PDMD_OUTPUT=/mnt/lm_data_afs/gushiqiao/outputs/h3_selfgen_pdmd4_gan_dcm1700_shift6_1000_shift12_2000_20261123
export H3_RDZV_ID=h3_selfgen_pdmd4_gan_shift6_1000_shift12_2000_20261123
export KERNELS_CACHE=/mnt/lm_data_afs/gushiqiao/cache/kernels
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash lightx2v_train/scripts/run_minimax_h3_selfgen_pdmd4_gan_32gpu.sh
```

Append `--dry-run` for model/cache/adapter metadata checks and the printed
32-rank torchrun command. Dry-run neither starts training nor publishes a new
data view. It requires access to the actual cluster files; the local CPU test
uses fixtures and does not validate the cluster's weights or train 32 GPUs.

The source must contain the full `metadata.jsonl` with `condition_path`,
`video_latent_path` and `audio_latent_path`, not `condition_metadata.jsonl`.
As in the previous selfgen GAN recipe, it selects T2AV, 124 frames, 24 fps,
768×1344 landscape rows, with at least 64 eligible examples. The immutable view
is published under `$H3_PDMD_OUTPUT/data/selfgen`. Conditioning and GAN positives
use the same eligible population with independent sampling streams. Positives
are teacher-generated examples, not camera recordings. The original cache is
not rewritten, and the view references its existing latent files.

## Schedule, resume and outputs

This is **one training process and one immutable recipe**, not a weights-only
handoff between two runs. At completed update1000 the next rollout uses shift12.
Student/fake/GAN weights, optimizers, schedulers and data cursors remain intact.
The boundary is a fake update and therefore does not split a cached
fake-trajectory/student pair. General schedules reject such unsafe boundaries.

Rerun the same command and output path to resume. Checkpoints record the entire
schedule and validate it strictly; do not edit the YAML mid-run. Restoring a
checkpoint at1000 validates its last-used shift6 before switching the next
update to12. Later checkpoints restore shift12. There is no relaxed recipe
check or optimizer reset. Inference previews are disabled in this recipe.

```text
$H3_PDMD_OUTPUT/
├── train.log
├── logs/launch.<node>.<timestamp>.<pid>.log
├── data/selfgen/t2av/landscape/metadata.jsonl
└── checkpoint-000003000/
    ├── pytorch_lora_weights.safetensors
    ├── adapter_config.json
    ├── trainer_state.pt
    ├── dist_state/
    ├── official_pdmd.rank*.pt
    ├── rollout_schedule.json
    └── official_pdmd.complete
```

Checkpoints are saved every100 role updates, retaining the latest two, as in
the previous GAN recipe. The final export should be sampled with **four steps,
video shift12 and audio shift3**. `rollout_schedule.json` records the last-used
and next-update shifts; the copied YAML intentionally retains the whole schedule.

The new CPU regression covers exact phase selection at999/1000, uninterrupted
versus resumed AdamW updates on both sides of the transition, cached trajectory
reuse before a student step, H3 checkpoint metadata, and rejection of a changed
schedule. Actual 32-GPU speed, memory use and generated quality require the
cluster run.

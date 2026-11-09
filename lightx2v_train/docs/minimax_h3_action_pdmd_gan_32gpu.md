# H3 action T2AV: PDMD + GAN, 32 GPUs

This documents the single-stage component. For the requested action-only PDMD
100-student-update stage followed by self-generated PDMD + GAN, use
`scripts/run_minimax_h3_action_pdmd_selfgen_gan_fsdp32_32gpu_acp.sh` and
`minimax_h3_action_pdmd_selfgen_gan_32gpu.md`.

Run the same launcher on all four ACP workers, with eight GPUs visible per worker. It first encodes the external 2,000 action prompts, merges the 32 global-rank cache shards, and then starts the new training stage. Each rank encodes 62 or 63 prompts. It does not generate target videos during prompt preprocessing.

```bash
export H3_CODE_ROOT=/path/to/LightX2V
export H3_PYTHON=python
export H3_GAN_REAL_CACHE=/path/to/real/t2av/metadata.jsonl
bash "$H3_CODE_ROOT/lightx2v_train/scripts/run_minimax_h3_t2av_action_pdmd8_gan_fsdp32_32gpu_acp.sh"
```

ACP must provide the same `MASTER_ADDR` and `MASTER_PORT` on all workers. Preprocessing uses a separate rendezvous port, `MASTER_PORT + 1`; override `H3_GAN_PREPROCESS_PORT` if needed. Both ports must be reachable between workers. `H3_GAN_PROMPT_CACHE` must be shared storage visible under the same path on every worker. The preprocessing workers exit before training starts, releasing the Qwen model and its GPU allocations.

`H3_GAN_REAL_CACHE` is required. It accepts a manifest or a directory containing `metadata.jsonl`. Eligible rows must be T2AV, 124 frames, 768×1344, 24 fps, and include their own caption, `condition_path`, `video_latent_path`, and `audio_latent_path`. Other geometries/tasks are filtered. The real stream uses each real sample's matching text condition. The 2,000 prompt-only records are insufficient as GAN real data. Teacher-generated full AV caches may be used as the real distribution once their matching conditions and latents exist.

The new configuration is `configs/train/dmd/minimax_h3_t2av_action_pdmd8_gan_shift12_3_r128_a128_adaln16_fsdp32.yaml`:

| Setting | Value |
| --- | --- |
| PDMD rollout | 8 steps, video/audio shifts 12/3 |
| Update order | 5 fake updates, then 1 student update |
| Student | LoRA rank/alpha 128/128; block/output AdaLN 16/16; no EMA |
| Fake | Full parameters, FP32 master weights |
| Parallelism | FSDP32, BF16 compute, FP32 reduction, no sequence parallelism |
| GAN loss weights | Student 0.005, discriminator 0.01 |
| GAN head | Feature block 49, hidden width 1024, 4 temporal bins |
| Budget | 15,000 role optimizer updates; override `H3_GAN_MAX_ITERS` |
| Checkpoints | Every 100 role updates, keep 2 |

Student/fake learning rates stay at `2e-5` / `4e-6`, with AdamW betas `(0, 0.9)`, no weight decay and global clipping at 1. The classifier uses the fake optimizer settings and update cadence. Its replicated gradients are averaged across ranks and clipped together with the sharded fake-backbone gradients, using one common norm. The budget counts one role update per iteration: 15,000 iterations contain 2,500 student updates and 12,500 fake updates.

The GAN objectives follow [DMD2's released implementation](https://github.com/tianweiy/DMD2/blob/main/main/sd_guidance.py):

```text
student loss = official PDMD loss + 0.005 * mean(softplus(-D(generated)))
fake loss = fake flow-matching loss
          + 0.01 * (mean(softplus(D(generated.detach())))
                  + mean(softplus(-D(real))))
```

The discriminator shares the trainable fake-score H3 backbone; only the small classification head is additional. Student GAN backward freezes fake/head parameters but retains the gradient through their inputs. H3 uses native RF noising at 1,000 physical sigma levels from 0 to 0.999, independently drawn for each GAN call; video and audio use the same sigma in that call. The head pools target video tokens into four ordered bins, and audio/text condition the shared backbone. This is an H3 adaptation of the DMD2 GAN, not a reproduction of its SDXL UNet, DDIM scheduler or overlapping optimizer-update loop. The existing official PDMD trajectory and 5-fake/1-student role schedule remain in place.

The DMD2 reference uses real images. Using `self_generated` teacher AV caches as positive samples instead makes this synthetic-data supervision. With an independent cache, generation and positive-sample prompt distributions should overlap: otherwise a conditioned discriminator can learn the prompt-domain difference. Both streams' metadata are fingerprinted for restart validation; cache payload files are assumed immutable.

The source defaults to:

```text
/mnt/lm_data_afs/gushiqiao/outputs/h3_fl2av_pdmd8_from_dcm1700_ema_32gpu_20261029/checkpoint-000005800
```

The live student adapter and fake model weights initialize from this checkpoint. Optimizers, data cursors, role counters, and the GAN head start fresh. Training starts at role iteration 0. Its own output directory supports automatic continuation, including the GAN state. Reusing the source PDMD output directory is rejected because shifts, update ratio, data, and GAN state have changed. The frozen teacher remains the base H3 model.

Own-stage checkpoints are stored directly at `$H3_GAN_OUTPUT/checkpoint-NNNNNNNNN`. On restart, preflight uses the trainer's latest-checkpoint selection and checks the complete GAN state. A valid own-stage checkpoint can resume after the original checkpoint 5800 has been pruned; the original files are required only for a fresh stage. An incomplete latest selected checkpoint is rejected, with no silent fallback to an older checkpoint or fresh warm-start. The trainer subsequently validates the full saved recipe and cursor state.

The external prompt input defaults to `/mnt/lm_data_afs/gushiqiao/datasets/data.json`; set `H3_GAN_PROMPTS` to use a different file with exactly 2,000 T2AV records. JSON arrays, `data`/`records`/`examples` wrappers and JSONL content are supported, including a `.json` suffix on JSONL data. Defaults for deployment paths are:

| Environment variable | Default |
| --- | --- |
| `H3_MODEL_PATH` | `/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3` |
| `H3_GAN_PROMPT_CACHE` | `/mnt/lm_data_afs/gushiqiao/datasets/h3_action_prompts_2000_20261109/condition_cache_124_768p_bf16` |
| `H3_GAN_INIT_CHECKPOINT` | Source checkpoint above |
| `H3_GAN_OUTPUT` | `/mnt/lm_data_afs/gushiqiao/outputs/h3_t2av_action_pdmd8_gan_shift12_3_fake5_from5800_32gpu_20261109` |
| `H3_GAN_CONFIG` | New configuration above |

Restarting preprocessing reuses only caches whose prompt, model, geometry, dtype, and schema fingerprints match. Each rank publishes its own manifest. Rank 0 requires disjoint, complete coverage before atomically publishing `metadata.jsonl` and `prompt_cache.complete.json`; any encoding/merge failure prevents training. Changing the prompt/model recipe requires a new cache directory.

Each prompt encoder writes its detailed output and traceback to `$H3_GAN_PROMPT_CACHE/logs/rank-NNNNN.TIMESTAMP.PID.log`; the console reports its assigned count, completion, error, and log path. Each ACP worker also tees the entire launcher output to `$H3_GAN_OUTPUT/logs/launch.HOST.TIMESTAMP.PID.log`. Set `H3_GAN_LAUNCH_LOG=0` to disable this launcher tee. The training logger additionally writes `$H3_GAN_OUTPUT/train.log`. New attempts create new preprocessing and launcher logs.

Append `--dry-run` to print the preflight, preprocessing, cache verification, and training commands without starting processes. To validate resources without encoding or training, run the printed first preflight command in the training environment with the same exported paths.

Local verification covers cache sharding, exact manifest coverage, stale-cache rejection, repeated preprocessing, launch topology, and prompt parsing using CPU tests. Actual 32-GPU model loading, source checkpoint compatibility, and training memory must be checked on ACP where the model and checkpoint paths are mounted.

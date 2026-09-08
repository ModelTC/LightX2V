# H3 VAE decoder: layer search and recovery

This is a decoder-only adaptation of [TinyFusion](https://github.com/VainF/TinyFusion), not a diffusion noise-prediction training run. The original 36-block, width-2048 decoder becomes a 12-block decoder; the encoder, latent normalization, register tokens, RoPE, output projection and unpatchification remain compatible with H3. The maintained VAE workflows are this decoder search/recovery and [encoder search/recovery](minimax_h3_encoder_pruning.md).

## Stage 1: search which layers to keep

The 36 original blocks are divided into 12 consecutive groups of three. Hard Gumbel-Softmax gates select one block per group. Base weights are frozen; rank-16 temporary LoRA adapters (`lora_alpha: 32`) and gate logits are optimized using reconstruction + LPIPS. The search still computes candidate blocks, so its runtime is not the final pruned decoder's runtime.

The default run is 1,000 optimizer updates, learning rate `1e-4`, gate learning-rate multiplier `10`, temperature `4 -> 0.1`, EMA decay `0.999`, and four gradient-accumulation micro-batches per rank. Each micro-batch contains one video sample. With four DDP ranks, an update sees 16 sampled videos/windows. Checkpoints are saved every 100 updates.

The final EMA gate selection is written to:

```text
output_train/minimax_h3_vit_prune_search_4gpu_ddp/export/kept_layers.json
```

Recovery reloads selected blocks from the original pretrained H3 weights, discarding the temporary search adapters. Do not use the stage-1 LoRA weights as the final decoder.

## Stage 2: recover the fixed student

All student decoder parameters are trainable. The frozen original decoder supplies intermediate features for the same latent tile/window. Student blocks 0..11 are supervised by teacher blocks `[2, 5, ..., 35]` (zero-based group endpoints), independently of which block in each group provided initialization.

| Updates | Intermediate MSE | RGB reconstruction | LPIPS | Adversarial | Auxiliary |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–599 | 0.01 | 1.0 | 0.1 | 0 | 0 |
| 600–2,999 | 0.005 | 1.0 | 0.2 | 0.5 | ramp to 0.1 |

Feature MSE uses FP32 and averages the matched video-token features. `feature_loss_type: masked_mse` enables TinyFusion-style outlier masking instead; ordinary `mse` is the default. Auxiliary tokens remain in attention even though feature supervision uses video tokens.

RGB reconstruction is Charbonnier against the original API video, not the teacher's decoded RGB. LPIPS is VGG-based and samples up to 64 frames available in the selected group. The conditional 3D discriminator plus frame-wise 2D PatchGAN use normalized adversarial-correction target MSE for the generator, not a direct BCE/hinge loss; `adversarial: 0.5` scales that objective. Discriminators start at update 600 with a 100-update warmup followed by a 300-update generator ramp. The auxiliary branch ramps over 200 updates from update 600 and matches teacher reconstruction through a frozen teacher suffix; see [reconstruction-aware recovery](minimax_h3_vit_pruning_aux.md). No standalone frequency, seam, gradient, KL, or teacher-RGB loss is enabled.

Training uses H3's `256 x 256` spatial tiles with `64` overlap. Initial sampling is 75% single / 25% temporal pair. Later substages add horizontal/vertical pairs, spatial quads and temporal quads; student tiles are overlap-blended before RGB loss. Teacher feature comparisons remain tile/window-aligned. This preserves H3 attention context instead of comparing single-tile student features with cropped full-frame teacher features.

Both stages reuse the current four cached-latent manifests (T2AV, I2AV, FL2AV, L2AV; 37,649 rows when checked). They reconstruct the source videos regardless of the generation task. `geometry_from_metadata: true` reads actual cache geometry; `num_frames: 362` is a fallback, not forced stretching of the 5-second cached videos. No recaching is needed. The three preview samples come from the training manifest and are progress checks, not a held-out validation set.

Stage 2 runs 3,000 updates with `5e-5` initial LR, cosine decay and 200 warmup updates. Checkpoints are saved every 100 updates (latest five retained); three full-video reconstructions and PNG comparisons are saved every 100 updates. LPIPS/GAN frame limits cannot exceed the frames actually present in a sampled window group. A temporal quad can cover roughly 60+ supervised frames.

## Commands

```bash
cd /data/nvme6/gushiqiao/codes/latest/vae/LightX2V/lightx2v_train
GPU_LIST=0,1,2,3 bash scripts/run_minimax_h3_vit_prune.sh all
```

Or run stages separately:

```bash
GPU_LIST=0,1,2,3 bash scripts/run_minimax_h3_vit_prune.sh search
GPU_LIST=0,1,2,3 bash scripts/run_minimax_h3_vit_prune.sh recover
```

The script defaults to the existing H3 `local_diffusers/.venv` Python. Override `H3_VAE_PYTHON` if needed; that environment needs `lpips`. `recover` and `all` use the maintained `minimax_h3_vit_prune_recover_aux_3k_4gpu_ddp.yaml` configuration. Repeating a command resumes its own latest completed checkpoint. Use new `H3_VAE_PRUNE_SEARCH_OUTPUT` and `H3_VAE_PRUNE_RECOVER_AUX_3K_OUTPUT` directories for a fresh experiment; `H3_VAE_PRUNE_SELECTION` can select a different exported architecture.

Four-GPU FSDP2 variants are also supplied. They keep original parameters and reduction in FP32, use BF16 autocast for eligible operations, and shard decoder blocks with sequence parallelism disabled:

```bash
GPU_LIST=0,1,2,3 H3_VAE_PARALLEL=fsdp bash scripts/run_minimax_h3_vit_prune.sh all
```

FSDP uses separate `*_4gpu_fsdp` output directories. The frozen teacher is still replicated, not sharded. Syntax and CPU-level tests do not establish GPU memory usage or multi-rank performance; these configs need a short real GPU run before committing a long training job.

These are initial 36-to-12 recovery settings, not a guarantee of original-decoder quality. Search/recovery quality needs to be checked on fixed held-out clips; shape compatibility and speed do not establish perceptual equivalence.

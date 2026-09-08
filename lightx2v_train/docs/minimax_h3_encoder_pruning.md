# H3 encoder: keep-three search and recovery

These two stages prune only the original causal CNN encoder. The original full H3 decoder is frozen and supplies RGB gradients to the student encoder; it is not replaced by a pruned or CNN decoder. Existing decoder runs, configurations and checkpoints are unchanged. The launchers default to physical GPUs 6 and 7 with two DDP ranks.

## Architecture and search

Each of the encoder's 12 residual blocks computes `P(x) + F(x)`, where `F` contains two 3D convolutions and `P` is either identity or a channel-changing projection. Pruning removes only `F`. All projection shortcuts, four spatial/temporal downsamplers, input/output convolutions, output normalization and `quant_conv` remain.

The fixed keep-three encoder contains 16 Conv3d modules including `quant_conv`, versus 34 in the original: ten mandatory convolutions plus six convolutions in the three retained branches. Encoder-only counts excluding `quant_conv` are 15 versus 33. The posterior still has 48 channels (24 means plus 24 log-variances), and the decoder receives the same normalized 24-channel latent interface. Fewer convolutions do not imply proportional end-to-end speedup, and search computes all candidate branches.

Search uses a hard, straight-through Gumbel categorical choice over the 220 valid three-of-twelve subsets. Each subset's score is the sum of its three learned branch logits, multiplied by `gate_scale: 100`. Every forward selection has exactly three active residual branches; this is neither independent Bernoulli gating nor one choice per encoder stage. Gates and rank-16 temporary Conv3d LoRA adapters (`lora_alpha: 32`) on all candidate `conv1`/`conv2` branches are trained. Original encoder weights, projections, norms and `quant_conv` are frozen during search.

The final EMA gate scores select three branches. Search exports the fixed architecture initialized from the original pretrained teacher weights, not the adapted search weights. Its `export` directory contains:

```text
kept_layers.json                       # contains kept_residual_indices
pruned_encoder_config.json
minimax_h3_pruned_encoder.safetensors
```

Temporary LoRA and gates are absent from this export. Recovery loads that export, starts a new optimizer/schedule at update zero, and trains all remaining encoder parameters, including retained projections, boundaries and `quant_conv`. A search checkpoint is only for resuming search; it is not a recovery initialization.

## Targets and gradient paths

The four existing T2AV/I2AV/FL2AV/L2AV manifests provide paths and geometry for the original RGB videos. The `minimax_h3_pruned_encoder` sample processor ignores latent caches; the configs do not request `video_latent_path`. Teacher posterior targets are freshly encoded from the same RGB tiles used by the student. No encoder-latent recaching is needed.

```text
original RGB x
  +-- frozen teacher encoder --------------------> teacher posterior / stage features
  |                                                  +-- frozen full decoder --> YT [detached]
  +-- trainable student encoder -----------------> student posterior
          |                                          +-- frozen full decoder --> YS
          +-- stage feature HS --> frozen teacher encoder suffix + quant_conv
                                                     +-- frozen full decoder --> YA
```

Teacher targets use `no_grad` and are detached. On the student RGB path, decoder parameters are frozen but its forward retains input gradients: reconstruction, LPIPS and generator losses propagate through the full original decoder to the student encoder. Wrapping this path in `no_grad` would sever the training signal. RGB decoding uses posterior mode, not a sampled latent.

Feature MSE matches FP32 student and detached teacher activations at the six post-downsample stage boundaries, `teacher_feature_indices: [0, 1, 2, 3, 4, 5]`. These are encoder stage indices, not the retained residual indices or decoder transformer-layer indices. No feature softmax, normalization or KL is applied.

The auxiliary branch samples one student stage from `[2, 3, 4]` per sample, using the same anchor for its tiles. It runs the remaining frozen teacher encoder stages, output head and `quant_conv`, then mode normalization and the frozen full teacher decoder. This entire auxiliary path retains input gradients. Its loss updates only the traversed student prefix; the main and posterior losses still supervise the complete student encoder. In symbols, its gradient is `J_student_prefix^T J_teacher_encoder_suffix^T J_teacher_decoder^T grad_YA`, with normalization included in the composed maps. Teacher parameters receive no updates, and no explicit Jacobian or second-order optimizer is needed.

## Losses

Main RGB reconstruction is Charbonnier against the original video, not teacher-decoded RGB. VGG LPIPS also uses the original RGB target. Recovery reuses the existing conditional 3D discriminator plus frame-wise 2D PatchGAN: discriminator training uses the existing least-squares objective, while the encoder's generator objective is MSE toward the detached normalized adversarial-correction target. GAN conditioning uses detached teacher latents. The discriminators have their own optimizer; they do not update the teacher encoder or decoder.

Posterior alignment has constant weight 1 in both stages. Let `s` be the published per-channel `latents_std`, and let `sigma = exp(0.5 * clamp(logvar, -30, 20))`. After native spatial blending of the full mean/log-variance moments:

```text
Lposterior = mean(((muS - muT) / s)^2)
           + mean(((sigmaS - sigmaT) / s)^2)
```

This is the expected normalized-latent MSE under shared posterior sampling noise. It preserves both mean and uncertainty; it is not raw log-variance MSE or KL toward a unit Gaussian. The published latent mean cancels in the difference. The RGB mode path supervises means, while this additional term also supplies gradients to the posterior uncertainty channels.

For auxiliary teacher RGB target `YT` and prediction `YA`, let `E = YA - YT` and `rho(u) = sqrt(u^2 + 0.001^2) - 0.001`:

```text
Laux = mean rho(E)
     + 0.1 * mean(d in {H,W}) mean rho(delta_d E)
     + 0.1 * mean(d in {H,W}) mean rho(delta_T delta_d E)
Ltotal = Lreconstruction + lambda_lpips * LLPIPS + Lposterior
       + lambda_feature * Lfeature + lambda_gan * LGAN + lambda_aux * Laux
```

The spatial term matches edges; the mixed temporal/spatial term matches their changes between adjacent valid frames. It does not force motion to zero. Main teacher-output, velocity, acceleration, frequency, Laplacian and seam losses are disabled. The auxiliary RGB target is teacher reconstruction, unlike the main original-video RGB target.

## Schedule and geometry

The schedules retain the decoder search's 1,000-update settings and the existing auxiliary recovery's 3,000-update settings, with posterior alignment added. All indices below count optimizer updates, not micro-batches.

| Phase / updates | Reconstruction | Posterior | Feature | LPIPS | GAN | Auxiliary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Search 0–999 | 1 | 1 | 0 | 0.05 | 0 | 0 |
| Recovery 0–599 | 1 | 1 | 0.01 | 0.1 | 0 | 0 |
| Recovery 600–2999 | 1 | 1 | 0.005 | 0.2 | 0.5 | ramp to 0.1 |

Search uses AdamW learning rate `1e-4`, 50 warmup updates and cosine decay; gate LR is ten times the adapter LR. Gumbel temperature decreases linearly from 4 to 0.1, with gate EMA decay 0.999. Recovery uses `5e-5`, 200 warmup updates and cosine decay. Both use optimizer betas `0.9 / 0.999`, zero weight decay and gradient clipping at 1. Auxiliary loss ramps over 200 updates beginning at recovery update 600. GAN starts discriminator training at 600, warms it for 100 updates, then ramps generator correction over 300 updates. The adversarial weight remains 0.5 while that internal correction ramp changes.

Both phases initially sample 75% single-window and 25% temporal-pair routes. From recovery update 600, probabilities are single 20%, horizontal pair 15%, vertical pair 15%, spatial quad 20%, temporal pair 20%, temporal quad 10%. Training uses native 256-pixel tiles with 64-pixel overlap, causal 17-frame encoder clips, original padding/blending and final token-drop conventions. All encoder tiles intersecting the chosen decoder crop, including neighboring overlap tiles, contribute before posterior/RGB losses. Main and auxiliary RGB losses share valid-frame cropping and exclude padded RGB frames. Feature and posterior alignment use matching native encoder clip/latent padding on both paths; feature targets can include repeated final-frame padding.

Search LPIPS samples up to 32 frames, recovery up to 64, in batches of two. Both stages use student/teacher tile batches of one and checkpointed windows. GAN samples up to 32 available frames, with 256-pixel crops at 75% probability and 384-pixel crops at 25%; both discriminator architectures, normalization, spectral normalization and correction settings are copied unchanged from the 3k decoder recovery config. Its optimizer uses `2e-5`, betas `0.5 / 0.9`, and zero weight decay. A sampled route may contain fewer frames than any configured maximum.

Student parameters are FP32 with BF16 autocast for eligible operations. Teacher encoder and full decoder computation is FP32 on both target and differentiable paths. The training entry point still enables TF32 kernels; this does not promise strict IEEE-FP32 arithmetic. Teacher modules remain replicated per DDP rank. Checkpointing limits saved activations but does not remove the compute/memory cost of backward through the full decoder or auxiliary encoder suffix.

Each rank uses one video per micro-batch and four accumulation micro-batches. Two ranks therefore process eight sampled videos/window groups per optimizer update. Training uses four workers and `pin_memory: false`; validation uses two workers and also disables pinned memory. `geometry_from_metadata` preserves each record's geometry; `num_frames: 362` is a fallback cap, not forced expansion of five-second videos.

Both stages save every 100 updates and retain five checkpoints. They reconstruct one full-video preview at update zero before the first optimizer step, then every 100 updates, with six PNG comparisons. This example comes from a training manifest, so it is a progress check rather than held-out validation. Full-original-decoder preview is expensive; the initial sample limit is deliberately one. Search previews use the current hard selection and temporary adapters, not the unadapted final export.

## Separate commands and outputs

```bash
cd /data/nvme6/gushiqiao/codes/latest/vae/LightX2V/lightx2v_train
GPU_LIST=6,7 bash scripts/run_minimax_h3_encoder_prune_search.sh

# Run only after search finishes and publishes its export.
GPU_LIST=6,7 bash scripts/run_minimax_h3_encoder_prune_recover.sh
```

The scripts launch exactly two local ranks with standalone `torchrun`, using the H3 `local_diffusers/.venv/bin/python` interpreter. `H3_VAE_PYTHON` overrides the interpreter. Neither script starts another stage automatically, and recovery fails clearly if selection or exported encoder weights are missing. Default output directories are:

```text
output_train/minimax_h3_encoder_prune_search_keep3_2gpu_ddp
output_train/minimax_h3_encoder_prune_recover_keep3_2gpu_ddp
```

`H3_VAE_ENCODER_SEARCH_OUTPUT` and `H3_VAE_ENCODER_RECOVER_OUTPUT` independently override these directories. Recovery resolves its selection from `${H3_VAE_ENCODER_SEARCH_OUTPUT}/export/kept_layers.json`, unless `H3_VAE_ENCODER_SELECTION` explicitly chooses another encoder export. Old decoder-output environment variables have no effect. Each stage auto-resumes only its own output directory's completed checkpoint; use a new output directory for a fresh run. Recovery never auto-resumes the search checkpoint.

Only the recovered student encoder is exported for encoder inference; training-only teachers, auxiliary paths, search adapters and discriminators are not part of that artifact. The existing full decoder can consume its unchanged latent interface. CPU checks establish contracts and gradient connectivity, not two-GPU memory fit, convergence, reconstruction quality or measured encoder speed. No training is launched merely by creating these configurations.

## Short real-weight validation

The opt-in `tests/smoke_minimax_h3_encoder_distillation_gpu.py` was run on two H200s (physical 6,7), using the original 124-frame 768x1344 API video and released teacher weights. Search uses a new random hard mask; recovery uses a fixed untrained selection `[0,4,10]`. These are forward/backward/optimizer checks, not quality or convergence evaluations. Every trainable encoder gradient was present and finite, and every teacher parameter remained frozen with no gradient.

| Smoke route | Enabled objectives | Peak allocated per GPU | One micro-batch plus update |
| --- | --- | ---: | ---: |
| Search temporal pair | posterior, reconstruction, LPIPS | 17.85 GiB | 11.62 s after first step |
| Recovery spatial quad | all recovery losses, including auxiliary and both GAN branches | 17.45 GiB | 15.42 s |
| Recovery temporal quad | all recovery losses, including auxiliary and both GAN branches | 17.15 GiB | 20.75 s |

The spatial-quad valid RGB crop was 12x256x272; the temporal-quad crop was 63x80x96. Decoder inputs still use the native full tile/windows; overlap boundaries are cropped from supervision. These runs used accumulation 1 and omit DataLoader, full-video preview and checkpoint time. Production configs use accumulation 4, so the table is not production optimizer-update throughput or an upper bound on memory. First-time VGG downloading is excluded from the reported warm timings. Reports are under `/data/nvme6/gushiqiao/codes/latest/vae/test/benchmark_results/encoder_distillation_smoke/`.

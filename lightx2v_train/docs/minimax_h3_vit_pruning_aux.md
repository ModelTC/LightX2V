# H3 pruned decoder: reconstruction-aware recovery

This is the maintained 3,000-update stage-2 decoder recovery workflow. It requires the finished search export, not a search LoRA checkpoint. The fixed student is initialized from the selected original teacher blocks; all student parameters are then trained.

## Branches and gradients

```text
cached latent z
  +-- frozen full Teacher ----------------------> YT (detached target)
  |                   +-- matched feature HT
  +-- Student prefix --> HS
                          +-- Student suffix ---> YS (main reconstruction)
                          +-- frozen Teacher suffix --> YA (auxiliary reconstruction)
```

The main branch compares `YS` with the aligned source video, using Charbonnier reconstruction, VGG LPIPS, and the existing conditional 3D + frame-wise 2D GAN. Teacher RGB does not replace the source-video target. The existing adversarial generator objective is normalized correction-target MSE, not a DMD fake-score objective or a newly introduced hinge loss.

Ordinary feature MSE directly compares FP32 student and detached teacher video-token features. There is no softmax, KL, or feature normalization. The auxiliary branch instead passes the full student token sequence, including auxiliary/register tokens, through the corresponding original teacher suffix and output head. Teacher parameters are frozen, but this suffix forward must retain input gradients:

```text
grad_student = J_student_prefix^T J_teacher_suffix^T grad_YA
```

Only the student prefix traversed by the auxiliary branch receives its gradients. The main branch still trains the whole student. Teacher target computation uses `no_grad`; the differentiable suffix does not. This is ordinary first-order backpropagation and requires no explicit Jacobian or trainable fake-score model.

Both teacher targets and the differentiable suffix use FP32 computation (`teacher_autocast_dtype: fp32` disables the teacher's autocast). An FP16 suffix can underflow the small mean-reduced RGB gradients, especially during auxiliary warmup without a gradient scaler. Student parameters and residual tokens are FP32, while student matmuls retain BF16 autocast. The teacher therefore receives FP32 tokens without detaching the student graph. If GPU memory is insufficient, setting `teacher_autocast_dtype: bf16` changes both teacher paths together; avoid FP16 for the differentiable suffix. Teacher parameters remain frozen. FP32 can reduce numerical rounding error but does not guarantee better reconstruction quality, and its memory/runtime cost still needs GPU measurement.

The existing training entry point enables TF32 for FP32 CUDA matmuls/convolutions; this variant does not change that global setting or promise strict IEEE-FP32 arithmetic in every kernel. H3 VAE's default native SDPA supports FP32, but a forced low-precision FlashAttention backend may not. Attention backend changes can increase memory beyond a simple twofold estimate.

## Anchors and aligned decoding

`auxiliary_decoder.student_feature_indices: [8, 9, 10]` uses zero-based student block indices. The corresponding zero-based teacher indices are `[26, 29, 32]`, through `teacher_feature_indices`. In one-based notation these are student blocks 9/10/11 paired with teacher blocks 27/30/33. One auxiliary anchor is sampled per training sample and reused for its windows.

Teacher and student operate on the same H3 latent tiles/windows with matching local RoPE. Auxiliary predictions and teacher targets use the same cropping, valid-frame handling, and overlap blending before RGB losses. Differences must only span adjacent valid frames, not padding or unrelated windows. Existing main-branch blending remains unchanged. Teacher-suffix checkpointing and student-window checkpointing limit saved activations but do not eliminate the extra backward cost.

## Auxiliary loss

Let `E = YA - YT`, with `YT` detached and all RGB tensors aligned. With Charbonnier `rho(u) = sqrt(u^2 + epsilon^2) - epsilon`, the auxiliary branch is:

```text
Laux = mean rho(E)
     + 0.1 * mean(d in {H,W}) mean rho(delta_d E)
     + 0.1 * mean(d in {H,W}) mean rho(delta_T delta_d E)
Ltotal = Lmain + lambda_feature * MSE(HS, HT) + lambda_aux * Laux
```

The spatial term matches edges and texture. The mixed time-space term matches how those edges change between adjacent decoded frames; it does not force motion to zero and is not optical-flow alignment. Applying it to decoded RGB also covers the four frames inside each H3 tubelet. The reconstruction term remains necessary: mixed differences alone cannot reliably penalize constant blur or a common reconstruction error. No extra wavelet, FFT, second-order temporal, or optical-flow objective is added in this variant.

## Schedule

All boundaries below are optimizer-update indices, not micro-batches. The stage-2 run starts at zero independently of search.

| Updates | Reconstruction | Feature MSE | LPIPS | GAN | Auxiliary |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–599 | 1.0 | 0.01 | 0.1 | 0 | 0 |
| 600–2,999 | 1.0 | 0.005 | 0.2 | 0.5 | ramp to 0.1 |

The auxiliary weight ramps linearly over 200 updates at its activation (update 600). Its internal reconstruction/spatial/mixed weights stay at `1 / 0.1 / 0.1`. These are engineering settings, not measured optimal weights. GAN starts at update 600 with a 100-update discriminator warmup, then a 300-update generator ramp.

Learning rate starts at `5e-5`, with 200 warmup updates and cosine decay. There are four accumulation micro-batches per rank; physical batch is one sample. Four DDP ranks therefore process 16 sampled videos/window groups per optimizer update. LPIPS samples up to 64 available frames in batches of two. GAN uses up to 32 available frames with 256/384 spatial crops. A sampled group may contain fewer frames than those limits.

The four existing T2AV/I2AV/FL2AV/L2AV latent manifests are reused with their aligned source videos. No recaching is needed. `geometry_from_metadata: true` reads actual geometry; `num_frames: 362` is a fallback, not forced expansion of five-second clips.

Checkpoints are saved every 100 updates, retaining five. Three full-video previews plus PNG comparisons are saved every 100 updates. These preview examples are from a training manifest, so they are progress checks rather than held-out validation.

## Stage-2 command

Wait until search has finished and written both `kept_layers.json` and `minimax_h3_pruned_vae.safetensors` into its export directory. The stage-2 script fails clearly if either file is missing; it does not automatically start another search or training job.

```bash
cd /data/nvme6/gushiqiao/codes/latest/vae/LightX2V/lightx2v_train
GPU_LIST=0,1,2,3 bash scripts/run_minimax_h3_vit_prune_recover_aux_3k.sh
```

The script uses the existing H3 `local_diffusers/.venv/bin/python`. It defaults to the DDP search export, even if recovery uses FSDP:

```text
output_train/minimax_h3_vit_prune_search_4gpu_ddp/export/kept_layers.json
```

To use four-card FSDP recovery with that same searched architecture:

```bash
GPU_LIST=0,1,2,3 H3_VAE_PARALLEL=fsdp bash scripts/run_minimax_h3_vit_prune_recover_aux_3k.sh
```

Use `H3_VAE_PRUNE_SELECTION` to select a different export, for example one produced by FSDP search. `H3_VAE_PRUNE_SEARCH_OUTPUT` also overrides the default search-output directory in the launcher. `H3_VAE_PYTHON` overrides the interpreter.

Recovery uses `output_train/minimax_h3_vit_prune_recover_aux_3k_4gpu_ddp` or `..._fsdp` directories. Repeating the command resumes only that variant's latest completed checkpoint. To start a separate fresh experiment, provide a new `H3_VAE_PRUNE_RECOVER_AUX_3K_OUTPUT` directory. Legacy recovery-output variables have no effect on this launcher.

Only the trained student is used at inference. The auxiliary teacher suffix and losses are training-only and do not change student inference architecture or compute. Actual GPU memory and quality must still be measured; frozen teacher parameters do not make its input-gradient activations free. FSDP shards the student, while the frozen teacher remains replicated.

# H3 T2AV MoE recovery experiment

This experiment adapts the existing FFN-only Dense2MoE conversion to H3. It is
not an exact reproduction of the FLUX paper, a step-distilled model, or DMD2.

## Diagnosis of the previous run

- The cache contains 8,560 videos. With four examples per optimizer update,
  shared-1000 saw about 0.47 epochs and MoE-400 about 0.19 epochs. Freezing the
  shared expert after this short recovery phase is an unvalidated assumption.
- Shared-only videos already had severe structural degradation. The MoE stage
  froze these shared experts and trained only the router and remaining channels.
- The average output KD loss was 0.5871 at MoE steps 1–100 and 0.5380 at
  301–400. These are different random batches, not a controlled validation.
- A real checkpoint-400 load changed 46.39% of sampled expert values at BF16
  precision; mean relative RMS change was 0.00370. Checkpoint loading and
  updates work. This comparison is against the copied dense expert weights,
  not against checkpoint-100, which was removed by checkpoint retention.
- The old feature formula follows Dense2MoE equation 9. Its scalar value
  largely cancels the per-layer error through detached weighting. Its gradients
  are nonzero, but its magnitude is not an independent feature-quality metric.
  A large scalar alone does not prove that its gradients dominate.
- Shift-12 of uniform base time gives only 1/13 of samples below video sigma
  0.5. Detail recovery receives relatively few training examples.

These observations motivate a recovery experiment; they do not establish a
single causal explanation without ablations.

## Recovery recipe

Start from `h3_dense2moe_t2av_cached/checkpoints/moe-00000400`, including its
shared-1000 initialization. Keep the old run unchanged. Reset Adam and count
100,000 **new optimizer updates** in the recovery output directory.

- Train shared experts, routed experts, and routers jointly. Keep attention,
  AdaLN and all other dense parameters frozen. Keep 12 experts/top-2 and the
  25% shared width; inference architecture and activated width do not change.
- Loss: `output_KD + 0.1 * relative_feature + 0.1 * AV_flow + 0.01 * balance`.
- Relative feature loss is the mean across layers of
  `MSE(student, teacher) / mean(teacher**2)`, with a small denominator floor.
- AV flow target is `clean - noise` (H3 convention). Video and audio each use
  a separate mean loss so the much larger video token count does not dilute audio.
- Flow is a conservative auxiliary with CFG scale 1, as supported by DiffSynth.
  It does not implement inverse-CFG. H3's distilled guidance makes an aggressive
  pure-flow replacement inappropriate; teacher KD remains the main objective.
- Half the time sampling retains uniform base time; half samples uniform video
  sigma and inverts shift-12 to obtain base time. Both modalities always share
  that base time, with video/audio shifts fixed at 12/3. Expected fraction below
  video sigma 0.5 becomes approximately 28.85%.
- Shared/expert learning rate `5e-5`, router `1e-4`; 100-step linear warmup,
  cosine decay to 20%; gradient norm clipped to 1.0. Router has no weight decay.
- Clean/noise interpolation is FP32; student master parameters and Adam states
  are FP32, FSDP compute BF16, gradient reduction FP32, attention FA3.
- Original cached video sizes and their empirical frequency are preserved.

No GAN is included in this experiment. A discriminator is not a replacement
for verifying recovery KD; adding it now would introduce adversarial dynamics
and additional memory/compute without isolating the observed failure.

## Run and inspect

```bash
cd /data/nvme6/gushiqiao/codes/latest/LightX2V
export CUDA_VISIBLE_DEVICES=0,1,2,3
export H3_PYTHON=/data/nvme6/gushiqiao/models/MiniMax-H3/local_diffusers/.venv/bin/python
bash lightx2v_train/scripts/run_minimax_h3_dense2moe_recovery_fsdp4.sh
```

Run in the environment/container that owns these CUDA libraries. The launcher
uses a file lock to prevent duplicate runs. Re-running resumes the latest
complete checkpoint and restores Adam state; warmup/cosine uses the saved step.

Output: `/data/nvme6/gushiqiao/outputs/h3_dense2moe_t2av_recovery`.

```bash
tail -f /data/nvme6/gushiqiao/outputs/h3_dense2moe_t2av_recovery/logs/train_moe.log
```

- `recipe.json`: resolved configuration and prompt-file checksum.
- `metrics.jsonl`: losses, learning rates, pre-clipping global gradient norm,
  per-group local parameter-update RMS, timing, peak memory.
- `validation` entries: four fixed examples/noises at three joint sigma pairs,
  measured before training and every 100 steps. These examples remain in the
  training set: this is a fixed diagnostic, **not held-out validation**.
- Save at 1/100/400 and every 1,000 updates, retain one complete checkpoint.
- `diagnostics/probe.json`: dense/shared-1000/MoE-400 comparison on one cached
  sample at three noise levels, plus expert parameter-change statistics.

Evaluate actual multi-prompt videos as well as these losses. New scalar loss
values must not be compared directly against the previous differently scaled
objective. Passing engineering checks is not evidence of improved video quality.

## References

- [Dense2MoE equations 7–10](https://arxiv.org/html/2510.09094v1)
- [DiffSynth H3 audio/video flow loss and optional inverse-CFG](https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/diffusion/loss.py)
- [DiffSynth H3 training options](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/MiniMax-H3.md)

# Numerical Validation

Detailed guidance from the [skill entry](../SKILL.md). Read the sections relevant to the task.

## Source Parity Checklist

Before declaring parity, compare with upstream:

- model architecture and base family
- checkpoint used
- converted weight key mapping
- precision/dtype
- attention backend
- scheduler type
- timesteps/sigmas/shift
- CFG scale and CFG execution style
- explicit CFG branch-enable flags such as `enable_text_cfg`, `enable_audio_cfg`, and `enable_action_cfg`
- CFG parallel branch count and branch combination formula
- block/model offload behavior if the base family supports it
- sequence parallel behavior if the base family supports it
- action CFG if any
- prompt and negative prompt
- text embedding length and null prompt
- input image/video resize/crop/camera layout
- VAE latent scaling and temporal stride
- autoregressive chunk size and number of chunks
- KV cache window/eviction algorithm
- action channel mask, inverse channel map, and denormalization stats
- output FPS and save path
- output duration formula, especially for VAE temporal stride:
  - latent frames can differ from decoded video frames
  - decoded frames are often `(latent_frames - 1) * vae_stride[0] + 1`

Do not equate "runs without crashing" with "matches upstream."

## Numerical Alignment Workflow

When the model runs but output quality does not match upstream, debug with a staged parity ladder before changing high-level sampling logic. Save compact `.pt` fixtures for shared inputs and compare max/mean absolute differences at each boundary.

Recommended order:

1. Preprocessing and encoder inputs:
   - Compare raw inputs after every resize/crop/pad/normalize/layout step.
   - Compare prompt formatting, negative prompt handling, text/audio/image/video/control embeddings, VAE latents, and any task-specific conditioning tensors.
   - For image/video outputs, create contact sheets or frame grids when useful. A localized visual failure often points to input layout, conditioning, or a branch-specific block path rather than the whole sampler.

2. Scheduler and noise:
   - Compare every initial random tensor with the same seed and dtype.
   - Compare timesteps, sigmas, shift, CFG scale, and any step-skipping mask.
   - Compare one scheduler step numerically before running the whole denoise loop.

3. `pre_infer` boundary:
   - Compare DiT input tokens, timestep embeddings/modulation, conditioning context, positional encodings, modality-specific tokens, grid sizes, masks, and sequence lengths.
   - Treat exact or near-exact `pre_infer` parity as a strong signal that remaining errors live in transformer blocks or cache semantics.

4. Single-block transformer parity:
   - Run the first representative upstream block and the matching LightX2V block on the same saved `pre_infer` tensors.
   - Compare intermediate tensors in the upstream execution order: normalization, modulation, projections, positional encoding application, attention outputs, adapter/control branches if any, MLP/FFN outputs, and final residuals.
   - Compare every distinct execution branch the model can take, such as prefill vs decode, cache update vs no-update, conditional vs unconditional, first chunk vs later chunks, or modality-present vs modality-absent.
   - Keep shape comparisons explicit. Squeezing or broadcasting mistakes can create false differences, especially when upstream keeps a singleton batch dimension and LightX2V removes it.

5. Full-DiT parity:
   - After block 0 is close, compare every block output or at least final DiT predictions.
   - If full-DiT differs but single-block formulas are close, check the actual `infer_block()` return path, cache update path, and per-block weight loading path. A hand-expanded formula can hide a bug in the implemented method.
   - Ensure residual/skip-connection structure matches upstream exactly. Do not drop intermediate branch outputs that upstream adds back later.

6. End-to-end visual parity:
   - Run the smallest smoke inference and then a longer inference that exercises cache/offload/chunking/branching when the model has those features.
   - Generate sparse-frame contact sheets for image/video models, or equivalent compact artifacts for other modalities.
   - Check modality-specific quality, output length, FPS or sample rate, auxiliary output shape, and any task-specific postprocess result.

Prefer numeric evidence over intuition. If a symptom looks like "bad VAE" or "bad scheduler", prove or eliminate that boundary with saved tensors before rewriting unrelated code.

## Wan-Derived Porting Checklist

For a model similar to LingBot-VA, use this order:

1. Identify base:
   - Confirm whether it is Wan 2.1, Wan 2.2 dense, Wan 2.2 MoE, or another family.
   - Inspect upstream transformer config and model class names, but verify actual weight shapes.

2. Register model:
   - Follow [Registry And CLI Choices](../SKILL.md#registry-and-cli-choices): register the runner and add its module path to `RUNNER_MODULES`, which supplies lazy imports and CLI model choices.

3. Build runner:
   - Inherit nearest Wan runner.
   - Reuse parent `init_modules()` behavior where possible.
   - Bind only supported task encoders.
   - Initialize scheduler(s).
   - Initialize `KVCacheManager` if needed.
   - Keep input image paths and save paths in `InputInfo`.

4. Build model:
   - Inherit nearest Wan model if possible.
   - Add only model-specific `pre_infer`, `transformer_infer`, `post_infer`, and weight classes.
   - Move serial CFG logic into `model.infer`.
   - Set `scheduler.noise_pred`.

5. Build weights:
   - Inherit Wan weight modules if converted keys match.
   - Add extra layers only for real architectural additions.
   - Check missing-key errors against converted x2v checkpoint keys.

6. Build infer:
   - Keep DiT internals unbatched.
   - Remove trivial `_linear` wrappers.
   - Use inherited Wan `infer_ffn`, cross attention, modulation, offload, and feature-cache code when compatible.
   - Add model-specific self-attention only when KV cache or token layout differs.

7. Build offload and parallel paths:
   - Add block offload buffers and movement hooks for every custom block/weight phase.
   - Support model offload when the parent Wan model supports it.
   - Implement sequence parallel with existing Wan distributed helpers and gather/chunk custom full-token modules explicitly.
   - Implement CFG parallel with explicit branch lists and prediction gather/combine logic.
   - Ensure text/audio encoders provide all embeddings required by CFG-parallel ranks.
   - Reject unsupported phase offload clearly if the model-specific phase order has not been mapped.

8. Build scheduler:
   - Reuse existing scheduler only if solver semantics match.
   - Otherwise inherit `BaseScheduler`.
   - Put latent/action random initialization in scheduler prepare, not runner.
   - Use `step_pre -> model.infer -> step_post`.

9. Build KV cache:
   - Add cache pool under `common/kvcache` if algorithm is new.
   - Register scheme with `KVCacheManager`.
   - Create caches in runner, not infer.
   - Use cache names for cond/uncond histories only when needed.

10. Build config/script:
   - Keep stable profile fields in JSON.
   - Keep `model_cls`, `task`, prompt, input paths, output path, and CUDA device in bash/CLI.
   - Use LightX2V field names, not upstream aliases.
   - Reuse existing path variables such as `dit_original_ckpt`, `dit_quantized_ckpt`, `adapter_model_path`, and `audio_encoder_path`.
   - Do not add a second input JSON file for values already covered by config or bash args.
   - Do not hard-code local model roots or upstream source roots in Python.

11. Verify:
    - Compile Python.
    - Validate JSON and bash syntax.
    - Run smallest available inference.
    - Compare preprocessing, scheduler, `pre_infer`, at least block 0, and final DiT predictions against upstream.
    - Smoke-test block/model offload, sequence parallel, and CFG parallel when hardware is available.
    - Compare source output length, auxiliary output shapes when applicable, compact visual artifacts, and qualitative result.

## Common Pitfalls

- Registering a runner without adding its module path to `RUNNER_MODULES`.
- Loading `transformer/config.json` assumed but `set_config.py` never merges it for this `model_cls`.
- Keeping an upstream batch dimension in Wan-style DiT internals that expect unbatched tokens; other families must retain their own tensor layout.
- Using batched CFG via `torch.cat` in a Wan-style unbatched DiT instead of its serial or supported CFG-parallel path.
- Letting a model "support" only the local serial path while skipping base-family block offload, model offload, sequence parallel, or CFG parallel.
- Controlling CFG branch count by `scale == 1` instead of explicit `enable_*_cfg` flags.
- Assuming a 2-branch Wan CFG-parallel path is correct for a model that needs 3 branches such as cond/drop-text/uncond.
- Assuming Wan text encoder rank splitting produces every context needed by custom CFG-parallel branches.
- Forgetting offload buffers or CPU lazy buffers for custom block classes.
- Reusing Wan phase offload when the new model has a different phase order.
- Chunking tensors for sequence parallel and then reshaping them as if each rank still owns the full spatial/temporal token grid.
- Reimplementing T5/VAE that already exists.
- Calling diffusers/transformers model modules directly in native DiT inference.
- Importing a third-party pipeline/model and hiding it behind a LightX2V runner instead of rebuilding the structure with LightX2V ops.
- Mixing `float32` and `bfloat16` in norm/MM without deliberate casts.
- Missing one residual/skip add in translated transformer blocks.
- Trusting a manually expanded debug formula while the real `infer_block()` return path still differs.
- Comparing tensors with incompatible singleton dimensions and accidentally broadcasting a fake difference.
- Checking only one execution branch while prefill/decode, cache/no-cache, cond/uncond, or modality-present/modality-absent branches differ.
- Repeated `.to(device).to(dtype)` in hot loops.
- Using `.item()`, CPU tensor comparisons, or `argmin` in scheduler hot path.
- Putting image paths and save defaults in config instead of `InputInfo`.
- Hard-coding `train_out`.
- Hard-coding adapter, audio encoder, model root, or upstream source-root paths in Python.
- Creating new model-specific path fields when `dit_original_ckpt`, `dit_quantized_ckpt`, `adapter_model_path`, or `audio_encoder_path` already fit.
- Adding a separate input JSON file for prompt/audio/image/save values that belong in bash/CLI or config.
- Creating model-private KV cache instead of using `KVCacheManager`.
- Duplicating base Wan/Qwen infer code when inheritance would work.
- Treating upstream config names as semantics; inspect the actual fields.
- Keeping `image_path`, `save_result_path`, `save_action_path`, or prompt defaults in config after the runner already receives `InputInfo`.
- Creating cond/uncond caches for every branch when a branch has explicit CFG disabled.
- Running uncond action inference when `enable_action_cfg` is false.
- Letting scheduler search timesteps by tensor comparison every step instead of using `step_index`.

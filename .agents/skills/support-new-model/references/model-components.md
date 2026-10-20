# Model Components

Detailed guidance from the [skill entry](../SKILL.md). Read the sections relevant to the task.

## Decide The Base Family

First answer: what architecture is this model based on?

- Wan 2.1 / Wan 2.2 dense / Wan 2.2 MoE
- Qwen Image / Flux / Hunyuan / LTX / another existing LightX2V family
- Image model using diffusers-native conventions
- Video/action/world model with a DiT derived from Wan or similar

If it is based on an existing family, inherit and reuse:

- runner from nearest runner (`Wan22DenseRunner`, `DefaultRunner`, etc.)
- model from nearest model class
- weight modules from nearest weight class
- infer phases from nearest infer class
- T5/text encoder and VAE wrappers when compatible
- scheduler if solver semantics match
- KV cache manager/base classes when compatible

Only add new code for actual architectural differences.

For Wan-derived models:

- Put the new runner under `models/runners/wan/`.
- Put the new model under `models/networks/wan/` if it reuses Wan T5/VAE/DiT conventions.
- Put model-specific infer code under `models/networks/wan/infer/<model>/`.
- Put model-specific weights under `models/networks/wan/weights/<model>/` only when Wan weights cannot be inherited.
- Reuse Wan T5 and VAE unless the upstream model has a genuinely different text encoder or autoencoder.
- If the model is based on Wan 2.2 but adds a few layers, inherit Wan weights and add only those layers.

## Weight Strategy

LightX2V native video model inference generally expects converted x2v-compatible weights.

Rules:

- For video/DiT ports, do **not** load a full upstream diffusers/transformers model at runtime.
- Do **not** call `DiffusionPipeline.from_pretrained`, `WanTransformer3DModel.from_pretrained`, or equivalent upstream model classes as the inference implementation.
- If the upstream checkpoint is diffusers format, ask/expect the user to convert it first, or add a conversion script if explicitly requested.
- If the user says weights are already converted to x2v format, trust that and make the loader match x2v keys instead of falling back to diffusers modules.
- Use conversion scripts and existing weight mapping rules as reference.
- The model config may live under `model_path/config.json`, `model_path/transformer/config.json`, `low_noise_model/config.json`, etc. Verify `set_config.py` actually loads that path for the new `model_cls`.
- If LightX2V does not load the upstream `transformer/config.json` for this model class, either add a model-class-specific merge or keep required structure fields in `configs/<family>/*.json`.
- For image models, LightX2V may already support diffusers-style loading; inspect the closest image runner before forcing x2v conversion.

Weight implementation:

- Put weight definitions under `lightx2v/models/networks/<family>/weights/...`.
- Use LightX2V common ops: MM, norm, attention, conv, embedding.
- Define tensors with the names produced by the converter.
- If the new model is mostly Wan, inherit Wan weight classes and add only extra layers.
- Keep raw checkpoint dictionaries out of infer code.
- Avoid live upstream `torch.nn.Module` objects in inference.
- If an upstream module is needed as reference, read its architecture and re-express it as LightX2V weights/infer ops.
- Do not add ad hoc key fallbacks inside infer. Fix the converter, the weight mapping, or the weight class.
- When a missing key error appears, inspect converted checkpoint keys before changing code. A diffusers key such as `patch_embedding.weight` may not be the x2v key after conversion.

## DiT Tensor Layout

Wan-style native DiTs use token tensors without a batch dimension. Follow the selected family's actual weight/infer/attention contract; this is not a repository-wide ban on batched tensors (for example, Bagel and SeedVR have different layouts):

```text
x:        [seq, dim]
q/k/v:    [seq, num_heads, head_dim]
context:  [text_seq, dim]
linear:   2D matmul by default
```

Avoid:

- `torch.cat([x] * 2)` for CFG
- `[B, L, C]` hidden states in transformer internals
- batch-shaped modulation unless the existing base class requires it
- helper APIs that fake `batch_size=1` unless the lower-level kernel needs prefix sums
- helper functions such as `_ensure_single_batch_inputs(...)` in model infer
- pre-infer returning `[1, L, C]` when transformer infer expects `[L, C]`

If an attention varlen API requires `cu_seqlens`, pass single-sequence prefix sums:

```python
cu_seqlens_q = torch.tensor([0, query_len], device=q.device, dtype=torch.int32)
cu_seqlens_kv = torch.tensor([0, kv_len], device=k.device, dtype=torch.int32)
```

Do not introduce a `batch_size` concept just to produce `[0, len]`.

Pre-infer should flatten/patchify into sequence tensors and remove only the outer input batch where necessary:

```text
input latent/action: [1, C, F, H, W]
DiT tokens:          [seq, dim]
text context:        [text_seq, dim]
timestep modulation: [seq, ...]
rotary ids/emb:      [seq, ...]
```

Keep the public latent/action tensors compatible with VAE/scheduler if they use `[1, C, ...]`, but the DiT internals should be unbatched.

## Input Encoders

Put input encoding under `lightx2v/models/input_encoders/...` if it is a reusable encoder.

Runner should bind input encoder methods in `init_modules()` like existing runners:

```python
if self.config["task"] == "i2v":
    self.run_input_encoder = self._run_input_encoder_local_i2v
elif self.config["task"] == "i2va":
    self.run_input_encoder = self._run_input_encoder_local_i2va
```

Rules:

- Reuse existing T5/UMT5/Qwen/etc. encoders when possible.
- If the upstream encoder already exists in LightX2V, do not reimplement it.
- Prompt and negative prompt are per-run CLI/script inputs, not model JSON defaults.
- Image/video/audio paths should come from `InputInfo` and CLI/script, not hard-coded train output paths.
- Do not read `input_img_path` or similar config fallbacks if the LightX2V CLI already has `--image_path`.
- Avoid extra private `encode_prompt` methods when `run_text_encoder` or existing runner methods already do the job.
- Bind task-specific encoder methods in `init_modules()` after `super().init_modules()`, following the existing runner style.

## VAE And Decoding

Reuse existing VAE wrappers when compatible.

Rules:

- Do not implement a new VAE for a model if Wan/Qwen/etc. VAE wrapper already loads it.
- For Wan-based video models, reuse Wan VAE/T5 runner logic.
- VAE weights can live under `model_path/vae` or config override paths such as `vae_original_ckpt`; inspect existing conventions.
- If T5/VAE weights are placed under `model_path`, load them through existing Wan-style paths or config overrides such as `t5_original_ckpt`. Do not require diffusers `from_pretrained` just for T5/VAE if LightX2V has a native loader.
- Keep decode/save in runner postprocess, not inside transformer infer.
- Save to `input_info.save_result_path`; do not add fallback `train_out` behavior unless the existing runner family requires it.
- Derived action outputs can use the video output path suffix, for example `.actions.npy`, but keep the user-facing root `save_result_path`.
- Avoid extra wrapper helpers such as `_vae_encode_batch(...)` unless they express real reusable behavior; call the existing VAE encoder/decoder API directly when possible.

## Model Layout

For a native DiT model, use this structure:

```text
lightx2v/models/networks/<family>/
  model.py
  weights/
    pre_weights.py
    transformer_weights.py
    post_weights.py
  infer/
    pre_infer.py
    transformer_infer.py
    post_infer.py
```

Model responsibilities:

- own `pre_weight`, `transformer_weights`, `post_weight`
- own `pre_infer`, `transformer_infer`, `post_infer`
- implement `infer(inputs)` and `_infer_cond_uncond(...)` when CFG is needed
- set `scheduler.noise_pred`
- manage model-level offload if inherited family already does
- support inherited block/model offload, sequence parallel, and CFG parallel paths after the serial inference path is correct
- expose cache helpers only if they are generic and aligned with manager APIs

Keep scheduler stepping out of model. Keep weight loading out of runner.

For CFG-capable models, keep the cond/uncond DiT execution in `model.py`, not in runner:

```python
def infer(self, inputs):
    if inputs.get("enable_cfg", self.config.get("enable_cfg", False)):
        noise_pred_cond = self._infer_cond_uncond(inputs, infer_condition=True)
        noise_pred_uncond = self._infer_cond_uncond(inputs, infer_condition=False)
        noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
    else:
        noise_pred = self._infer_cond_uncond(inputs, infer_condition=True)
    self.scheduler.noise_pred = noise_pred
```

Runner should build inputs and call `model.infer(...)`; it should not run `pre_infer`, transformer blocks, or CFG branches itself.

## Infer Implementation

Infer code should consume LightX2V weight wrappers directly:

```python
out = phase.linear.apply(x)
```

Avoid local helper wrappers like `_linear(...)` unless they add real behavior. Prefer inherited methods from the base family if the behavior is identical.

Do not import or instantiate upstream model blocks for forward execution. If the upstream has `nn.Module` blocks, translate their tensor operations into LightX2V `infer` code over LightX2V `weights`.

Rules:

- Directly call `mm.apply`, norm `.apply`, attention `.apply`.
- Keep tensors on the intended device/dtype; avoid repeated `.float().to(dtype).to(device)` churn.
- Normalize dtype once around numerically sensitive ops.
- Use `GET_DTYPE()`, `GET_SENSITIVE_DTYPE()`, and `AI_DEVICE`; do not hard-code CUDA.
- Reuse existing `infer_ffn`, `infer_cross_attn`, modulation helpers, offload hooks, and feature-cache hooks when compatible.
- Add only model-specific self-attn/cross-attn blocks.
- Remove duplicated code if the base class already implements it.
- If a local `_linear(...)` only wraps `module.apply(x)`, delete it.
- If a local `infer_ffn`, `infer_cross_attn`, or modulation helper matches the base class, inherit the base class method.
- Keep layer norm dtype deliberate. A common failure is passing bfloat16 weights to a float input or vice versa; inspect the norm weight wrapper instead of scattering casts everywhere.
- If an upstream implementation stores per-block phases, keep LightX2V phase order compatible with existing Wan infer code.
- If feature caching/offload exists in the base family, preserve the base hooks rather than bypassing them in custom infer.

## CFG

For Wan-style unbatched DiTs, the serial CFG pattern is:

```python
noise_pred_cond = self._infer_cond_uncond(inputs, infer_condition=True)
noise_pred_uncond = self._infer_cond_uncond(inputs, infer_condition=False)
noise_pred = noise_pred_uncond + scale * (noise_pred_cond - noise_pred_uncond)
```

Rules:

- Do not concatenate cond/uncond along batch for DiT.
- Keep cond and uncond prompt embeddings separate in encoder output.
- For uncond pass, create a shallow copy of inputs and replace `text_emb` with `negative_text_emb`; do not mutate the caller's dict permanently.
- If the model has multiple modalities, separate CFG flags when useful:
  - `enable_cfg` for video/image branch
  - `enable_action_cfg` for action branch
- If a modality has optional CFG, use explicit booleans such as `enable_text_cfg`, `enable_audio_cfg`, or `enable_action_cfg` to choose branches. Do not use `scale == 1` as the only signal for whether a branch should run.
- If a modality has `enable_*_cfg == false`, do not run its uncond/drop branch.
- If any branch uses CFG and the model also uses KV cache, create cond/uncond caches only for branches that need separate histories; reuse the cond cache for non-CFG branches when that preserves history and avoids duplicate compute.
- `cfg_parallel` should only be enabled if the existing distributed CFG path is implemented and tested.
- Store branch-local controls in inputs when useful:
  - `enable_cfg`
  - `enable_text_cfg`
  - `enable_audio_cfg`
  - `guide_scale`
  - `action_mode`
  - `cache_name`
  - `update_cache`

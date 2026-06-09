---
name: lightx2v-native-model-porting
description: Use this skill when adding native LightX2V support for a new model or task: understand an upstream inference repo, map it onto LightX2V runner/model/weight/infer/scheduler/input-encoder/VAE conventions, convert or load weights, add configs and Wan-style scripts without hard-coded paths, implement CFG/KV cache/block-model offload/SP/CFG-parallel without batch dimensions, and verify output parity with the upstream pipeline.
---

# LightX2V Native Model Porting

Default LightX2V root: `/data/nvme4/gushiqiao/new/LightX2V`.

Goal: build a **native LightX2V runner** whose output matches the upstream model inference. Do not wrap the upstream repo at runtime unless the user explicitly asks for a temporary bridge.

Hard rule: do **not** depend on diffusers/transformers/third-party model or pipeline classes to execute the core model. Rebuild the model structure with LightX2V's data structures and ops (`weights`, `infer`, `common/ops/mm`, norm, attention, conv, scheduler, KV cache), or reuse an existing LightX2V implementation that already does so.

## Core Principle

Always map the upstream pipeline onto LightX2V's existing architecture:

```text
CLI/config/script
  -> runner
  -> input encoder(s)
  -> scheduler.step_pre()
  -> model.infer()
       -> pre_infer
       -> transformer_infer
       -> post_infer
       -> weights/common ops
  -> scheduler.step_post()
  -> VAE decode/postprocess/save
```

Prefer reusing existing LightX2V families (`wan`, `qwen_image`, `hunyuan_video`, `ltx2`, etc.) over creating a parallel implementation. New code should look like a small specialization of an existing family, not a copied upstream pipeline.

For a new DiT/video model, the desired ownership chain is strict:

```text
new runner
  -> creates/owns scheduler(s)
  -> loads new model
new model
  -> owns new/reused weights
  -> owns new/reused infer modules
weights
  -> describe x2v tensor layout using common ops
infer
  -> executes DiT math with those weights
```

Do not put model path in runner. Do not put checkpoint mapping in infer. Do not call the upstream repo from runner/model to get a result. The result should come from LightX2V-native modules.

Treat base inference as the first milestone, not the finish line. When a model family supports block/model offload, sequence parallel, or CFG parallel, the new model should support the corresponding LightX2V modes too unless there is a specific architectural blocker documented in the final report.

Native means:

- model parameters are loaded into LightX2V weight wrappers
- linear projections use LightX2V MM ops, not upstream `nn.Linear` modules
- normalization/attention/conv use LightX2V common ops or existing family wrappers
- scheduler state is managed by LightX2V scheduler classes
- third-party libraries may be used for tokenizers, file IO, or conversion scripts when necessary, but not as the runtime DiT/pipeline implementation

## First Inspection

Before editing, inspect the current repo and the upstream model:

- LightX2V entrypoints:
  - `lightx2v/infer.py`
  - `lightx2v/utils/set_config.py`
  - `lightx2v/utils/input_info.py`
  - `lightx2v/utils/registry_factory.py`
  - `lightx2v/models/runners/default_runner.py`
  - nearest family runner, for example `lightx2v/models/runners/wan/wan_runner.py`
- Model implementation:
  - `lightx2v/models/networks/<family>/model.py`
  - `lightx2v/models/networks/<family>/weights/`
  - `lightx2v/models/networks/<family>/infer/`
  - `lightx2v/models/schedulers/<family>/`
  - `lightx2v/common/ops/`
  - `lightx2v/common/kvcache/`
- Upstream repo:
  - README model list and quickstart
  - config registry and launch scripts
  - model definition
  - weight loading path and checkpoint format
  - scheduler/solver code
  - CFG logic
  - KV cache logic
  - input preprocessing and output postprocessing

Use `rg` first. Do not guess from filenames when config files or README disagree.

When upstream has several config names or model names, cross-check:

- README model table and release notes
- config registry, for example `VA_CONFIGS`
- per-config camera keys, action channel ids, normalization stats, resolution, and chunk settings
- local model directory contents
- actual launch script defaults

Treat edited local paths as weak evidence. Dataset/domain fields and README model descriptions are stronger evidence for deciding which weight belongs to which task.

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

## No Batch Dimension In DiT Inference

This is important.

LightX2V DiT inference should run on token tensors without a batch dimension:

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

Default LightX2V CFG should be serial, not batched:

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

## Offload And Parallel Support

After the single-device serial path matches upstream, add the same execution modes that the base family supports. For Wan-derived models, this normally means block/model offload, sequence parallel (`sp`/`seq_parallel`), and CFG parallel (`cfg_parallel`).

Rules:

- Reuse the base family offload and distributed machinery before adding model-private code.
- For Wan-derived transformer infer, prefer inheriting or composing with existing Wan offload infer classes such as `WanOffloadTransformerInfer` when the phase order is compatible.
- Implement block offload for every custom weight/block phase introduced by the new model:
  - create matching CUDA block buffers such as `offload_block_cuda_buffers`
  - create lazy CPU-side buffers when the base weight class uses them
  - move non-block weights through the existing `non_block_weights_to_cuda/cpu` hooks
- Support model-level offload when the parent model already provides `to_cuda`/`to_cpu` style movement. Do not special-case devices in runner hot loops.
- Reject phase offload clearly if the new model has a custom phase order and phase offload has not been mapped yet. Do not silently reuse Wan phase offload for a different phase graph.
- For sequence parallel:
  - reuse existing process groups and helpers such as `seq_p_group`, sequence chunking, all-gather, and parallel attention ops
  - use the base self-attention parallel kernel when tensor layout matches
  - if a custom module needs the full token grid, gather full tokens for that module and chunk the result back to the local rank
  - for custom reference maps or multi-human masks, gather `q/k` before computing a full attention map when local chunks are insufficient
- For CFG parallel:
  - keep CFG branch orchestration in `model.infer`, not runner
  - keep DiT internals unbatched; distribute branches across ranks instead of concatenating a batch dimension
  - build an explicit branch list, for example `cond`, `drop_text`, `drop_audio`, `uncond`
  - support the actual branch count needed by the model, including 2-branch and 3-branch CFG
  - gather branch predictions and combine them with the same formula as serial CFG
  - make text/audio encoders produce every embedding needed by each rank; do not assume Wan's default rank0 prompt / rank1 negative split is enough for multi-branch CFG
- Keep serial, offload, SP, CFG parallel, and combined modes numerically aligned with the same seed/config whenever practical.

## Scheduler

Put scheduler code in:

```text
lightx2v/models/schedulers/<family>/<model>/scheduler.py
```

First check if an existing scheduler matches exactly:

- Wan UniPC scheduler
- Wan audio Euler scheduler
- Qwen/Hunyuan/WorldPlay flow schedulers
- self-forcing scheduler
- simple flow-match Euler scheduler

If not exact, inherit `BaseScheduler` and keep the model-specific scheduler thin.

Runner denoise loop should look like:

```python
with ProfilingContext4DebugL1("step_pre"):
    self.model.scheduler.step_pre(step_index=step_index)

with ProfilingContext4DebugL1("infer_main"):
    self.model.infer(self.inputs)

with ProfilingContext4DebugL1("step_post"):
    self.model.scheduler.step_post()
```

Scheduler responsibilities:

- prepare initial latents/noise
- set timesteps/sigmas
- own `latents` during the denoise loop
- store current timestep for model input
- step latents after model predicts noise
- preserve conditioning latent prefixes when needed
- clear loop-local state at end
- expose a `prepare_loop(...)` or existing-family equivalent for new latent/action loops

If the upstream runner creates random latents/actions directly, move that logic into scheduler preparation:

```python
scheduler.prepare_loop(
    infer_steps=config["infer_steps"],
    latent_shape=latent_shape,
    seed=input_info.seed,
    dtype=GET_DTYPE(),
    cond_latent=cond_latent,
)
```

Runner may bind a per-step input builder:

```python
scheduler.bind_step_inputs(self.inputs, self._build_video_step_inputs)
scheduler.bind_noise_pred_processor(self._postprocess_video_noise_pred)
```

Then `step_pre()` can update `self.inputs` for the current timestep before `model.infer(...)`.

Avoid per-step GPU-to-CPU synchronization:

- Store `self.step_index = int(step_index)` in `step_pre`.
- In `step`, use `self.step_index` directly for `sigmas`.
- Avoid `torch.argmin(...).item()` or checking tensor values on CPU in the hot loop.

## Runner

Runner should follow the closest existing family style.

Responsibilities:

- register with `@RUNNER_REGISTER("<model_cls>")`
- initialize/load modules
- initialize scheduler(s)
- bind `run_input_encoder`
- prepare per-run state in `init_run`
- run the denoise loop
- decode/save outputs
- clear caches and modules in `end_run`

Rules:

- Inherit `DefaultRunner` or nearest family runner.
- Do not put transformer math in runner.
- Do not put scheduler stepping inside model unless existing family requires it.
- Use `self.model.set_scheduler(self.scheduler)` before infer loop.
- For multi-stage models, it is fine to have multiple schedulers, for example video scheduler and action scheduler, but keep both using the same `step_pre -> infer -> step_post` pattern.
- Avoid fallback output paths like `train_out`; require or use `input_info.save_result_path`.
- Input paths such as `image_path` should come from `input_info`, not config defaults.
- `init_modules()` should follow the parent style: load modules, set scheduler on model, bind `run_input_encoder`, lock config, optionally compile.
- `init_run()` should prepare per-run state, KV cache manager, prompt embeds from `self.inputs`, action masks/stats, and AR dimensions.
- `run_segment()` or equivalent should only orchestrate scheduler loops and postprocess outputs.
- `end_run()` should clear schedulers, caches, `self.inputs`, lazy-loaded model state, and CUDA cache.

## KV Cache

For KV-cache models, inspect existing cache infrastructure first:

- `lightx2v/common/kvcache/base.py`
- `lightx2v/common/kvcache/manager.py`
- self-forcing runner and infer:
  - `lightx2v/models/runners/wan/wan_sf_runner.py`
  - `lightx2v/models/networks/wan/infer/self_forcing/transformer_infer.py`

Rules:

- Prefer adding a new `BaseKVCachePool` subclass under `common/kvcache/` rather than private cache classes inside the model.
- Register the cache scheme with `KVCacheManager`.
- Reuse `KVCacheManager` creation from runner.
- Create caches in runner through `KVCacheManager`, not inside transformer infer.
- Avoid model-specific methods like `create_<model>_self_attn_cache(...)` if the generic manager can create it with `kv_cache_scheme` and `kv_size`.
- Use scheme names that describe the algorithm, for example `rolling` or `fifo`.
- Align APIs with existing cache classes:
  - `store_kv`
  - `k_cache`
  - `v_cache`
  - `reset`
  - optional `clear_pred` / `restore` only if algorithm needs temporary speculative writes.
- Keep cache names meaningful:
  - `pos`
  - `pos_cond`
  - `pos_uncond`
  - other branch names when multiple histories are needed.
- Put autoregressive/cache parameters under `ar_config` in JSON.
- Do not keep a `batch_size` property unless the cache algorithm truly supports multiple batches. LightX2V DiT cache should normally be single sequence.
- Prefer `reset()` for clearing a cache. Add `clear_pred()` only when speculative/predicted entries need to be dropped without clearing committed history.
- Add `restore(layer_id, slots)` only when non-committed temporary writes are stored during a no-update pass.
- FIFO cache means evicting the oldest committed ids when full. Rolling cache means sliding/rotating a fixed attention window. Name the class/scheme after the actual algorithm.

Example `ar_config` fields:

```json
{
  "ar_config": {
    "num_frame_per_chunk": 2,
    "num_action_per_frame": 16,
    "num_chunks": 10,
    "kv_cache_scheme": "fifo",
    "step_kv_cache": false,
    "local_attn_size": 72
  }
}
```

## Config Conventions

Configs should contain model/run defaults that are stable for a profile. Scripts should contain per-run values.

Do not hard-code local model roots, upstream source roots, adapter paths, audio paths, or sample inputs inside Python. Prefer existing CLI/config fields and set machine-specific values in the bash script or the selected config profile.

Do not add extra input JSON dependency files for a new model unless the existing LightX2V workflow already requires them. Keep structured model defaults in `configs/<family>/*.json`, and put per-run arguments in bash/CLI. A script can pass a JSON-valued config path, but it should not require a second model-private `input_json` just to provide prompt/audio/image paths that LightX2V already accepts.

Do not add a model-local `_apply_default_config()` that silently fills many fields. Put required stable values in JSON, or inherit existing global defaults intentionally. If a field is required for correctness, fail early with a clear error instead of hiding it behind a private defaults function.

Prefer LightX2V naming:

- `target_height`, `target_width`
- `infer_steps`, `action_infer_steps`
- `sample_shift`, `action_sample_shift`
- `sample_guide_scale`, `action_sample_guide_scale`
- `enable_cfg`, `enable_action_cfg`
- `target_fps`
- `ar_config.num_frame_per_chunk`
- `ar_config.num_action_per_frame`
- `ar_config.num_chunks`
- `ar_config.local_attn_size`
- `ar_config.kv_cache_scheme`
- `ar_config.step_kv_cache`

Reuse existing path fields before inventing new ones:

- `model_path`
- `dit_original_ckpt`
- `dit_quantized_ckpt`
- `adapter_model_path`
- `audio_encoder_path`
- `vae_original_ckpt`
- `t5_original_ckpt`

Do not create parallel model-specific path fields such as `<model>_model_root`, `<model>_source_root`, `<model>_single_ckpt`, or `<model>_multi_ckpt` if `adapter_model_path`, `audio_encoder_path`, or the existing DiT/VAE/T5 checkpoint fields already express the same thing. If a model has multiple variants, pick the variant in the config or script by assigning the existing field to the desired checkpoint.

Avoid old/upstream names in JSON:

- `height`, `width`
- `num_inference_steps`
- `action_num_inference_steps`
- `snr_shift`
- `action_snr_shift`
- `guidance_scale`
- `action_guidance_scale`
- `frame_chunk_size`
- `action_per_frame`
- `num_chunks_to_infer`
- `attn_window`

Do not put these in model JSON when scripts/CLI should own them:

- `model_cls`
- `task`
- `prompt`
- `negative_prompt`
- `sample_neg_prompt` when it duplicates `negative_prompt`
- input paths
- output paths
- CUDA device selection

Use `config_json` for stable profile config and bash for:

- `--model_cls`
- `--task`
- `--model_path`
- `--prompt`
- `--negative_prompt`
- `--image_path` / `--video_path` / `--audio_path`
- `--save_result_path`

When choosing config vs bash:

- Put architecture and stable profile fields in JSON: resolution, fps, attention types, latent/audio dimensions, scheduler shift, offload/parallel capability flags, and model variant defaults.
- Put run-local fields in bash/CLI: prompt, negative prompt, image/video/audio path, save path, visible devices, model class, task, and checkpoint path overrides that vary by machine or experiment.
- Before adding a new argparse/config name, search for an existing equivalent and reuse it. For example, use `adapter_model_path` for model adapters and `audio_encoder_path` for audio encoders instead of adding model-specific aliases.

For profiles copied from upstream names, keep filename compatibility if useful, but keep internal LightX2V semantics clear. Example: upstream may call a config `robotwin_i2av` while its actual `infer_mode` is `i2va`; in LightX2V the script can use `robotwin_i2av.json` while still passing `--task i2va`.

When deciding whether a field can be removed from JSON because a model config exists under `model_path`, verify `set_config.py` loads that config for this `model_cls`. If it does not, either add the merge or keep the field. Do not assume `transformer/config.json` is loaded globally.

## Bash Script Conventions

Follow existing LightX2V scripts:

```bash
#!/bin/bash

lightx2v_path=/data/nvme4/gushiqiao/new/LightX2V
model_path=/path/to/model

export CUDA_VISIBLE_DEVICES=0

source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls my_model \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/my_model/profile.json \
--prompt "..." \
--negative_prompt "" \
--image_path /path/to/input.png \
--save_result_path ${lightx2v_path}/save_results/output.mp4
```

Rules:

- Always source `scripts/base/base.sh`.
- Use absolute config paths based on `${lightx2v_path}`.
- Keep model path and device near the top.
- Keep adapter/audio encoder/checkpoint overrides near the top and pass them through existing variable names such as `adapter_model_path` and `audio_encoder_path`.
- Keep per-run prompt/input/output in bash, not JSON.
- Validate scripts with `bash -n`.
- Follow existing Wan-style script formatting when adding Wan-derived models.
- Keep action output derived from `save_result_path` when the runner does that; do not add a separate train-output root unless explicitly required.

## Registry And CLI Choices

When adding `model_cls`:

- register runner with `RUNNER_REGISTER`
- ensure the runner module is imported wherever registry population happens
- update argparse `choices` only if the repo uses explicit choices for `--model_cls`
- update task choice only if a new `--task` string is truly needed

If the upstream config name says `i2av` but the actual infer mode is image-to-video-action, choose one LightX2V task name and keep it consistent. Prefer semantic names in code (`i2va`) and use config filenames only for compatibility if needed.

When an error says `argument --model_cls: invalid choice`, fix registry/import/argparse choices first. Do not rename the model to an unrelated existing `model_cls` just to pass argument parsing.

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

## Wan-Derived Porting Checklist

For a model similar to LingBot-VA, use this order:

1. Identify base:
   - Confirm whether it is Wan 2.1, Wan 2.2 dense, Wan 2.2 MoE, or another family.
   - Inspect upstream transformer config and model class names, but verify actual weight shapes.

2. Register model:
   - Add `RUNNER_REGISTER("<model_cls>")`.
   - Ensure the runner file is imported by LightX2V startup.
   - Add argparse choices only if the repo uses explicit choices.

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
    - Smoke-test block/model offload, sequence parallel, and CFG parallel when hardware is available.
    - Compare source output length, action shape, and qualitative result.

## Common Pitfalls

- Registering a runner but forgetting argparse choices or module import.
- Loading `transformer/config.json` assumed but `set_config.py` never merges it for this `model_cls`.
- Keeping upstream batch dimension in DiT internals.
- Batched CFG via `torch.cat` instead of serial CFG.
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

## Verification

Fast checks after every change:

```bash
cd <LightX2V repo root>
python3 -m py_compile <changed python files>
bash -n <changed scripts>
python3 -m json.tool <changed configs>
python -m lightx2v.infer --help
```

Run inference when weights and inputs are available. Confirm:

- output video/image exists
- action file exists if applicable
- output length matches `num_chunks`, chunk size, VAE stride, and FPS
- no unwanted batch dimension appears in debug shapes
- no unexpected diffusers/transformers runtime model dependency
- results are visually/numerically close to upstream with the same seed/config
- block/model offload mode completes and produces the same shape/duration as local mode
- CFG parallel returns the same branch-combined prediction shape and comparable output as serial CFG
- sequence parallel preserves token counts and reconstructs full-output shapes after all-gather/chunk operations

Report skipped checks clearly.

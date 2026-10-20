# Scheduler And Cache

Detailed guidance from the [skill entry](../SKILL.md). Read the sections relevant to the task.

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

If the upstream runner creates random latents/actions directly, move that logic into scheduler preparation.

The following example uses the [LingBot-VA scheduler](../../../../lightx2v/models/schedulers/wan/lingbot_va/scheduler.py). Its `prepare_loop`, `bind_step_inputs` and `bind_noise_pred_processor` hooks are not `BaseScheduler` APIs. For other families, follow the selected scheduler's preparation interface and parameter contract.

```python
from lightx2v_platform.base.global_var import AI_DEVICE

scheduler.prepare_loop(
    infer_steps=config["infer_steps"],
    device=AI_DEVICE,
    latent_shape=latent_shape,
    seed=input_info.seed,
    dtype=GET_DTYPE(),
    cond_latent=cond_latent,
)
```

For a matching LingBot-VA loop, the runner may also bind per-step inputs and prediction processing:

```python
scheduler.bind_step_inputs(self.inputs, self._build_video_step_inputs)
scheduler.bind_noise_pred_processor(self._postprocess_video_noise_pred)
```

Then `step_pre()` can update `self.inputs` for the current timestep before `model.infer(...)`.

Avoid per-step GPU-to-CPU synchronization:

- Store `self.step_index = int(step_index)` in `step_pre`.
- In `step`, use `self.step_index` directly for `sigmas`.
- Avoid `torch.argmin(...).item()` or checking tensor values on CPU in the hot loop.

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

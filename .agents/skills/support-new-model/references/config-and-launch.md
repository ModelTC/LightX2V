# Config And Launch

Detailed guidance from the [skill entry](../SKILL.md). Read the sections relevant to the task.

## Config Conventions

Configs should contain model/run defaults that are stable for a profile. Scripts should contain per-run values.

Do not hard-code local model roots, upstream source roots, adapter paths, audio paths, or sample inputs inside Python. Prefer existing CLI/config fields and set machine-specific values in the bash script or the selected config profile.

Do not add extra input JSON dependency files for a new model unless the existing LightX2V workflow already requires them. Keep structured model defaults in `configs/<family>/*.json`, and put per-run arguments in bash/CLI. A script can pass a JSON-valued config path, but it should not require a second model-private `input_json` just to provide prompt/audio/image paths that LightX2V already accepts.

Do not add a model-local `_apply_default_config()` that silently fills many fields. Put required stable values in JSON, or inherit existing global defaults intentionally. If a field is required for correctness, fail early with a clear error instead of hiding it behind a private defaults function.

Prefer LightX2V naming:

- `size: [height, width]` (pixels; CLI `--size HEIGHT WIDTH`)
- `infer_steps`, `action_infer_steps`
- `sample_shift`, `action_sample_shift`
- `sample_guide_scale`, `action_sample_guide_scale`
- `enable_cfg`, `enable_action_cfg`
- `fps` (startup JSON; Python `create_generator(fps=...)`)
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

- `height`, `width`, `target_height`, `target_width`, `target_fps`
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

- Put architecture and deployment fields in JSON: `size`, `num_frames`, `fps`, `infer_steps`, attention types, scheduler shift, offload/parallel settings, `use_compile`, and `warmup`. These are not all CLI request options.
- `build_cli_inputs()` separates startup fields (`config_json`, `model_cls`, `model_variant`, `model_path`, `task`) from request fields. `prepare_request()` rejects fields not declared for that task; add new request fields consistently in CLI, `InputInfo`, runner declarations and Python API when applicable.
- `build_startup_config()` merges model defaults and deployment JSON, with explicit handling for task, variant and API fps. Read that function for precedence; do not assume every CLI argument overrides a JSON field.
- Put run-local fields in bash/CLI: prompt, negative prompt, image/video/audio path, save path, visible devices, model class, task, and checkpoint path overrides that vary by machine or experiment.
- Before adding a new argparse/config name, search for an existing equivalent and reuse it. For example, use `adapter_model_path` for model adapters and `audio_encoder_path` for audio encoders instead of adding model-specific aliases.

For profiles copied from upstream names, keep filename compatibility if useful, but keep internal LightX2V semantics clear. Example: upstream may call a config `robotwin_i2av` while its actual `infer_mode` is `i2va`; in LightX2V the script can use `robotwin_i2av.json` while still passing `--task i2va`.

When deciding whether a field can be removed from JSON because a model config exists under `model_path`, verify `set_config.py` loads that config for this `model_cls`. If it does not, either add the merge or keep the field. Do not assume `transformer/config.json` is loaded globally.

## Bash Script Conventions

Follow existing LightX2V scripts:

```bash
#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/model

export CUDA_VISIBLE_DEVICES=0

source "${lightx2v_path}/scripts/base/base.sh"
cd "${lightx2v_path}"

python -m lightx2v.infer \
--model_cls my_model \
--task i2v \
--model_path "$model_path" \
--config_json "${lightx2v_path}/configs/my_model/profile.json" \
--prompt "..." \
--negative_prompt "" \
--image_path /path/to/input.png \
--save_result_path "${lightx2v_path}/save_results/output.mp4"
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

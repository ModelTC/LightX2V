# Python API

Install LightX2V using the [quickstart](quickstart.md). The examples below run from the repository root; replace model, input and output paths with local paths.

## Create a pipeline and generate

```python
from lightx2v import LightX2VPipeline

pipe = LightX2VPipeline(
    model_path="/path/to/Wan2.1-T2V-1.3B",
    model_cls="wan2.1",
    task="t2v",
)
pipe.create_generator(config_json="configs/wan/wan_t2v.json")
pipe.generate(
    seed=42,
    prompt="Two cats wearing boxing gloves on a stage.",
    negative_prompt="blurry, low quality",
    save_result_path="save_results/first_video.mp4",
)
```

Create the generator once, then call `generate()` repeatedly to reuse the loaded model. Each call supplies its own prompt, input paths, seed and output path. Select a task supported by the loaded model; `image_path`, `video_path`, `audio_path` and other inputs depend on that task.

For ordinary models, an omitted or `None` seed defaults to 42. NeoPP instead continues the session RNG state. Most runners skip saving when `save_result_path` is omitted; WorldMirror writes to its default output directory. Return values and support for `return_result_tensor=True` are model-specific.

## Configuration

Use the same JSON as the CLI to align startup settings. JSON fields override matching constructor and `enable_*()` startup fields, except that `task` and an explicitly set `model_variant` retain their constructor settings. With `config_json`, the manual sampling arguments of `create_generator()` are not applied; `fps` is handled separately. An explicit non-`None` `fps` argument takes precedence over JSON; omitting it or passing `None` uses the JSON value when present. Configure the chosen mode in one place.

Alternatively, configure a new generator manually. This is the configuration used by the [Wan2.1 14B Python example](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan/inference/wan_t2v.py); it differs from the 1.3B CLI example above:

```python
pipe.create_generator(
    attn_mode="sage_attn2",
    infer_steps=50,
    size=(480, 832),
    num_frames=81,
    guidance_scale=5.0,
    sample_shift=5.0,
)
```

`size` is `(height, width)`. `guidance_scale`, `sample_shift`, `num_frames`, attention operators and distilled schedules must match the selected model. Do not call `create_generator()` twice on the same pipeline.

The constructor accepts `dit_original_ckpt` for a separate DiT checkpoint, or `high_noise_original_ckpt` and `low_noise_original_ckpt` for Wan2.2 MoE. Distilled Wan models use `distill_method="dmd2"` where required. Model-specific instructions live beside the launchers below.

## Optional startup settings

Apply the following methods before `create_generator()`. Choose the settings supported by your model; these snippets describe separate options rather than one configuration to enable in full.

### Offload

```python
pipe.enable_offload(
    cpu_offload=True,
    offload_granularity="block",
    text_encoder_offload=True,
    image_encoder_offload=False,
    vae_offload=False,
)
```

Select the granularity and encoder/VAE offload settings for the model and available memory. See [offload](../deploy_guides/for_low_resource.md).

### Quantization

```python
pipe.enable_quantize(
    dit_quantized=True,
    dit_quantized_ckpt="/path/to/quantized_model.safetensors",
    quant_scheme="fp8-sgl",
)
```

For Wan2.2 MoE use `high_noise_quantized_ckpt` and `low_noise_quantized_ckpt`. Encoder quantization uses `text_encoder_quantized`, `text_encoder_quantized_ckpt`, and, where supported, `text_encoder_quant_scheme`; image encoder equivalents are also available. Match checkpoint format and installed kernels to the quantization scheme. See [quantization](../method_tutorials/quantization.md).

### LoRA

```python
pipe.enable_lora(
    [{"path": "/path/to/lora.safetensors", "strength": 1.0}],
    lora_dynamic_apply=True,
)
```

`lora_dynamic_apply=False` merges weights during loading. Dynamic application supports changing LoRA on a running pipeline through `switch_lora(lora_path=..., strength=...)`; see the [Qwen switch example](https://github.com/ModelTC/LightX2V/blob/main/scripts/qwen_image/inference/qwen_2511_with_distill_lora_dynamic_switch_lora.py). Wan2.2 LoRA entries also name the `high_noise_model` or `low_noise_model` branch. See [LoRA deployment](../deploy_guides/lora_deploy.md).

### Parallel inference

```python
pipe.enable_parallel(cfg_p_size=1, seq_p_size=4, seq_p_attn_type="ulysses")
```

```bash
torchrun --standalone --nproc_per_node=4 your_script.py
```

The process count must match the selected parallel configuration. See [parallel inference](../method_tutorials/parallel.md) for model-specific restrictions.

### Cache and lightweight VAE

`enable_cache(cache_method="Tea", ...)` and `enable_cache(cache_method="Mag", ...)` take model-specific coefficients or calibration ratios. Reuse a matching configuration from `configs/caching` or `configs/hunyuan_video_15/cache`; see [cache](../method_tutorials/cache.md).

`enable_lightvae(use_tae=True, tae_path="/path/to/tae.safetensors")` selects LightTAE. LightVAE uses `use_lightvae=True` and `vae_path`. Use weights for the selected model from the [Autoencoders collection](https://huggingface.co/lightx2v/Autoencoders).

## Model examples

Edit the selected Python file's model/input/output paths before running it. Keep the model, checkpoint, sampling parameters and attention implementation together when comparing results with a Shell launcher. For CFG-disabled requests, omit `negative_prompt`, including the empty string.

| Model or task | Entry |
| --- | --- |
| Wan2.1 / VACE / NVFP4 | [scripts/wan](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan/README.md) |
| Wan2.2 MoE / TI2V / Animate | [scripts/wan22](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan22/README.md) |
| Qwen Image / LoRA | [scripts/qwen_image](https://github.com/ModelTC/LightX2V/blob/main/scripts/qwen_image/README.md) |
| HunyuanVideo-1.5 | [scripts/hunyuan_video_15](https://github.com/ModelTC/LightX2V/tree/main/scripts/hunyuan_video_15/inference) |
| LTX-2 | [scripts/ltx2](https://github.com/ModelTC/LightX2V/blob/main/scripts/ltx2/README.md) |
| MiniMax-H3 DMD | [scripts/minimax_h3](https://github.com/ModelTC/LightX2V/blob/main/scripts/minimax_h3/inference/distill/minimax_h3_t2av_dmd.py) |
| LongCat Image | [scripts/longcat](https://github.com/ModelTC/LightX2V/blob/main/scripts/longcat/README.md) |
| SeedVR2 | [scripts/seedvr2](https://github.com/ModelTC/LightX2V/blob/main/scripts/seedvr2/README.md) |
| Z-Image-Turbo | [scripts/z_image](https://github.com/ModelTC/LightX2V/blob/main/scripts/z_image/README.md) |
| NeoPP KV replay | [scripts/neopp](https://github.com/ModelTC/LightX2V/blob/main/scripts/neopp/README.md) |
| WorldMirror | [scripts/worldmirror](https://github.com/ModelTC/LightX2V/blob/main/scripts/worldmirror/README.md) |
| WorldPlay | [scripts/worldplay](https://github.com/ModelTC/LightX2V/blob/main/scripts/worldplay/README.md) |

For robotics inference and evaluation, see [RealtimeWAM](https://github.com/ModelTC/LightX2V/blob/main/scripts/realtimewam/README.md). Model launch instructions belong in `scripts/<model>/README*`; shared API and deployment instructions belong in this documentation.

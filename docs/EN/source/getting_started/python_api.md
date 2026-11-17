# Python API

Install LightX2V using the [quickstart](quickstart.md). Run these examples from the repository root and replace model, input and output paths with local paths.

## Load a model and generate

Pass a configuration dictionary to `LightX2VPipeline(config)`. Construction loads the model; repeated `generate()` calls reuse it. A context manager closes the runner when finished.

```python
import json
from pathlib import Path

from lightx2v import LightX2VPipeline

config = json.loads(Path("configs/wan/wan_t2v.json").read_text())
config.update({
    "model_cls": "wan2.1",
    "model_path": "/path/to/Wan2.1-T2V-1.3B",
    "task": "t2v",
})

with LightX2VPipeline(config) as pipe:
    result = pipe.generate(
        seed=42,
        prompt="Two cats wearing boxing gloves on a stage.",
        negative_prompt="blurry, low quality",
        save_result_path="save_results/first_video.mp4",
    )
```

You can also construct the dictionary directly. Use the same deployment fields as the CLI and server. There are no separate sampling, quantization or offload function arguments.

## Configuration and request inputs

| Where | Contents |
| --- | --- |
| `config` | Model identity and paths, inference steps, CFG, default size/frame count, attention operators, quantization, offload, LoRA, caching and parallel settings |
| `generate(**inputs)` | Task, prompts, input media, seed, output path and other request fields supported by the model's Spec |

Example scripts read inference settings from JSON and supplement model selection fields: `model_path`, `model_cls` and, when needed, `task` and `model_variant`. Configure sampling, component checkpoints and runtime options in the selected JSON. The model configuration loader combines checkpoint metadata with this deployment configuration. Pipeline adds no sampling defaults or model-specific parameter conversions.

For example, add these fields to the JSON to configure Wan DiT and T5 offload:

```json
{
    "cpu_offload": true,
    "offload_granularity": "block",
    "t5_cpu_offload": true,
    "vae_cpu_offload": false
}
```

Encoder settings use their actual deployment names, such as `t5_cpu_offload`, `qwen25vl_cpu_offload` or `gemma_cpu_offload`. Select a configuration matching the model instead of copying encoder fields between models.

Quantization uses `dit_quantized`, `dit_quant_scheme` and the corresponding checkpoint paths. LoRA uses `lora_mode` and `lora_configs`. Parallel inference uses the `parallel` dictionary and a matching `torchrun` process count. Caches and lightweight VAEs likewise use their existing JSON fields. See [offload](../deploy_guides/for_low_resource.md), [quantization](../method_tutorials/quantization.md), [LoRA](../deploy_guides/lora_deploy.md), [parallel inference](../method_tutorials/parallel.md) and [cache](../method_tutorials/cache.md).

Inspect the loaded configuration through `pipe.config`. To change startup settings, create a new pipeline with the desired config. Model-specific runtime operations are available on `pipe.runner`; for example, supported LoRA runners expose `switch_lora(...)`.

## Requests and results

Request fields come from the same Spec used by CLI and server. The runner validates the task and supported fields before creating InputInfo. `generate()` does not modify startup configuration.

A `task` explicitly included in the config is the default for requests that omit it. A request can choose another supported task without changing that default. Without a configured default, single-task runners select their only task; multi-task runners require a request task. NeoPP retains its runner-specific omitted-task behavior.

Multi-input fields use lists, including for a single item. Size is `[height, width]`. Fields already supported as per-request overrides by the Spec retain that behavior.

For ordinary models, an omitted or `None` seed defaults to 42. NeoPP instead continues the session RNG state. Explicit negative prompts are ignored with a warning when CFG is disabled. Most runners skip saving when `save_result_path` is omitted; WorldMirror uses its default output directory. Return values and support for `return_result_tensor=True` remain model-specific.

Use `with`, or call `pipe.close()` when finished. Cleanup releases runner resources and any distributed process group created by the pipeline. An existing caller-owned process group is kept.

## Saving image files

Image generation, editing and super-resolution save through OpenCV by default. Set `save_image_method="pil"` in the config to select PIL. CLI and server use the same setting. Each model keeps its postprocessing and normalization rules.

With `PROFILING_DEBUG_LEVEL>=1`, image-saving profiling includes tensor conversion, required postprocessing and file writing. This option controls saved image files; in-memory PNG/Base64 encoding keeps its existing behavior.

## Model examples

Edit the selected Python file's model/input/output paths before running it. Keep the model, checkpoint, sampling parameters and attention implementation together when comparing results with a Shell launcher. For CFG-disabled requests, explicitly supplied `negative_prompt` values, including the empty string, are ignored with a warning.

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

Multi-input fields take lists even for one input, for example `image_path=["a.png"]`. Use `image_path=["a.png", "b.png"]` only for models that support multiple images; single-image models still take a string. Strengths and frame indices also use lists, for example `image_strength=[1.0, 0.8]` and `image_frame_indices=[0, 32]`.

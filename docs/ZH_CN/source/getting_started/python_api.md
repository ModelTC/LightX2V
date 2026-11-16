# Python API

请先按照[快速开始](quickstart.md)安装 LightX2V。以下示例从仓库根目录运行，模型、输入和输出路径需要替换为本地路径。

## 加载模型并生成

使用配置字典创建 `LightX2VPipeline(config)`。构造时加载模型，多次调用 `generate()` 复用已加载的权重；退出 `with` 时释放资源。

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
        prompt="两只戴着拳击手套的猫站在舞台上。",
        negative_prompt="模糊，低质量",
        save_result_path="save_results/first_video.mp4",
    )
```

也可以直接编写配置字典。字段名称与 CLI、server 的部署配置一致，不提供单独设置采样、量化或 offload 的函数参数。

## 配置与请求输入

| 位置 | 内容 |
| --- | --- |
| `config` | 模型名称和路径、推理步数、CFG、默认尺寸和帧数、注意力算子、量化、offload、LoRA、缓存、并行设置 |
| `generate(**inputs)` | 本次任务、提示词、输入媒体、随机种子、输出路径，以及模型 Spec 支持的其他请求字段 |

示例脚本从 JSON 读取推理配置，补充用于选择模型的 `model_path`、`model_cls`，以及需要时的 `task` 和 `model_variant`。采样参数、组件权重路径和运行选项都在对应 JSON 中设置。模型配置加载器负责合并 checkpoint 元数据与部署配置，Pipeline 不再添加通用采样默认值或转换模型专属参数。

例如，在 JSON 中添加以下字段，配置 Wan 的 DiT 与 T5 offload：

```json
{
    "cpu_offload": true,
    "offload_granularity": "block",
    "t5_cpu_offload": true,
    "vae_cpu_offload": false
}
```

编码器配置使用实际部署字段，例如 `t5_cpu_offload`、`qwen25vl_cpu_offload`、`gemma_cpu_offload`。请选择对应模型的配置，不要在模型之间直接照搬编码器字段。

量化使用 `dit_quantized`、`dit_quant_scheme` 与相应权重路径；LoRA 使用 `lora_mode`、`lora_configs`；并行使用 `parallel` 字典，并通过对应进程数的 `torchrun` 启动。缓存和轻量 VAE 同样使用已有 JSON 字段。具体参见 [offload](../deploy_guides/for_low_resource.md)、[量化](../method_tutorials/quantization.md)、[LoRA](../deploy_guides/lora_deploy.md)、[并行推理](../method_tutorials/parallel.md)和[缓存](../method_tutorials/cache.md)。

通过 `pipe.config` 查看加载后的配置。修改启动配置时，使用新的 config 创建 Pipeline。模型专属的运行时操作通过 `pipe.runner` 调用，例如支持动态 LoRA 的 runner 提供 `switch_lora(...)`。

## 请求与返回值

请求字段来自 CLI、server 共用的 Spec。runner 检查任务和字段后创建 InputInfo，`generate()` 不修改启动配置。

config 中明确指定的 `task` 是请求的默认任务。请求可以选择另一个支持的任务，不会改变默认值。没有配置默认任务时，单任务 runner 使用唯一任务，多任务 runner 要求请求传入 task；NeoPP 保留自身对省略 task 的处理。

多输入字段使用列表，只有一个输入时也使用列表。尺寸顺序为 `[height, width]`。Spec 原本支持的逐请求覆盖字段继续有效。

普通模型省略 seed 或传入 `None` 时使用 42；NeoPP 延续会话随机数状态。关闭 CFG 时，明确传入的 negative_prompt 会被警告并忽略。省略 save_result_path 时，大部分 runner 不保存文件，WorldMirror 使用默认输出目录。返回值及 `return_result_tensor=True` 的支持情况由具体模型决定。

使用 `with`，或在使用结束后调用 `pipe.close()`。关闭时释放 runner 资源，以及 Pipeline 自己创建的分布式进程组；调用方已有的进程组会保留。

## 图片文件保存

图像生成、编辑和超分默认通过 OpenCV 保存文件。可以在 config 中设置 `save_image_method="pil"` 改用 PIL，CLI 和 server 使用相同配置。各模型的后处理和归一化规则不变。

当 `PROFILING_DEBUG_LEVEL>=1` 时，图片保存耗时包含张量转换、必要的后处理和文件写入。该选项控制图片文件保存，内存中的 PNG/Base64 编码行为不变。

## 模型示例

运行前修改所选 Python 文件中的模型、输入和输出路径。与 Shell 入口比较结果时，同时核对模型、权重、采样参数和注意力实现。关闭 CFG 时，显式传入的 `negative_prompt`（包括空字符串）会被忽略，并记录警告。

| 模型或任务 | 入口 |
| --- | --- |
| Wan2.1 / VACE / NVFP4 | [scripts/wan](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan/README_zh.md) |
| Wan2.2 MoE / TI2V / Animate | [scripts/wan22](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan22/README_zh.md) |
| Qwen Image / LoRA | [scripts/qwen_image](https://github.com/ModelTC/LightX2V/blob/main/scripts/qwen_image/README_zh.md) |
| HunyuanVideo-1.5 | [scripts/hunyuan_video_15](https://github.com/ModelTC/LightX2V/tree/main/scripts/hunyuan_video_15/inference) |
| LTX-2 | [scripts/ltx2](https://github.com/ModelTC/LightX2V/blob/main/scripts/ltx2/README_zh.md) |
| MiniMax-H3 DMD | [scripts/minimax_h3](https://github.com/ModelTC/LightX2V/blob/main/scripts/minimax_h3/inference/distill/minimax_h3_t2av_dmd.py) |
| LongCat Image | [scripts/longcat](https://github.com/ModelTC/LightX2V/blob/main/scripts/longcat/README_zh.md) |
| SeedVR2 | [scripts/seedvr2](https://github.com/ModelTC/LightX2V/blob/main/scripts/seedvr2/README_zh.md) |
| Z-Image-Turbo | [scripts/z_image](https://github.com/ModelTC/LightX2V/blob/main/scripts/z_image/README_zh.md) |
| NeoPP KV replay | [scripts/neopp](https://github.com/ModelTC/LightX2V/blob/main/scripts/neopp/README.md) |
| WorldMirror | [scripts/worldmirror](https://github.com/ModelTC/LightX2V/blob/main/scripts/worldmirror/README_zh.md) |
| WorldPlay | [scripts/worldplay](https://github.com/ModelTC/LightX2V/blob/main/scripts/worldplay/README.md) |

机器人推理和评测见 [RealtimeWAM](https://github.com/ModelTC/LightX2V/blob/main/scripts/realtimewam/README.md)。模型启动说明维护在 `scripts/<model>/README*`，通用 API 和部署说明维护在正式文档中。

多输入字段传入列表，即使只有一个输入，例如 `image_path=["a.png"]`。仅当模型支持多图时使用 `image_path=["a.png", "b.png"]`；单图模型继续传字符串。条件强度与帧索引同样使用列表，例如 `image_strength=[1.0, 0.8]`、`image_frame_indices=[0, 32]`。

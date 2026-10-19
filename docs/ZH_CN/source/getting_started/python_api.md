# Python API

先按[快速入门](quickstart.md)安装 LightX2V。下列示例从仓库根目录运行，请将模型、输入和输出路径替换为本地路径。

## 创建 Pipeline 并生成

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

只创建一次 generator，然后重复调用 `generate()` 复用已加载的模型。每次调用分别传入提示词、输入路径、seed 和输出路径。任务必须由已加载的模型支持；`image_path`、`video_path`、`audio_path` 等输入随任务而定。

普通模型省略 seed 或传入 `None` 时默认使用 42；NeoPP 则延续会话中的随机数状态。大多数 runner 在省略 `save_result_path` 时不保存文件；WorldMirror 会写入默认输出目录。返回值以及是否支持 `return_result_tensor=True` 取决于模型。

## 配置方式

使用与 CLI 相同的 JSON 可以对齐启动设置。JSON 中的字段覆盖构造函数和 `enable_*()` 中的同名启动字段，但 `task` 和显式设置的 `model_variant` 沿用构造时的设置。传入 `config_json` 时，`create_generator()` 的手动采样参数不生效；`fps` 单独处理，显式传入的非 `None` 值优先于 JSON 中的 `fps`，省略或传入 `None` 时才使用 JSON 的值（若有）。请在选定的一种方式中维护配置。

也可以手动配置新的 generator。下面是 [Wan2.1 14B Python 示例](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan/inference/wan_t2v.py)采用的设置，与前面的 1.3B CLI 示例不同：

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

`size` 为 `(高, 宽)`。`guidance_scale`、`sample_shift`、`num_frames`、注意力算子和蒸馏采样序列应与所选模型匹配。同一个 Pipeline 不能调用两次 `create_generator()`。

构造函数可通过 `dit_original_ckpt` 指定独立 DiT 权重；Wan2.2 MoE 使用 `high_noise_original_ckpt`、`low_noise_original_ckpt` 指定双分支权重。需要时，Wan 蒸馏模型设置 `distill_method="dmd2"`。模型专属说明见下方启动入口。

## 可选启动设置

以下方法在 `create_generator()` 之前调用。根据模型选择支持的选项；这些片段分别演示不同功能，不代表需要全部开启。

### 参数卸载

```python
pipe.enable_offload(
    cpu_offload=True,
    offload_granularity="block",
    text_encoder_offload=True,
    image_encoder_offload=False,
    vae_offload=False,
)
```

根据模型和可用内存选择卸载粒度、编码器及 VAE 的卸载方式，见[参数卸载](../deploy_guides/for_low_resource.md)。

### 量化

```python
pipe.enable_quantize(
    dit_quantized=True,
    dit_quantized_ckpt="/path/to/quantized_model.safetensors",
    quant_scheme="fp8-sgl",
)
```

Wan2.2 MoE 使用 `high_noise_quantized_ckpt`、`low_noise_quantized_ckpt`。文本编码器量化使用 `text_encoder_quantized`、`text_encoder_quantized_ckpt`，支持时可通过 `text_encoder_quant_scheme` 指定方案；图像编码器也有对应选项。权重格式和安装的算子须与量化方案匹配，见[模型量化](../method_tutorials/quantization.md)。

### LoRA

```python
pipe.enable_lora(
    [{"path": "/path/to/lora.safetensors", "strength": 1.0}],
    lora_dynamic_apply=True,
)
```

`lora_dynamic_apply=False` 在加载时合并权重。动态应用模式可在同一个 Pipeline 中调用 `switch_lora(lora_path=..., strength=...)` 切换 LoRA，见 [Qwen 切换示例](https://github.com/ModelTC/LightX2V/blob/main/scripts/qwen_image/inference/qwen_2511_with_distill_lora_dynamic_switch_lora.py)。Wan2.2 LoRA 条目还需用 `name` 指定 `high_noise_model` 或 `low_noise_model` 分支，见 [LoRA 部署](../deploy_guides/lora_deploy.md)。

### 并行推理

```python
pipe.enable_parallel(cfg_p_size=1, seq_p_size=4, seq_p_attn_type="ulysses")
```

```bash
torchrun --standalone --nproc_per_node=4 your_script.py
```

进程数须与所选并行配置一致。各模型的限制见[并行推理](../method_tutorials/parallel.md)。

### 特征缓存与轻量 VAE

`enable_cache(cache_method="Tea", ...)` 和 `enable_cache(cache_method="Mag", ...)` 使用模型对应的系数或校准比率。复用 `configs/caching` 或 `configs/hunyuan_video_15/cache` 中匹配的配置，见[特征缓存](../method_tutorials/cache.md)。

`enable_lightvae(use_tae=True, tae_path="/path/to/tae.safetensors")` 选择 LightTAE；LightVAE 使用 `use_lightvae=True` 与 `vae_path`。从 [Autoencoders](https://huggingface.co/lightx2v/Autoencoders) 选择与模型匹配的权重。

## 模型示例

运行前修改所选 Python 文件中的模型、输入和输出路径。与 Shell 入口比较结果时，同时核对模型、权重、采样参数和注意力实现。关闭 CFG 的请求不传 `negative_prompt`，包括空字符串。

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

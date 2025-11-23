# LightX2V 使用示例

本文档介绍如何使用 LightX2V 进行视频生成，包括基础使用和进阶配置。

## 📋 目录

- [环境安装](#环境安装)
- [基础运行示例](#基础运行示例)
- [模型路径配置](#模型路径配置)
- [创建生成器](#创建生成器)
- [进阶配置](#进阶配置)
  - [参数卸载 (Offload)](#参数卸载-offload)
  - [模型量化 (Quantization)](#模型量化-quantization)
  - [并行推理 (Parallel Inference)](#并行推理-parallel-inference)
  - [特征缓存 (Cache)](#特征缓存-cache)
  - [轻量 VAE (Light VAE)](#轻量-vae-light-vae)

## 🔧 环境安装

请参考主项目的[快速入门文档](../docs/ZH_CN/source/getting_started/quickstart.md)进行环境安装。

## 🚀 基础运行示例

最小化代码示例可参考 `examples/wan_t2v.py`：

```python
from lightx2v import LightX2VPipeline

pipe = LightX2VPipeline(
    model_path="/path/to/Wan2.1-T2V-14B",
    model_cls="wan2.1",
    task="t2v",
)

pipe.create_generator(
    attn_mode="sage_attn2",
    infer_steps=50,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=5.0,
    sample_shift=5.0,
)

seed = 42
prompt = "Your prompt here"
negative_prompt = ""
save_result_path="/path/to/save_results/output.mp4"

pipe.generate(
    seed=seed,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
```

## 📁 模型路径配置

### 基础配置

将模型路径传入 `LightX2VPipeline`：

```python
pipe = LightX2VPipeline(
    image_path="/path/to/img_0.jpg",  # I2V 任务需要
    model_path="/path/to/Wan2.2-I2V-A14B",
    model_cls="wan2.2_moe",  # 对于 wan2.1，使用 "wan2.1"
    task="i2v",
)
```

### 多版本模型权重指定

当 `model_path` 目录下存在多个不同版本的 bf16 精度 DIT 模型 safetensors 文件时，需要使用以下参数指定具体使用哪个权重：

- **`dit_original_ckpt`**: 用于指定 wan2.1 和 hunyuan15 等模型的原始 DIT 权重路径
- **`low_noise_original_ckpt`**: 用于指定 wan2.2 模型的低噪声分支权重路径
- **`high_noise_original_ckpt`**: 用于指定 wan2.2 模型的高噪声分支权重路径

**使用示例：**

```python
pipe = LightX2VPipeline(
    model_path="/path/to/Wan2.2-I2V-A14B",
    model_cls="wan2.2_moe",
    task="i2v",
    low_noise_original_ckpt="/path/to/low_noise_model.safetensors",
    high_noise_original_ckpt="/path/to/high_noise_model.safetensors",
)
```

## 🎛️ 创建生成器

### 从配置文件加载

生成器可以从 JSON 配置文件直接加载，配置文件位于 `configs` 目录：

```python
pipe.create_generator(config_json="../configs/wan/wan_t2v.json")
```

### 手动创建生成器

也可以手动创建生成器，并配置多个参数：

```python
pipe.create_generator(
    attn_mode="flash_attn2",  # 可选: flash_attn2, flash_attn3, sage_attn2, sage_attn3 (B架构显卡适用)
    infer_steps=50,           # 推理步数
    num_frames=81,            # 视频帧数
    height=480,               # 视频高度
    width=832,                # 视频宽度
    guidance_scale=5.0,       # CFG引导强度 (=1时弃用CFG)
    sample_shift=5.0,         # 采样偏移
    fps=16,                   # 帧率
    aspect_ratio="16:9",      # 宽高比
    boundary=0.900,           # 边界值
    boundary_step_index=2,    # 边界步索引
    denoising_step_list=[1000, 750, 500, 250],  # 去噪步列表
)
```

**参数说明：**
- **分辨率**: 通过 `height` 和 `width` 指定
- **CFG**: 通过 `guidance_scale` 指定（设置为 1 时禁用 CFG）
- **FPS**: 通过 `fps` 指定帧率
- **视频长度**: 通过 `num_frames` 指定帧数
- **推理步数**: 通过 `infer_steps` 指定
- **采样偏移**: 通过 `sample_shift` 指定
- **注意力模式**: 通过 `attn_mode` 指定，可选 `flash_attn2`, `flash_attn3`, `sage_attn2`, `sage_attn3`（B架构显卡适用）

## ⚙️ 进阶配置

**⚠️ 重要提示：手动创建生成器时，可以配置一些进阶选项，所有进阶配置必须在 `create_generator()` 之前指定，否则会失效！**

### 参数卸载 (Offload)

显著降低显存占用，几乎不影响推理速度，适用于 RTX 30/40/50 系列显卡。

```python
pipe.enable_offload(
    cpu_offload=True,              # 启用 CPU 卸载
    offload_granularity="block",   # 卸载粒度: "block" 或 "phase"
    text_encoder_offload=False,    # 文本编码器是否卸载
    image_encoder_offload=False,   # 图像编码器是否卸载
    vae_offload=False,             # VAE 是否卸载
)
```

**说明：**
- 对于 Wan 模型，`offload_granularity` 支持 `"block"` 和 `"phase"`
- 对于 HunyuanVideo-1.5，目前只支持 `"block"`

### 模型量化 (Quantization)

量化可以显著降低显存占用并加速推理。

```python
pipe.enable_quantize(
    dit_quantized=False,                    # 是否使用量化的 DIT 模型
    text_encoder_quantized=False,           # 是否使用量化的文本编码器
    image_encoder_quantized=False,          # 是否使用量化的图像编码器
    dit_quantized_ckpt=None,                # DIT 量化权重路径（当 model_path 下没有量化权重或存在多个权重时需要指定）
    low_noise_quantized_ckpt=None,          # Wan2.2 低噪声分支量化权重路径
    high_noise_quantized_ckpt=None,         # Wan2.2 高噪声分支量化权重路径
    text_encoder_quantized_ckpt=None,       # 文本编码器量化权重路径（当 model_path 下没有量化权重或存在多个权重时需要指定）
    image_encoder_quantized_ckpt=None,      # 图像编码器量化权重路径（当 model_path 下没有量化权重或存在多个权重时需要指定）
    quant_scheme="fp8-sgl",                 # 量化方案
)
```

**参数说明：**
- **`dit_quantized_ckpt`**: 当 `model_path` 目录下没有量化权重，或存在多个权重文件时，需要指定具体的 DIT 量化权重路径
- **`text_encoder_quantized_ckpt`** 和 **`image_encoder_quantized_ckpt`**: 类似地，用于指定编码器的量化权重路径
- **`low_noise_quantized_ckpt`** 和 **`high_noise_quantized_ckpt`**: 用于指定 Wan2.2 模型的双分支量化权重

**量化模型下载：**

- **Wan-2.1 量化模型**: 从 [Hy1.5-Quantized-Models](https://huggingface.co/lightx2v/Wan2.1-Distill-Models) 下载
- **Wan-2.2 量化模型**: 从 [Hy1.5-Quantized-Models](https://huggingface.co/lightx2v/Wan2.2-Distill-Models) 下载
- **HunyuanVideo-1.5 量化模型**: 从 [Hy1.5-Quantized-Models](https://huggingface.co/lightx2v/Hy1.5-Quantized-Models) 下载
  - `hy15_qwen25vl_llm_encoder_fp8_e4m3_lightx2v.safetensors` 是文本编码器的量化权重

**使用示例：**

```python
# HunyuanVideo-1.5 量化示例
pipe.enable_quantize(
    quant_scheme='fp8-sgl',
    dit_quantized=True,
    dit_quantized_ckpt="/path/to/hy15_720p_i2v_fp8_e4m3_lightx2v.safetensors",
    text_encoder_quantized=True,
    image_encoder_quantized=False,
    text_encoder_quantized_ckpt="/path/to/hy15_qwen25vl_llm_encoder_fp8_e4m3_lightx2v.safetensors",
)

# Wan2.1 量化示例
pipe.enable_quantize(
    dit_quantized=True,
    dit_quantized_ckpt="/path/to/wan2.1_i2v_480p_scaled_fp8_e4m3_lightx2v_4step.safetensors",
)

# Wan2.2 量化示例
pipe.enable_quantize(
    dit_quantized=True,
    low_noise_quantized_ckpt="/path/to/wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors",
    high_noise_quantized_ckpt="/path/to/wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step_1030.safetensors",
)
```

**量化方案参考：** 详细说明请参考 [量化文档](../docs/ZH_CN/source/method_tutorials/quantization.md)

### 并行推理 (Parallel Inference)

支持多 GPU 并行推理，需要使用 `torchrun` 运行：

```python
pipe.enable_parallel(
    seq_p_size=4,                    # 序列并行大小
    seq_p_attn_type="ulysses",       # 序列并行注意力类型
)
```

**运行方式：**
```bash
torchrun --nproc_per_node=4 your_script.py
```

### 特征缓存 (Cache)

可以指定缓存方法为 Mag 或 Tea，使用 MagCache 和 TeaCache 方法：

```python
pipe.enable_cache(
    cache_method='Tea',  # 缓存方法: 'Tea' 或 'Mag'
    coefficients=[-3.08907507e+04, 1.67786188e+04, -3.19178643e+03,
                  2.60740519e+02, -8.19205881e+00, 1.07913775e-01],  # 系数
    teacache_thresh=0.15,  # TeaCache 阈值
)
```

**系数参考：** 可参考 `configs/caching` 或 `configs/hunyuan_video_15/cache` 目录下的配置文件

### 轻量 VAE (Light VAE)

使用轻量 VAE 可以加速解码并降低显存占用。

```python
pipe.enable_lightvae(
    use_lightvae=False,    # 是否使用 LightVAE
    use_tae=False,         # 是否使用 LightTAE
    vae_path=None,         # LightVAE 的路径
    tae_path=None,         # LightTAE 的路径
)
```

**支持情况：**
- **LightVAE**: 目前只支持 wan2.1、wan2.2 moe
- **LightTAE**: 目前只支持 wan2.1、wan2.2-ti2v、wan2.2 moe、HunyuanVideo-1.5

**模型下载：** 轻量 VAE 模型可从 [Autoencoders](https://huggingface.co/lightx2v/Autoencoders) 下载

- Wan-2.1 的 LightVAE: [lightvaew2_1.safetensors](https://huggingface.co/lightx2v/Autoencoders/blob/main/lightvaew2_1.safetensors)
- Wan-2.1 的 LightTAE: [lighttaew2_1.safetensors](https://huggingface.co/lightx2v/Autoencoders/blob/main/lighttaew2_1.safetensors)
- Wan-2.2-ti2v 的 LightTAE: [lighttaew2_2.safetensors](https://huggingface.co/lightx2v/Autoencoders/blob/main/lighttaew2_2.safetensors)
- HunyuanVideo-1.5 的 LightTAE: [lighttaehy1_5.safetensors](https://huggingface.co/lightx2v/Autoencoders/blob/main/lighttaehy1_5.safetensors)

**使用示例：**

```python
# 使用 HunyuanVideo-1.5 的 LightTAE
pipe.enable_lightvae(
    use_tae=True,
    tae_path="/path/to/lighttaehy1_5.safetensors",
    use_lightvae=False,
    vae_path=None
)
```

## 📚 更多资源

- [完整文档](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/)
- [GitHub 仓库](https://github.com/ModelTC/LightX2V)
- [HuggingFace 模型库](https://huggingface.co/lightx2v)

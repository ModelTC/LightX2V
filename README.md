<div align="center" style="font-family: charter;">
  <h1>⚡️ LightX2V:<br> Light Video Generation Inference Framework</h1>

<img alt="logo" src="assets/img_lightx2v.png" width=75%></img>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ModelTC/lightx2v)
[![Doc](https://img.shields.io/badge/docs-English-99cc2)](https://lightx2v-en.readthedocs.io/en/latest)
[![Doc](https://img.shields.io/badge/文档-中文-99cc2)](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest)
[![Papers](https://img.shields.io/badge/论文集-中文-99cc2)](https://lightx2v-papers-zhcn.readthedocs.io/zh-cn/latest)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://hub.docker.com/r/lightx2v/lightx2v/tags)

**\[ English | [中文](README_zh.md) \]**

</div>

--------------------------------------------------------------------------------

**LightX2V** is an advanced lightweight video generation inference framework engineered to deliver efficient, high-performance video synthesis solutions. This unified platform integrates multiple state-of-the-art video generation techniques, supporting diverse generation tasks including text-to-video (T2V) and image-to-video (I2V). **X2V represents the transformation of different input modalities (X, such as text or images) into video output (V)**.

> 🌐 **Try it online now!** Experience LightX2V without installation: **[LightX2V Online Service](https://x2v.light-ai.top/login)** - Free, lightweight, and fast AI digital human video generation platform.

## :fire: Latest News

- **December 15, 2025:** 🚀 Supported deployment on Hygon DCU.

- **December 4, 2025:** 🚀 Supported GGUF format model inference & deployment on Cambricon MLU590/MetaX C500.

- **November 24, 2025:** 🚀 We released 4-step distilled models for HunyuanVideo-1.5! These models enable **ultra-fast 4-step inference** without CFG requirements, achieving approximately **25x speedup** compared to standard 50-step inference. Both base and FP8 quantized versions are now available: [Hy1.5-Distill-Models](https://huggingface.co/lightx2v/Hy1.5-Distill-Models).

- **November 21, 2025:** 🚀 We support the [HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5) video generation model since Day 0. With the same number of GPUs, LightX2V can achieve a speed improvement of over 2 times and supports deployment on GPUs with lower memory (such as the 24GB RTX 4090). It also supports CFG/Ulysses parallelism, efficient offloading, TeaCache/MagCache technologies, and more. We will soon update more models on our [HuggingFace page](https://huggingface.co/lightx2v), including step distillation, VAE distillation, and other related models. Quantized models and lightweight VAE models are now available: [Hy1.5-Quantized-Models](https://huggingface.co/lightx2v/Hy1.5-Quantized-Models) for quantized inference, and [LightTAE for HunyuanVideo-1.5](https://huggingface.co/lightx2v/Autoencoders/blob/main/lighttaehy1_5.safetensors) for fast VAE decoding. Refer to [this](https://github.com/ModelTC/LightX2V/tree/main/scripts/hunyuan_video_15) for usage tutorials, or check out the [examples directory](https://github.com/ModelTC/LightX2V/tree/main/examples) for code examples.


## 🏆 Performance Benchmarks (Updated on 2025.12.01)

### 📊 Cross-Framework Performance Comparison (H100)

| Framework | GPUs | Step Time | Speedup |
|-----------|---------|---------|---------|
| Diffusers | 1 | 9.77s/it | 1x |
| xDiT | 1 | 8.93s/it | 1.1x |
| FastVideo | 1 | 7.35s/it | 1.3x |
| SGL-Diffusion | 1 | 6.13s/it | 1.6x |
| **LightX2V** | 1 | **5.18s/it** | **1.9x** 🚀 |
| FastVideo | 8 | 2.94s/it | 1x |
| xDiT | 8 | 2.70s/it | 1.1x |
| SGL-Diffusion | 8 | 1.19s/it | 2.5x |
| **LightX2V** | 8 | **0.75s/it** | **3.9x** 🚀 |

### 📊 Cross-Framework Performance Comparison (RTX 4090D)

| Framework | GPUs | Step Time | Speedup |
|-----------|---------|---------|---------|
| Diffusers | 1 | 30.50s/it | 1x |
| FastVideo | 1 | 22.66s/it | 1.3x |
| xDiT | 1 | OOM | OOM |
| SGL-Diffusion | 1 | OOM | OOM |
| **LightX2V** | 1 | **20.26s/it** | **1.5x** 🚀 |
| FastVideo | 8 | 15.48s/it | 1x |
| xDiT | 8 | OOM | OOM |
| SGL-Diffusion | 8 | OOM | OOM |
| **LightX2V** | 8 | **4.75s/it** | **3.3x** 🚀 |

### 📊 LightX2V Performance Comparison

| Framework | GPU | Configuration | Step Time | Speedup |
|-----------|-----|---------------|-----------|---------------|
| **LightX2V** | H100 | 8 GPUs + cfg | 0.75s/it | 1x |
| **LightX2V** | H100 | 8 GPUs + no cfg | 0.39s/it | 1.9x |
| **LightX2V** | H100 | **8 GPUs + no cfg + fp8** | **0.35s/it** | **2.1x** 🚀 |
| **LightX2V** | 4090D | 8 GPUs + cfg | 4.75s/it | 1x |
| **LightX2V** | 4090D | 8 GPUs + no cfg | 3.13s/it | 1.5x |
| **LightX2V** | 4090D | **8 GPUs + no cfg + fp8** | **2.35s/it** | **2.0x** 🚀 |

**Note**: All the above performance data were tested on Wan2.1-I2V-14B-480P(40 steps, 81 frames). In addition, we also provide 4-step distilled models on the [HuggingFace page](https://huggingface.co/lightx2v).


## 💡 Quick Start

For comprehensive usage instructions, please refer to our documentation: **[English Docs](https://lightx2v-en.readthedocs.io/en/latest/) | [中文文档](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/)**

**We highly recommend using the Docker environment, as it is the simplest and fastest way to set up the environment. For details, please refer to the Quick Start section in the documentation.**

### Installation from Git
```bash
pip install -v git+https://github.com/ModelTC/LightX2V.git
```

### Building from Source
```bash
git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V
uv pip install -v . # pip install -v .
```

### (Optional) Install Attention/Quantize Operators
For attention operators installation, please refer to our documentation: **[English Docs](https://lightx2v-en.readthedocs.io/en/latest/getting_started/quickstart.html#step-4-install-attention-operators) | [中文文档](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/getting_started/quickstart.html#id9)**

### Usage Example

```python
# examples/wan/wan_i2v.py
"""
Wan2.2 image-to-video generation example.
This example demonstrates how to use LightX2V with Wan2.2 model for I2V generation.
"""

from lightx2v import LightX2VPipeline

# Initialize pipeline for Wan2.2 I2V task
# For wan2.1, use model_cls="wan2.1"
pipe = LightX2VPipeline(
    model_path="/path/to/Wan2.2-I2V-A14B",
    model_cls="wan2.2_moe",
    task="i2v",
)

# Alternative: create generator from config JSON file
# pipe.create_generator(
#     config_json="configs/wan22/wan_moe_i2v.json"
# )

# Enable offloading to significantly reduce VRAM usage with minimal speed impact
# Suitable for RTX 30/40/50 consumer GPUs
pipe.enable_offload(
    cpu_offload=True,
    offload_granularity="block",  # For Wan models, supports both "block" and "phase"
    text_encoder_offload=True,
    image_encoder_offload=False,
    vae_offload=False,
)

# Create generator manually with specified parameters
pipe.create_generator(
    attn_mode="sage_attn2",
    infer_steps=40,
    height=480,  # Can be set to 720 for higher resolution
    width=832,  # Can be set to 1280 for higher resolution
    num_frames=81,
    guidance_scale=[3.5, 3.5],  # For wan2.1, guidance_scale is a scalar (e.g., 5.0)
    sample_shift=5.0,
)

# Generation parameters
seed = 42
prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
image_path="/path/to/img_0.jpg"
save_result_path = "/path/to/save_results/output.mp4"

# Generate video
pipe.generate(
    seed=seed,
    image_path=image_path,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
```

> 💡 **More Examples**: For more usage examples including quantization, offloading, caching, and other advanced configurations, please refer to the [examples directory](https://github.com/ModelTC/LightX2V/tree/main/examples).



## 🤖 Supported Model Ecosystem

### Official Open-Source Models
- ✅ [HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5)
- ✅ [Wan2.1 & Wan2.2](https://huggingface.co/Wan-AI/)
- ✅ [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)
- ✅ [Qwen-Image-Edit](https://huggingface.co/spaces/Qwen/Qwen-Image-Edit)
- ✅ [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)

### Quantized and Distilled Models/LoRAs (**🚀 Recommended: 4-step inference**)
- ✅ [Wan2.1-Distill-Models](https://huggingface.co/lightx2v/Wan2.1-Distill-Models)
- ✅ [Wan2.2-Distill-Models](https://huggingface.co/lightx2v/Wan2.2-Distill-Models)
- ✅ [Wan2.1-Distill-Loras](https://huggingface.co/lightx2v/Wan2.1-Distill-Loras)
- ✅ [Wan2.2-Distill-Loras](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras)

### Lightweight Autoencoder Models (**🚀 Recommended: fast inference & low memory usage**)
- ✅ [Autoencoders](https://huggingface.co/lightx2v/Autoencoders)

### Autoregressive Models
- ✅ [Wan2.1-T2V-CausVid](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-CausVid)
- ✅ [Self-Forcing](https://github.com/guandeh17/Self-Forcing)
- ✅ [Matrix-Game-2.0](https://huggingface.co/Skywork/Matrix-Game-2.0)

🔔 Follow our [HuggingFace page](https://huggingface.co/lightx2v) for the latest model releases from our team.

💡 Refer to the [Model Structure Documentation](https://lightx2v-en.readthedocs.io/en/latest/getting_started/model_structure.html) to quickly get started with LightX2V

## 🚀 Frontend Interfaces

We provide multiple frontend interface deployment options:

- **🎨 Gradio Interface**: Clean and user-friendly web interface, perfect for quick experience and prototyping
  - 📖 [Gradio Deployment Guide](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/deploy_gradio.html)
- **🎯 ComfyUI Interface**: Powerful node-based workflow interface, supporting complex video generation tasks
  - 📖 [ComfyUI Deployment Guide](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/deploy_comfyui.html)
- **🚀 Windows One-Click Deployment**: Convenient deployment solution designed for Windows users, featuring automatic environment configuration and intelligent parameter optimization
  - 📖 [Windows One-Click Deployment Guide](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/deploy_local_windows.html)

**💡 Recommended Solutions**:
- **First-time Users**: We recommend the Windows one-click deployment solution
- **Advanced Users**: We recommend the ComfyUI interface for more customization options
- **Quick Experience**: The Gradio interface provides the most intuitive operation experience

## 🚀 Core Features

### 🎯 **Ultimate Performance Optimization**
- **🔥 SOTA Inference Speed**: Achieve **~20x** acceleration via step distillation and system optimization (single GPU)
- **⚡️ Revolutionary 4-Step Distillation**: Compress original 40-50 step inference to just 4 steps without CFG requirements
- **🛠️ Advanced Operator Support**: Integrated with cutting-edge operators including [Sage Attention](https://github.com/thu-ml/SageAttention), [Flash Attention](https://github.com/Dao-AILab/flash-attention), [Radial Attention](https://github.com/mit-han-lab/radial-attention), [q8-kernel](https://github.com/KONAKONA666/q8_kernels), [sgl-kernel](https://github.com/sgl-project/sglang/tree/main/sgl-kernel), [vllm](https://github.com/vllm-project/vllm)

### 💾 **Resource-Efficient Deployment**
- **💡 Breaking Hardware Barriers**: Run 14B models for 480P/720P video generation with only **8GB VRAM + 16GB RAM**
- **🔧 Intelligent Parameter Offloading**: Advanced disk-CPU-GPU three-tier offloading architecture with phase/block-level granular management
- **⚙️ Comprehensive Quantization**: Support for `w8a8-int8`, `w8a8-fp8`, `w4a4-nvfp4` and other quantization strategies

### 🎨 **Rich Feature Ecosystem**
- **📈 Smart Feature Caching**: Intelligent caching mechanisms to eliminate redundant computations
- **🔄 Parallel Inference**: Multi-GPU parallel processing for enhanced performance
- **📱 Flexible Deployment Options**: Support for Gradio, service deployment, ComfyUI and other deployment methods
- **🎛️ Dynamic Resolution Inference**: Adaptive resolution adjustment for optimal generation quality
- **🎞️ Video Frame Interpolation**: RIFE-based frame interpolation for smooth frame rate enhancement


## 📚 Technical Documentation

### 📖 **Method Tutorials**
- [Model Quantization](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/quantization.html) - Comprehensive guide to quantization strategies
- [Feature Caching](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/cache.html) - Intelligent caching mechanisms
- [Attention Mechanisms](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/attention.html) - State-of-the-art attention operators
- [Parameter Offloading](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/offload.html) - Three-tier storage architecture
- [Parallel Inference](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/parallel.html) - Multi-GPU acceleration strategies
- [Changing Resolution Inference](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/changing_resolution.html) - U-shaped resolution strategy
- [Step Distillation](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/step_distill.html) - 4-step inference technology
- [Video Frame Interpolation](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/video_frame_interpolation.html) - Base on the RIFE technology

### 🛠️ **Deployment Guides**
- [Low-Resource Deployment](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/for_low_resource.html) - Optimized 8GB VRAM solutions
- [Low-Latency Deployment](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/for_low_latency.html) - Ultra-fast inference optimization
- [Gradio Deployment](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/deploy_gradio.html) - Web interface setup
- [Service Deployment](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/deploy_service.html) - Production API service deployment
- [Lora Model Deployment](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/lora_deploy.html) - Flexible Lora deployment

## 🧾 Contributing Guidelines

We maintain code quality through automated pre-commit hooks to ensure consistent formatting across the project.

> [!TIP]
> **Setup Instructions:**
>
> 1. Install required dependencies:
> ```shell
> pip install ruff pre-commit
> ```
>
> 2. Run before committing:
> ```shell
> pre-commit run --all-files
> ```

We appreciate your contributions to making LightX2V better!

## 🤝 Acknowledgments

We extend our gratitude to all the model repositories and research communities that inspired and contributed to the development of LightX2V. This framework builds upon the collective efforts of the open-source community.

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ModelTC/lightx2v&type=Timeline)](https://star-history.com/#ModelTC/lightx2v&Timeline)

## ✏️ Citation

If you find LightX2V useful in your research, please consider citing our work:

```bibtex
@misc{lightx2v,
 author = {LightX2V Contributors},
 title = {LightX2V: Light Video Generation Inference Framework},
 year = {2025},
 publisher = {GitHub},
 journal = {GitHub repository},
 howpublished = {\url{https://github.com/ModelTC/lightx2v}},
}
```

## 📞 Contact & Support

For questions, suggestions, or support, please feel free to reach out through:
- 🐛 [GitHub Issues](https://github.com/ModelTC/lightx2v/issues) - Bug reports and feature requests

---

<div align="center">
Built with ❤️ by the LightX2V team
</div>

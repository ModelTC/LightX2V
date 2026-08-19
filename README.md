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

**LightX2V** is an advanced lightweight image/video generation inference framework engineered to deliver efficient, high-performance image/video synthesis solutions. This unified platform integrates multiple state-of-the-art image/video generation techniques, supporting diverse generation tasks including text-to-video (T2V), image-to-video (I2V), text-to-image (T2I), image-editing (I2I). **X2V represents the transformation of different input modalities (X, such as text or images) into vision output (Vision)**.

> 🌐 **Try it online now!** Experience LightX2V without installation: **[LightX2V Studio](https://x2v.light-ai.top/)** — a free, lightweight AI video platform with **LTX 2.3**, **Wan 2.2**, **SekoTalk**, **Qwen-Image**, **SeedVR2**, and more models and tasks.

> 🤗 **HuggingFace Model Repository: [LightX2V HuggingFace](https://huggingface.co/lightx2v)**

> 📝 **More content is available on our [LightX2V Blog](https://light-ai.top/LightX2V-BLOG/)**

> 🌟 **Developer Newbie Guide: [LightX2V Developer Quick Start Guide](https://github.com/ModelTC/LightX2V/tree/main/examples/BeginnerGuide)**

> 👋 **Join our WeChat group! LightX2V Robot WeChat ID: random42seed**

## 🧾 Community Code Contribution Guidelines

Before submitting, please ensure that the code format conforms to the project standard. You can use the following execution command to ensure the consistency of project code format.

```bash
pip install ruff pre-commit
pre-commit run --all-files
```

Besides the contributions from the LightX2V team, we have received contributions from some community developers, including but not limited to:

- [zhtshr](https://github.com/zhtshr)
- [triple-Mu](https://github.com/triple-Mu)
- [vivienfanghuagood](https://github.com/vivienfanghuagood)
- [yeahdongcn](https://github.com/yeahdongcn)
- [kikidouloveme79](https://github.com/kikidouloveme79)
- [ziyanxzy](https://github.com/ziyanxzy)
- [Tyr0727](https://github.com/Tyr0727)
- [hufangjian2017](https://github.com/hufangjian2017)
- [Fatemanx](https://github.com/Fatemanx)

## :fire: Latest News

- **August 11, 2026:** 🚀 We release and support the [MiniMax-H3 Turbo 4-step v1.0 768p distilled LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors). The released DMD configs under `configs/minimax_h3/dmd` run H3 at 1344x768 with `video_flow_shift=6`, `audio_flow_shift=3`, LoRA alpha 128, and 4-step guidance-free inference.

- **August 7, 2026:** 🚀 LightX2V introduces comprehensive inference support for [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), an omni-modal generative model that produces video with native synchronized stereo audio. The integration covers T2AV, I2AV, L2AV, FL2AV, and Ref2AV workflows, and incorporates model- and block-level offloading, tensor and sequence parallelism, quantized DiT inference, and feature caching. Single- and multi-GPU examples are available in the [MiniMax-H3 inference scripts](scripts/minimax_h3). Alongside this integration, we release the [MiniMax-H3 T2VA Prompt Rewriter LoRA](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA), fine-tuned from Qwen3.6-27B to transform concise user prompts into structured, H3-oriented multimodal descriptions spanning visual narrative, soundscape, and non-diegetic music.

- **July 28, 2026:** 🚀 We release the [LightLingBot-Video](https://huggingface.co/lightx2v/LightLingBot-Video) 4-step distilled LoRA for LingBot-Video. It supports T2V, T2I, and I2V generation in only 4 inference steps without CFG. See the [LingBot-Video inference scripts](scripts/lingbot_video) for usage.

- **July 23, 2026:** ⚡️ We release the blog: [LightX2V ROS: Closing the Loop for Action-Generating World Models](https://light-ai.top/LightX2V-BLOG/posts/LightX2V_ROS/)

- **July 19, 2026:** ⚡️ We release the blog: [Wan2.2-NVFP4-Sparse: Extremely Fast Wan 2.2 14B Inference](https://light-ai.top/LightX2V-BLOG/posts/Wan22-NVFP4-Sparse/)

- **June 15, 2026:** 🚀 Supported deployment on T-head PPU.

- **May 29, 2026:** 🚀 We introduce an extremely efficient Wan 2.2 14B variant (T2V and I2V): [NVFP4 Quantization-Aware Step Distillation with Sparse Attention for Blackwell Architecture](https://huggingface.co/lightx2v/Wan2.2-NVFP4-Sparse). On a single RTX 5090 GPU, it achieves over 50× speedup.

- **April 30, 2026:** 🚀 We now support deployment on iluvatar. Thanks to the iluvatar team.

- **April 20, 2026:** 🚀 We are excited to release the [Wan2.2-I2V-A14B-4step-720p-high](https://huggingface.co/lightx2v/Wan2.2-Distill-Models/blob/main/wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412.safetensors) and [Wan2.2-I2V-A14B-4step-720p-low](https://huggingface.co/lightx2v/Wan2.2-Distill-Models/blob/main/wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412.safetensors) models. Compared to previous iterations, this version was trained on a high-quality 720p dataset and features an optimized low-noise training algorithm. These enhancements significantly boost the model's performance in fine-grained detail rendering and visual texture.

- **April 17, 2026:** 🚀 We support the [WorldMirror 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0.git) model. On a single H100 GPU, LightX2V delivers approximately 1.2× speedup. For detailed usage instructions, please refer to [this guide](examples/worldmirror/README.md).

- **April 10, 2026:** 🎉 We released a technical blog on disaggregated deployment
  - [LightX2V Disaggregated Deployment: Breaking the Memory and Throughput Bottlenecks of Diffusion Model Inference](https://light-ai.top/LightX2V-BLOG/posts/Disaggregation/)

- **March 5, 2026:** 🚀 We now support deployment on Intel AIPC PTL. Thanks to the Intel team.

- **March 5, 2026:** 🚀 We now support disaggregated deployment based on [Mooncake](https://github.com/kvcache-ai/Mooncake). More improvements and documentation for disaggregated deployment are in progress. Thanks to the Mooncake team for their help!

- **February 27, 2026:** 🚀 We now support FP8 and NVFP4 quantization for autoregressive video generation models ([Self Forcing](https://github.com/guandeh17/Self-Forcing))! You can find the quantized model here: **[Self-Forcing-FP8](https://huggingface.co/lightx2v/Self-Forcing-FP8), [Self-Forcing-NVFP4](https://huggingface.co/lightx2v/Self-Forcing-NVFP4)**.

- **February 11, 2026:** 🎉 We are excited to announce **[GenRL](https://github.com/ModelTC/GenRL)** - a scalable reinforcement learning framework for visual generation! GenRL enables training diffusion/flow models with multi-reward optimization (HPSv3, VideoAlign, etc.) using GRPO algorithm. We've released high-performance LoRA checkpoints trained with multi-node multi-GPU setup, demonstrating significant improvements in aesthetic quality, motion coherence, and text-video alignment. Check out our [model collection](https://huggingface.co/collections/lightx2v/genrl) on HuggingFace! Give us a ⭐ if you find it useful!

- **January 20, 2026:** 🚀 We support the [LTX-2](https://huggingface.co/Lightricks/LTX-2) audio-video generation model, featuring CFG parallelism, block-level offload, and FP8 per-tensor quantization. Usage examples can be found in [examples/ltx2](https://github.com/ModelTC/LightX2V/tree/main/examples/ltx2) and [scripts/ltx2](https://github.com/ModelTC/LightX2V/tree/main/scripts/ltx2).

- **January 6, 2026:** 🚀 We updated the 8-step CFG/step-distilled models for [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) and [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511). You can download the corresponding weights from [Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) and [Qwen-Image-2512-Lightning](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning) for use. Usage tutorials can be found [here](https://github.com/ModelTC/LightX2V/tree/main/examples/qwen_image).

- **January 6, 2026:** 🚀 Supported deployment on Enflame S60 (GCU).

- **December 31, 2025:** 🚀 We support the [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) text-to-image model since Day 0. Our [HuggingFace](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning) has been updated with CFG / step-distilled LoRA. Usage examples can be found [here](https://github.com/ModelTC/LightX2V/tree/main/examples/qwen_image).

- **December 27, 2025:** 🚀 Supported deployment on MThreads MUSA.

- **December 25, 2025:** 🚀 Supported deployment on AMD ROCm and Ascend 910B.

- **December 23, 2025:** 🚀 We support the [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) image editing model since Day 0. On a single H100 GPU, LightX2V delivers approximately 1.4× speedup. We support for CFG parallelism, Ulysses parallelism, and efficient offloading technologies. Our [HuggingFace](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) has been updated with CFG / step-distilled LoRA and FP8 weights. Usage examples can be found [here](https://github.com/ModelTC/LightX2V/tree/main/examples/qwen_image). Combined with LightX2V, 4-step CFG / step distillation, and the FP8 model, the maximum acceleration can reach up to approximately 42×. Feel free to try [LightX2V Online Service](https://x2v.light-ai.top/login) with *Image to Image* and *Qwen-Image-Edit-2511* model.

- **December 22, 2025:** 🚀 Added **Wan2.1 NVFP4 quantization-aware 4-step distilled models**; weights are available on HuggingFace: [Wan-NVFP4](https://huggingface.co/lightx2v/Wan-NVFP4).

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
# examples/minimax_h3/minimax_h3_t2av_dmd.py
"""
MiniMax-H3 T2AV generation with the 4-step 768p distilled LoRA.
"""

from lightx2v import LightX2VPipeline

# Initialize the MiniMax-H3 T2AV pipeline.
pipe = LightX2VPipeline(
    model_path="/path/to/MiniMax-H3",
    model_cls="minimax_h3",
    task="t2av",
)

# The DMD config uses the released 768p LoRA, 4 inference steps,
# video_flow_shift=6, audio_flow_shift=3, and lora alpha=128.
pipe.create_generator(
    config_json="configs/minimax_h3/dmd/minimax_h3_bf16_4step_single_gpu_offload.json"
)

# Generation parameters
seed = 42
prompt = "A cinematic fox walks through a snowy forest while soft wind and distant birds create an immersive winter soundscape."
save_result_path = "outputs/minimax_h3_t2av_768p.mp4"

# Generate video with synchronized audio.
pipe.generate(
    seed=seed,
    prompt=prompt,
    save_result_path=save_result_path,
)
```

**NVFP4 (quantization-aware 4-step) resources**
- Inference examples: `examples/wan/wan_i2v_nvfp4.py` (I2V) and `examples/wan/wan_t2v_nvfp4.py` (T2V).
- NVFP4 operator build/install guide: see `lightx2v_kernel/README.md`.

> 💡 **More Examples**: For more usage examples including quantization, offloading, caching, and other advanced configurations, please refer to the [examples directory](https://github.com/ModelTC/LightX2V/tree/main/examples).



## 🤖 Supported Model Ecosystem

### Official Open-Source Models
- ✅ [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3)
- ✅ [LTX-2](https://huggingface.co/Lightricks/LTX-2)
- ✅ [HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5)
- ✅ [Wan2.1 & Wan2.2](https://huggingface.co/Wan-AI/)
- ✅ [SeedVR2](https://huggingface.co/ByteDance-Seed/SeedVR2-3B)
- ✅ [SwiftVR](https://huggingface.co/H-oliday/SwiftVR); convert the checkpoint with [convert_swiftvr.py](tools/convert/examples/convert_swiftvr.py), then run the [SR script](scripts/swiftvr/run_swiftvr_sr.sh).
- ✅ [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)
- ✅ [Qwen-Image-Edit](https://huggingface.co/spaces/Qwen/Qwen-Image-Edit)
- ✅ [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
- ✅ [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)

### Quantized and Distilled Models/LoRAs (**🚀 Recommended: 4-step inference**)
- ✅ [MiniMax-H3 Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) — 4-step 768p distilled LoRA for MiniMax-H3 T2AV/FL2AV
- ✅ [LightLingBot-Video](https://huggingface.co/lightx2v/LightLingBot-Video) — 4-step distilled LoRA for LingBot-Video T2V, T2I, and I2V
- ✅ [Wan2.1-Distill-Models](https://huggingface.co/lightx2v/Wan2.1-Distill-Models)
- ✅ [Wan2.2-Distill-Models](https://huggingface.co/lightx2v/Wan2.2-Distill-Models)
- ✅ [Wan2.1-Distill-Loras](https://huggingface.co/lightx2v/Wan2.1-Distill-Loras)
- ✅ [Wan2.2-Distill-Loras](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras)
- ✅ [Wan2.1-Distill-NVFP4](https://huggingface.co/lightx2v/Wan-NVFP4)
- ✅ [Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)

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

## 🤝 Acknowledgments

We sincerely thank all the model repositories and research communities that inspired and promoted the development of LightX2V. This framework is built on the collective efforts of the open-source community. It includes but is not limited to:

- [Tencent-Hunyuan](https://github.com/Tencent-Hunyuan)
- [Wan-Video](https://github.com/Wan-Video)
- [Qwen-Image](https://github.com/QwenLM/Qwen-Image)
- [LightLLM](https://github.com/ModelTC/LightLLM)
- [sglang](https://github.com/sgl-project/sglang)
- [vllm](https://github.com/vllm-project/vllm)
- [flash-attention](https://github.com/Dao-AILab/flash-attention)
- [SageAttention](https://github.com/thu-ml/SageAttention)
- [flashinfer](https://github.com/flashinfer-ai/flashinfer)
- [MagiAttention](https://github.com/SandAI-org/MagiAttention)
- [radial-attention](https://github.com/mit-han-lab/radial-attention)
- [xDiT](https://github.com/xdit-project/xDiT)
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [Mooncake](https://github.com/kvcache-ai/Mooncake)

We also thank the cloud inference platforms supporting the wider ecosystem:

- [Atlas Cloud](https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=lightx2v) — a full-modal AI inference platform whose hosted API serves the Wan / Seedance / Kling model families that LightX2V also supports, for teams that prefer a managed API over self-hosting.

- [Sensecore](https://www.sensecore.cn/about) —— Sensecore. Build a high-efficiency, low-cost, and scalable AI cloud infrastructure, create a professional deep learning platform and algorithm model system, lead AI innovation, and help industry and academia explore the boundaries of AI.

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

<div align="center" style="font-family: charter;">
  <h1>⚡️ LightX2V:<br> Light Video Generation Inference Framework</h1>

<img alt="logo" src="assets/img_lightx2v.png" width=75%></img>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Doc](https://img.shields.io/badge/docs-English-99cc2)](https://lightx2v-en.readthedocs.io/en/latest)
[![Doc](https://img.shields.io/badge/文档-中文-99cc2)](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://hub.docker.com/r/lightx2v/lightx2v/tags)

**\[ English | [中文](README_zh.md) \]**

</div>

--------------------------------------------------------------------------------

**LightX2V** is an advanced lightweight image/video generation inference framework engineered to deliver efficient, high-performance image/video synthesis solutions. This unified platform integrates multiple state-of-the-art image/video generation techniques, supporting diverse generation tasks including text-to-video (T2V), image-to-video (I2V), text-to-image (T2I), image-editing (I2I). **X2V represents the transformation of different input modalities (X, such as text or images) into vision output (Vision)**.

[Quick Start](#-quick-start) · [Models](#-supported-model-ecosystem) · [Documentation & Deployment](#-documentation--deployment)

> 🌐 **Try the LightX2V-powered AI avatar and multimodal content production platform online: [LightX2V Studio](https://x2v.light-ai.top/)**

> 🍎 **MacBook local inference app: [LightX2V APP](https://github.com/ModelTC/LightX2V-Mac-App)**

> 🤗 **HuggingFace Model Repository: [LightX2V HuggingFace](https://huggingface.co/lightx2v)**

> 🌟 **Developer Quick Start: [LightX2V Developer Quick Start Guide](docs/EN/source/getting_started/first_generation.md)**

> 📝 **Technical Blog: [LightX2V Blog](https://light-ai.top/LightX2V-BLOG/)**

> 👋 **Join our WeChat group! LightX2V Robot WeChat ID: Jo-1212liz**

## 🧾 Community Code Contribution Guidelines

Before submitting, please ensure that the code format conforms to the project standard. You can use the following execution command to ensure the consistency of project code format.

```bash
pip install ruff pre-commit
pre-commit run --all-files
```

We thank the following community developers for their contributions to LightX2V:

[zhtshr](https://github.com/zhtshr) · [triple-Mu](https://github.com/triple-Mu) · [vivienfanghuagood](https://github.com/vivienfanghuagood) · [yeahdongcn](https://github.com/yeahdongcn) · [kikidouloveme79](https://github.com/kikidouloveme79) · [ziyanxzy](https://github.com/ziyanxzy) · [Tyr0727](https://github.com/Tyr0727) · [hufangjian2017](https://github.com/hufangjian2017) · [Fatemanx](https://github.com/Fatemanx) · [qiuxin2012](https://github.com/qiuxin2012)

## :fire: Latest News

- **October 18, 2026:** 🚀 We now support [EdenLive](scripts/edenlive/README.md) for video generation controlled by text, a first-frame image, and a proxy recording. Wan2.1 14B and Wan2.2 5B each support bidirectional and four-step autoregressive inference, with realtime profiles for RTX 5090 and H100/H200. See the [deployment guide](scripts/edenlive/README.md) for model conversion, quantization, and streaming inference.

- **October 13, 2026:** 🎉 We released the [LightX2V APP](https://github.com/ModelTC/LightX2V-Mac-App), which supports local inference on MacBook with Qwen-Image-2.1 [Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo), [PE-T2I GGUF Q8_0](https://huggingface.co/lightx2v/Qwen-Image-2.1-PE-T2I-GGUF-Q8_0), and [PE-I2I GGUF Q8_0](https://huggingface.co/lightx2v/Qwen-Image-2.1-PE-I2I-GGUF-Q8_0) models.

- **September 22, 2026:** 🚀 We added and optimized support for [SwiftVR](https://huggingface.co/H-oliday/SwiftVR), covering image and video restoration and super-resolution. See the [usage guide](scripts/swiftvr/README.md).

- **September 20, 2026:** 🚀 We added day-0 support for [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1), covering text-to-image generation and prompt-based image editing. See the [usage guide](scripts/qwen_image_21/README.md).

- **August 27, 2026:** 🚀 We released the [MiniMax-H3 Turbo 8-step 768p distilled LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors) for generating 768p video with synchronized audio in 8 inference steps using MiniMax-H3. See the [MiniMax-H3 guide](scripts/minimax_h3/README.md).

- **August 11, 2026:** 🚀 We released the [MiniMax-H3 Turbo 4-step 768p distilled LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors) and added support for generating 768p video with synchronized audio in 4 inference steps without CFG. See the [MiniMax-H3 guide](scripts/minimax_h3/README.md).

<details>
<summary>Earlier updates</summary>

- **August 7, 2026:** 🚀 We added support for synchronized video and audio generation with [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) and released a [Prompt Rewriter LoRA](https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA) that expands short prompts into structured descriptions of visuals, soundscapes, and music. See the [MiniMax-H3 guide](scripts/minimax_h3/README.md).

- **July 28, 2026:** 🚀 We release the [LightLingBot-Video](https://huggingface.co/lightx2v/LightLingBot-Video) 4-step distilled LoRA for LingBot-Video. It supports T2V, T2I, and I2V generation in only 4 inference steps without CFG. See the [LingBot-Video inference scripts](scripts/lingbot_video) for usage.

- **July 23, 2026:** ⚡️ We release the blog: [LightX2V ROS: Closing the Loop for Action-Generating World Models](https://light-ai.top/LightX2V-BLOG/posts/LightX2V_ROS/)

- **July 19, 2026:** ⚡️ We release the blog: [Wan2.2-NVFP4-Sparse: Extremely Fast Wan 2.2 14B Inference](https://light-ai.top/LightX2V-BLOG/posts/LightWan22-A14B/)

- **June 15, 2026:** 🚀 Supported deployment on T-head PPU.

- **May 29, 2026:** 🚀 We introduce an extremely efficient Wan 2.2 14B variant (T2V and I2V): [NVFP4 Quantization-Aware Step Distillation with Sparse Attention for Blackwell Architecture](https://huggingface.co/lightx2v/Wan2.2-NVFP4-Sparse). Under the [test conditions in the model card](https://huggingface.co/lightx2v/LightWan2.2-A14B), inference with 4 NFE achieves over 50× end-to-end speedup over the original model with 40 NFE on a single RTX 5090 GPU.

- **April 30, 2026:** 🚀 We now support deployment on iluvatar. Thanks to the iluvatar team.

- **April 20, 2026:** 🚀 We are excited to release the [Wan2.2-I2V-A14B-4step-720p-high](https://huggingface.co/lightx2v/Wan2.2-Distill-Models/blob/main/wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412.safetensors) and [Wan2.2-I2V-A14B-4step-720p-low](https://huggingface.co/lightx2v/Wan2.2-Distill-Models/blob/main/wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412.safetensors) models. Compared to previous iterations, this version was trained on a high-quality 720p dataset and features an optimized low-noise training algorithm. These enhancements significantly boost the model's performance in fine-grained detail rendering and visual texture.

- **April 17, 2026:** 🚀 We support the [WorldMirror 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0.git) model. On a single H100 GPU, LightX2V delivers approximately 1.2× speedup. For detailed usage instructions, please refer to [this guide](scripts/worldmirror/README.md).

- **April 10, 2026:** 🎉 We released a technical blog on disaggregated deployment
  - [LightX2V Disaggregated Deployment: Breaking the Memory and Throughput Bottlenecks of Diffusion Model Inference](https://light-ai.top/LightX2V-BLOG/posts/Disaggregation/)

- **March 5, 2026:** 🚀 We now support deployment on Intel AIPC PTL. Thanks to the Intel team.

- **March 5, 2026:** 🚀 We now support disaggregated deployment based on [Mooncake](https://github.com/kvcache-ai/Mooncake). More improvements and documentation for disaggregated deployment are in progress. Thanks to the Mooncake team for their help!

- **February 27, 2026:** 🚀 We now support FP8 and NVFP4 quantization for autoregressive video generation models ([Self Forcing](https://github.com/guandeh17/Self-Forcing))! You can find the quantized model here: **[Self-Forcing-FP8](https://huggingface.co/lightx2v/Self-Forcing-FP8), [Self-Forcing-NVFP4](https://huggingface.co/lightx2v/Self-Forcing-NVFP4)**.

- **February 11, 2026:** 🎉 We are excited to announce **[GenRL](https://github.com/ModelTC/GenRL)** - a scalable reinforcement learning framework for visual generation! GenRL enables training diffusion/flow models with multi-reward optimization (HPSv3, VideoAlign, etc.) using GRPO algorithm. We've released high-performance LoRA checkpoints trained with multi-node multi-GPU setup, demonstrating significant improvements in aesthetic quality, motion coherence, and text-video alignment. Check out our [model collection](https://huggingface.co/collections/lightx2v/genrl) on HuggingFace! Give us a ⭐ if you find it useful!

- **January 20, 2026:** 🚀 We support the [LTX-2](https://huggingface.co/Lightricks/LTX-2) audio-video generation model, featuring CFG parallelism, block-level offload, and FP8 per-tensor quantization. Usage examples can be found in [scripts/ltx2/inference](https://github.com/ModelTC/LightX2V/tree/main/scripts/ltx2/inference) and [scripts/ltx2](https://github.com/ModelTC/LightX2V/tree/main/scripts/ltx2).

- **January 6, 2026:** 🚀 We updated the 8-step CFG/step-distilled models for [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) and [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511). You can download the corresponding weights from [Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) and [Qwen-Image-2512-Lightning](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning) for use. Usage tutorials can be found [here](scripts/qwen_image/README.md).

- **January 6, 2026:** 🚀 Supported deployment on Enflame S60 (GCU).

- **December 31, 2025:** 🚀 We support the [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) text-to-image model since Day 0. Our [HuggingFace](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning) has been updated with CFG / step-distilled LoRA. Usage examples can be found [here](scripts/qwen_image/README.md).

- **December 27, 2025:** 🚀 Supported deployment on MThreads MUSA.

- **December 25, 2025:** 🚀 Supported deployment on AMD ROCm and Ascend 910B.

- **December 23, 2025:** 🚀 We support the [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) image editing model since Day 0. On a single H100 GPU, LightX2V delivers approximately 1.4× speedup. We support for CFG parallelism, Ulysses parallelism, and efficient offloading technologies. Our [HuggingFace](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) has been updated with CFG / step-distilled LoRA and FP8 weights. Usage examples can be found [here](scripts/qwen_image/README.md). Combined with LightX2V, 4-step CFG / step distillation, and the FP8 model, the maximum acceleration can reach up to approximately 42×. Feel free to try [LightX2V Online Service](https://x2v.light-ai.top/login) with *Image to Image* and *Qwen-Image-Edit-2511* model.

- **December 22, 2025:** 🚀 Added **Wan2.1 NVFP4 quantization-aware 4-step distilled models**; weights are available on HuggingFace: [Wan-NVFP4](https://huggingface.co/lightx2v/Wan-NVFP4).

- **December 15, 2025:** 🚀 Supported deployment on Hygon DCU.

- **December 4, 2025:** 🚀 Supported GGUF format model inference & deployment on Cambricon MLU590/MetaX C500.

- **November 24, 2025:** 🚀 We released 4-step distilled models for HunyuanVideo-1.5! These models enable **ultra-fast 4-step inference** without CFG requirements, achieving approximately **25x speedup** compared to standard 50-step inference. Both base and FP8 quantized versions are now available: [Hy1.5-Distill-Models](https://huggingface.co/lightx2v/Hy1.5-Distill-Models).

- **November 21, 2025:** 🚀 We support the [HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5) video generation model since Day 0. With the same number of GPUs, LightX2V can achieve a speed improvement of over 2 times and supports deployment on GPUs with lower memory (such as the 24GB RTX 4090). It also supports CFG/Ulysses parallelism, efficient offloading, TeaCache/MagCache technologies, and more. We will soon update more models on our [HuggingFace page](https://huggingface.co/lightx2v), including step distillation, VAE distillation, and other related models. Quantized models and lightweight VAE models are now available: [Hy1.5-Quantized-Models](https://huggingface.co/lightx2v/Hy1.5-Quantized-Models) for quantized inference, and [LightTAE for HunyuanVideo-1.5](https://huggingface.co/lightx2v/Autoencoders/blob/main/lighttaehy1_5.safetensors) for fast VAE decoding. Refer to [this](https://github.com/ModelTC/LightX2V/tree/main/scripts/hunyuan_video_15) for usage tutorials, or check out the [Python API guide](docs/EN/source/getting_started/python_api.md) for code examples.

</details>

## 💡 Quick Start

We recommend using LightX2V through its service API. The examples below cover Qwen-Image-2.1 text-to-image (T2I) generation and MiniMax-H3 text-to-audio-video (T2AV) generation.

### Environment Setup

We recommend the [Docker environment](https://lightx2v-en.readthedocs.io/en/latest/getting_started/quickstart.html). Clone the repository in your prepared environment:

```bash
git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V
```

<details>
<summary>Other installation options (Git / source)</summary>

Run these commands in a prepared Python environment. The service example uses scripts from the cloned repository.

#### Installation from Git

```bash
pip install -v git+https://github.com/ModelTC/LightX2V.git
```

#### Installation from Source (Repository Root)

```bash
uv pip install -v . # or: pip install -v .
```

See the [environment setup guide](https://lightx2v-en.readthedocs.io/en/latest/getting_started/quickstart.html#step-4-install-attention-operators) for attention and quantization kernel installation.

</details>

### Usage Examples

Both servers use port `8000` by default. Stop the running server before switching examples.

#### Qwen-Image-2.1: Text-to-Image

Follow the [Qwen-Image-2.1 guide](scripts/qwen_image_21/README.md) to prepare the weights, then set the repository path, model path, and GPU in the server script. Run the commands below from the repository root.

```bash
# Terminal 1: start the server
bash scripts/qwen_image_21/server/start_server.sh
```

After the model loads, submit the example request from another terminal:

```bash
# Terminal 2: submit a T2I request
python scripts/qwen_image_21/server/post_t2i.py
```

Edit the prompt, size, and seed in the [request script](scripts/qwen_image_21/server/post_t2i.py) to generate your own images. Once the request completes, the image is saved by default to `save_results/qwen_image_21_t2i.png` under the repository root.

#### MiniMax-H3: Text-to-Audio-Video

Follow the [MiniMax-H3 guide](scripts/minimax_h3/README.md) to prepare the weights and initial AdaLN cache, and set the repository path, model path, and GPU in the server script. Run the commands below from the repository root.

```bash
# Terminal 1: start the server
bash scripts/minimax_h3/server/start_server.sh
```

After the model loads, submit the example request from another terminal:

```bash
# Terminal 2: submit a T2AV request
python scripts/minimax_h3/server/post_t2av.py
```

Edit the prompt in the [request script](scripts/minimax_h3/server/post_t2av.py) to generate your own content. Requests are asynchronous; results are saved by default to `save_results/server_cache/outputs/` under the repository root.

## 🤖 Supported Model Ecosystem

Models and variants are grouped by their primary use.

### Video and Audio-Video Generation

- ✅ [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) (including the causal variant)
- ✅ [LTX-2](https://huggingface.co/Lightricks/LTX-2) / [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) (including the autoregressive variant) / [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) (distilled)
- ✅ [HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5)
- ✅ [Wan2.1 / Wan2.2](https://huggingface.co/Wan-AI/)
- ✅ [Wan2.1-VACE](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B) / [Wan2.2-VACE-Fun-A14B](https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B)
- ✅ [Wan2.2-S2V](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)
- ✅ [Wan2.2-Animate](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) / [Wan2.2-Animate-2](https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B) (distilled)
- ✅ [LingBot-Video](https://huggingface.co/robbyant/lingbot-video-moe-30b-a3b) (dense / MoE)
- ✅ [SekoTalk](https://github.com/OpenSenseNova/SekoTalk) (including the autoregressive variant)
- ✅ [InfiniteTalk](https://huggingface.co/MeiGen-AI/InfiniteTalk)
- ✅ [Wan-Dancer](https://huggingface.co/Wan-AI/Wan-Dancer-14B)
- ✅ [Self-Forcing](https://github.com/guandeh17/Self-Forcing)

### Image Generation and Editing

- ✅ [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) / [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512)
- ✅ [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) / [2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) / [2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- ✅ [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) / [Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)
- ✅ [Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered)
- ✅ FLUX.2 [dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) / [klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) (including base and distilled variants)
- ✅ [HunyuanImage-3.0-Instruct](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct)
- ✅ [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) / [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- ✅ [LongCat-Image](https://huggingface.co/meituan-longcat/LongCat-Image) / [Edit](https://huggingface.co/meituan-longcat/LongCat-Image-Edit) / [Edit-Turbo](https://huggingface.co/meituan-longcat/LongCat-Image-Edit-Turbo)
- ✅ [BAGEL](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)
- ✅ [HiDream-O1-Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image) / [Dev-2604](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev-2604)
- ✅ [ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image) / [Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo)
- ✅ [SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1) (NeoPP backend, used with LightLLM)

### Restoration and Super-Resolution

- ✅ [SeedVR2](https://huggingface.co/ByteDance-Seed/SeedVR2-3B)
- ✅ [SwiftVR](https://huggingface.co/H-oliday/SwiftVR)

### World Models, Robotics, and 3D

- ✅ [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) / [Super](https://huggingface.co/nvidia/Cosmos3-Super) (including [Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID))
- ✅ [LingBot-World](https://huggingface.co/robbyant/lingbot-world-base-act-preview) / [Fast](https://huggingface.co/robbyant/lingbot-world-fast)
- ✅ [Matrix-Game-2.0](https://huggingface.co/Skywork/Matrix-Game-2.0) / [3.0](https://huggingface.co/Skywork/Matrix-Game-3.0)
- ✅ [WorldPlay](https://huggingface.co/tencent/HY-WorldPlay) (bidirectional, autoregressive, and distilled)
- ✅ [EdenLive](scripts/edenlive/README.md) (Wan2.1 14B / Wan2.2 5B; bidirectional and four-step autoregressive)
- ✅ [WorldMirror 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) (3D reconstruction)
- ✅ [DreamZero](https://huggingface.co/GEAR-Dreams/DreamZero-DROID)
- ✅ [LingBot-VA](https://huggingface.co/robbyant/lingbot-va-base)
- ✅ [Motus](https://huggingface.co/motus-robotics/Motus)
- ✅ [Fast-WAM](https://github.com/yuantianyuan01/FastWAM)
- ✅ [RealtimeWAM](https://huggingface.co/lightx2v/RealtimeWAM) (based on Fast-WAM / Faster-WAM)
- ✅ [Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) (image-to-3D shape generation)

### Quantized and Distilled Models/LoRAs

- ✅ [MiniMax-H3 Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) — 4-/8-step 768p distilled LoRAs for T2AV / FL2AV
- ✅ [LightLingBot-Video](https://huggingface.co/lightx2v/LightLingBot-Video) — 4-step distilled LoRA for T2V, T2I, and I2V
- ✅ [Wan2.1-Distill-Models](https://huggingface.co/lightx2v/Wan2.1-Distill-Models) / [Wan2.2-Distill-Models](https://huggingface.co/lightx2v/Wan2.2-Distill-Models)
- ✅ [Wan2.1-Distill-Loras](https://huggingface.co/lightx2v/Wan2.1-Distill-Loras) / [Wan2.2-Distill-Loras](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras)
- ✅ [Wan-NVFP4](https://huggingface.co/lightx2v/Wan-NVFP4) / [LightWan2.2-A14B](https://huggingface.co/lightx2v/LightWan2.2-A14B)
- ✅ [Hy1.5-Distill-Models](https://huggingface.co/lightx2v/Hy1.5-Distill-Models) / [Hy1.5-Quantized-Models](https://huggingface.co/lightx2v/Hy1.5-Quantized-Models)
- ✅ [Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning) / [Qwen-Image-2512-Lightning](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning) / [Qwen-Image-2.1-Lightning](https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning)
- ✅ [Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)
- ✅ [Z-Image-Turbo-Quantized](https://huggingface.co/lightx2v/Z-Image-Turbo-Quantized)
- ✅ [Self-Forcing-FP8](https://huggingface.co/lightx2v/Self-Forcing-FP8) / [Self-Forcing-NVFP4](https://huggingface.co/lightx2v/Self-Forcing-NVFP4)

### Lightweight Autoencoders

- ✅ [Autoencoders](https://huggingface.co/lightx2v/Autoencoders) (LightVAE / LightTAE)

## 🚀 Core Features

- **Few-Step Inference**: Support distilled checkpoints and distillation LoRAs, including DMD2 and Wan2.1 MeanFlow inference.
- **Quantization & Optimized Kernels**: Support W8A8 (INT8 / FP8) and W4A4 (NVFP4) quantization, with optimized attention backends such as FlashAttention and SageAttention.
- **Hierarchical Offloading**: Manage weights across disk, CPU, and GPU with model-level and block-level offloading.
- **Feature Caching**: Support TeaCache, MagCache, and TaylorSeer to reuse intermediate results across denoising steps.
- **Parallel Inference**: Support tensor parallelism (TP), sequence parallelism (SP), CFG parallelism, and hybrid configurations.
- **Variable-Resolution Inference**: Schedule latent resolutions across denoising stages to control computation at each stage.

Available optimizations depend on the model, hardware, and selected configuration.

## 📚 Documentation & Deployment

| Use Case | Documentation & Examples |
| --- | --- |
| Environment Setup & Configuration | [Quick Start Guide](https://lightx2v-en.readthedocs.io/en/latest/getting_started/quickstart.html) |
| Model Preparation & Loading | [MiniMax-H3](scripts/minimax_h3/README.md) · [Wan2.1 / VACE](scripts/wan/README.md) · [Model Formats & Loading](https://lightx2v-en.readthedocs.io/en/latest/getting_started/model_structure.html) |
| Service Deployment & API Usage | [Service Deployment](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/deploy_service.html) · [API Reference](lightx2v/server/README.md) |
| Interactive Interfaces & Workflow Integration | [Gradio-Lightx2v](https://github.com/ModelTC/Gradio-Lightx2v) · [ComfyUI](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/deploy_comfyui.html) |
| Local Deployment on Windows | [Windows Deployment Guide](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/deploy_local_windows.html) |
| Resource-Constrained and Low-Latency Deployment | [Low-Resource Deployment](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/for_low_resource.html) · [Low-Latency Deployment](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/for_low_latency.html) |
| Model Quantization & LoRA Deployment | [Quantization](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/quantization.html) · [LoRA Deployment](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/lora_deploy.html) |
| Inference Performance Optimization | [Weight Offloading](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/offload.html) · [Feature Caching](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/cache.html) · [Attention Kernels](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/attention.html) · [Parallel Inference](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/parallel.html) · [Variable-Resolution Inference](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/changing_resolution.html) · [Step Distillation](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/step_distill.html) |
| Wan2.1 NVFP4 4-Step Inference | [I2V Example](scripts/wan/inference/distill/wan_i2v_nvfp4.py) · [T2V Example](scripts/wan/inference/distill/wan_t2v_nvfp4.py) · [Quantization Kernel Installation](lightx2v_kernel/nvidia/README.md) |
| Code Examples & Development References | [Python API](docs/EN/source/getting_started/python_api.md) · [scripts](scripts) |

## 🤝 Acknowledgments

We thank the following open-source projects and communities for the models, tools, and research that have helped shape LightX2V:

[Tencent-Hunyuan](https://github.com/Tencent-Hunyuan) · [Wan-Video](https://github.com/Wan-Video) · [Qwen-Image](https://github.com/QwenLM/Qwen-Image) · [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) · [FlashAttention](https://github.com/Dao-AILab/flash-attention) · [SageAttention](https://github.com/thu-ml/SageAttention) · [LightLLM](https://github.com/ModelTC/LightLLM) · [sglang](https://github.com/sgl-project/sglang) · [vllm](https://github.com/vllm-project/vllm) · [flashinfer](https://github.com/flashinfer-ai/flashinfer) · [MagiAttention](https://github.com/SandAI-org/MagiAttention) · [radial-attention](https://github.com/mit-han-lab/radial-attention) · [xDiT](https://github.com/xdit-project/xDiT) · [FastVideo](https://github.com/hao-ai-lab/FastVideo) · [Mooncake](https://github.com/kvcache-ai/Mooncake)

We also thank the cloud inference platforms supporting the wider ecosystem:

- [Atlas Cloud](https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=lightx2v) — a full-modal AI inference platform whose hosted API serves the Wan / Seedance / Kling model families, for teams that prefer a managed API over self-hosting.

- [Sensecore](https://www.sensecore.cn/about) — Sensecore builds efficient, low-cost, scalable AI cloud infrastructure and develops deep learning platforms, algorithms, and models to support AI innovation in industry and academia.

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

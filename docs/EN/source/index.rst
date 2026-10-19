Welcome to Lightx2v!
==================

.. figure:: ../../../assets/img_lightx2v.png
  :width: 80%
  :align: center
  :alt: Lightx2v
  :class: no-scaled-link

.. raw:: html

    <div align="center" style="font-family: charter;">

    <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
    <a href="https://deepwiki.com/ModelTC/lightx2v"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
    <a href="https://lightx2v-en.readthedocs.io/en/latest"><img src="https://img.shields.io/badge/docs-English-99cc2" alt="Doc"></a>
    <a href="https://lightx2v-zhcn.readthedocs.io/zh-cn/latest"><img src="https://img.shields.io/badge/文档-中文-99cc2" alt="Doc"></a>
    <a href="https://hub.docker.com/r/lightx2v/lightx2v/tags"><img src="https://badgen.net/badge/icon/docker?icon=docker&label" alt="Docker"></a>

    </div>

    <div align="center" style="font-family: charter;">
    <strong>LightX2V: Light Video Generation Inference Framework</strong>
    </div>

LightX2V is a lightweight inference framework for image and video generation. It brings together inference acceleration techniques and supports text-to-video (T2V), image-to-video (I2V), text-to-image (T2I), and image editing (I2I) across multiple models. X2V means transforming different input modalities (X, such as text or images) into visual output (Vision).

GitHub: https://github.com/ModelTC/lightx2v

HuggingFace: https://huggingface.co/lightx2v

Documentation
-------------

.. toctree::
   :maxdepth: 1
   :caption: Quick Start

   Quick Start <getting_started/quickstart.md>
   First Video <getting_started/first_generation.md>
   Python API <getting_started/python_api.md>
   Model Structure <getting_started/model_structure.md>
   Benchmark <getting_started/benchmark.md>
   VGGT and TRELLIS.2 <getting_started/reconstruction.md>

.. toctree::
   :maxdepth: 1
   :caption: Method Tutorials

   Model Quantization <method_tutorials/quantization.md>
   Feature Caching <method_tutorials/cache.md>
   Attention Module <method_tutorials/attention.md>
   Offload <method_tutorials/offload.md>
   Parallel Inference <method_tutorials/parallel.md>
   Changing Resolution Inference <method_tutorials/changing_resolution.md>
   Step Distill <method_tutorials/step_distill.md>

.. toctree::
   :maxdepth: 1
   :caption: Deployment Guides

   Low Latency Deployment <deploy_guides/for_low_latency.md>
   Low Resource Deployment <deploy_guides/for_low_resource.md>
   Lora Deployment <deploy_guides/lora_deploy.md>
   Service Deployment <deploy_guides/deploy_service.md>
   Gradio Deployment <deploy_guides/deploy_gradio.md>
   ComfyUI Deployment <deploy_guides/deploy_comfyui.md>
   Local Windows Deployment <deploy_guides/deploy_local_windows.md>
   Disaggregated Deployment <deploy_guides/disagg.md>
   Qwen Image Text Encoder <deploy_guides/qwen_image_encoder.md>
   Qwen Image TensorRT VAE <deploy_guides/qwen_image_vae_tensorrt.md>

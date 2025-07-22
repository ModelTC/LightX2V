欢迎了解 Lightx2v!
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
    <strong>LightX2V: 一个轻量级的视频生成推理框架</strong>
    </div>


LightX2V 是一个轻量级的视频生成推理框架，集成多种先进的视频生成推理技术，统一支持 文本生成视频 (T2V)、图像生成视频 (I2V) 等多种生成任务及模型。X2V 表示将不同的输入模态（X，如文本或图像）转换（to）为视频输出（V）。

GitHub: https://github.com/ModelTC/lightx2v

HuggingFace: https://huggingface.co/lightx2v

文档列表
-------------

.. toctree::
   :maxdepth: 1
   :caption: 快速入门

   快速入门 <getting_started/quickstart.md>
   基准测试 <getting_started/benchmark.md>

.. toctree::
   :maxdepth: 1
   :caption: 方法教程

   模型量化 <method_tutorials/quantization.md>
   特征缓存 <method_tutorials/cache.md>
   注意力机制 <method_tutorials/attention.md>
   参数卸载 <method_tutorials/offload.md>
   并行推理 <method_tutorials/parallel.md>
   变分辨率推理 <method_tutorials/changing_resolution.md>
   步数蒸馏 <method_tutorials/step_distill.md>
   自回归蒸馏 <method_tutorials/autoregressive_distill.md>
   视频帧插值 <method_tutorials/video_frame_interpolation.md>

.. toctree::
   :maxdepth: 1
   :caption: 部署指南

   模型结构 <deploy_guides/model_structure.md>
   低延迟场景部署 <deploy_guides/for_low_latency.md>
   低资源场景部署 <deploy_guides/for_low_resource.md>
   Lora模型部署 <deploy_guides/lora_deploy.md>
   服务化部署 <deploy_guides/deploy_service.md>
   Gradio部署 <deploy_guides/deploy_gradio.md>
   ComfyUI部署 <deploy_guides/deploy_comfyui.md>
   本地windows电脑部署 <deploy_guides/deploy_local_windows.md>

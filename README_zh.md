<div align="center" style="font-family: charter;">
  <h1>⚡️ LightX2V:<br> 轻量级视频生成推理框架</h1>

<img alt="logo" src="assets/img_lightx2v.png" width=75%></img>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ModelTC/lightx2v)
[![Doc](https://img.shields.io/badge/docs-English-99cc2)](https://lightx2v-en.readthedocs.io/en/latest)
[![Doc](https://img.shields.io/badge/文档-中文-99cc2)](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest)
[![Papers](https://img.shields.io/badge/论文集-中文-99cc2)](https://lightx2v-papers-zhcn.readthedocs.io/zh-cn/latest)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://hub.docker.com/r/lightx2v/lightx2v/tags)

**\[ [English](README.md) | 中文 \]**

</div>

--------------------------------------------------------------------------------

**LightX2V** 是一个先进的轻量级视频生成推理框架，专为提供高效、高性能的视频合成解决方案而设计。该统一平台集成了多种前沿的视频生成技术，支持文本生成视频(T2V)和图像生成视频(I2V)等多样化生成任务。**X2V 表示将不同的输入模态(X，如文本或图像)转换为视频输出(V)**。

## :fire: 最新动态

- **2025年11月21日:** 🚀 我们Day0支持了HunyuanVideo1.5的视频生成模型，同样GPU数量，LightX2V可带来约2倍以上的速度提升，并支持更低显存GPU部署(如24G RTX4090)。支持CFG并行/Ulysses并行，高效Offload，TeaCache/MagCache等技术。同时支持沐曦，寒武纪等国产芯片部署。我们很快将在我们的[HuggingFace主页](https://huggingface.co/lightx2v)更新量化，步数蒸馏，VAE蒸馏等相关模型。


## 💡 快速开始

详细使用说明请参考我们的文档：**[英文文档](https://lightx2v-en.readthedocs.io/en/latest/) | [中文文档](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/)**

## 🤖 支持的模型生态

### 官方开源模型
- ✅ [HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5)
- ✅ [Wan2.1 & Wan2.2](https://huggingface.co/Wan-AI/)
- ✅ [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)
- ✅ [Qwen-Image-Edit](https://huggingface.co/spaces/Qwen/Qwen-Image-Edit)
- ✅ [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)

### 量化模型和蒸馏模型/Lora (**🚀 推荐：4步推理**)
- ✅ [Wan2.1-Distill-Models](https://huggingface.co/lightx2v/Wan2.1-Distill-Models)
- ✅ [Wan2.2-Distill-Models](https://huggingface.co/lightx2v/Wan2.2-Distill-Models)
- ✅ [Wan2.1-Distill-Loras](https://huggingface.co/lightx2v/Wan2.1-Distill-Loras)
- ✅ [Wan2.2-Distill-Loras](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras)

🔔 可以关注我们的[HuggingFace主页](https://huggingface.co/lightx2v)，及时获取我们团队的模型。

### 自回归模型
- ✅ [Wan2.1-T2V-CausVid](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-CausVid)
- ✅ [Self-Forcing](https://github.com/guandeh17/Self-Forcing)
- ✅ [Matrix-Game-2.0](https://huggingface.co/Skywork/Matrix-Game-2.0)

### 轻量级自编码器模型(**🚀 推荐：推理快速 + 内存占用低**)
- ✅ [Autoencoders](https://huggingface.co/lightx2v/Autoencoders)

💡 参考[模型结构文档](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/getting_started/model_structure.html)快速上手 LightX2V

## 🚀 前端展示

我们提供了多种前端界面部署方式：

- **🎨 Gradio界面**: 简洁易用的Web界面，适合快速体验和原型开发
  - 📖 [Gradio部署文档](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/deploy_guides/deploy_gradio.html)
- **🎯 ComfyUI界面**: 强大的节点式工作流界面，支持复杂的视频生成任务
  - 📖 [ComfyUI部署文档](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/deploy_guides/deploy_comfyui.html)
- **🚀 Windows一键部署**: 专为Windows用户设计的便捷部署方案，支持自动环境配置和智能参数优化
  - 📖 [Windows一键部署文档](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/deploy_guides/deploy_local_windows.html)

**💡 推荐方案**:
- **首次使用**: 建议选择Windows一键部署方案
- **高级用户**: 推荐使用ComfyUI界面获得更多自定义选项
- **快速体验**: Gradio界面提供最直观的操作体验

## 🚀 核心特性

### 🎯 **极致性能优化**
- **🔥 SOTA推理速度**: 通过步数蒸馏和系统优化实现**20倍**极速加速(单GPU)
- **⚡️ 革命性4步蒸馏**: 将原始40-50步推理压缩至仅需4步，且无需CFG配置
- **🛠️ 先进算子支持**: 集成顶尖算子，包括[Sage Attention](https://github.com/thu-ml/SageAttention)、[Flash Attention](https://github.com/Dao-AILab/flash-attention)、[Radial Attention](https://github.com/mit-han-lab/radial-attention)、[q8-kernel](https://github.com/KONAKONA666/q8_kernels)、[sgl-kernel](https://github.com/sgl-project/sglang/tree/main/sgl-kernel)、[vllm](https://github.com/vllm-project/vllm)

### 💾 **资源高效部署**
- **💡 突破硬件限制**: **仅需8GB显存 + 16GB内存**即可运行14B模型生成480P/720P视频
- **🔧 智能参数卸载**: 先进的磁盘-CPU-GPU三级卸载架构，支持阶段/块级别的精细化管理
- **⚙️ 全面量化支持**: 支持`w8a8-int8`、`w8a8-fp8`、`w4a4-nvfp4`等多种量化策略

### 🎨 **丰富功能生态**
- **📈 智能特征缓存**: 智能缓存机制，消除冗余计算，提升效率
- **🔄 并行推理加速**: 多GPU并行处理，显著提升性能表现
- **📱 灵活部署选择**: 支持Gradio、服务化部署、ComfyUI等多种部署方式
- **🎛️ 动态分辨率推理**: 自适应分辨率调整，优化生成质量
- **🎞️ 视频帧插值**: 基于RIFE的帧插值技术，实现流畅的帧率提升


## 🏆 性能基准测试

详细的性能指标和对比分析，请参考我们的[基准测试文档](https://github.com/ModelTC/LightX2V/blob/main/docs/ZH_CN/source/getting_started/benchmark_source.md)。

[详细服务部署指南 →](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/deploy_guides/deploy_service.html)

## 📚 技术文档

### 📖 **方法教程**
- [模型量化](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/quantization.html) - 量化策略全面指南
- [特征缓存](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/cache.html) - 智能缓存机制详解
- [注意力机制](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/attention.html) - 前沿注意力算子
- [参数卸载](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/offload.html) - 三级存储架构
- [并行推理](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/parallel.html) - 多GPU加速策略
- [变分辨率推理](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/changing_resolution.html) - U型分辨率策略
- [步数蒸馏](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/step_distill.html) - 4步推理技术
- [视频帧插值](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/video_frame_interpolation.html) - 基于RIFE的帧插值技术

### 🛠️ **部署指南**
- [低资源场景部署](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/deploy_guides/for_low_resource.html) - 优化的8GB显存解决方案
- [低延迟场景部署](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/deploy_guides/for_low_latency.html) - 极速推理优化
- [Gradio部署](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/deploy_guides/deploy_gradio.html) - Web界面搭建
- [服务化部署](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/deploy_guides/deploy_service.html) - 生产级API服务部署
- [Lora模型部署](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/deploy_guides/lora_deploy.html) - Lora灵活部署

## 🧾 代码贡献指南

我们通过自动化的预提交钩子来保证代码质量，确保项目代码格式的一致性。

> [!TIP]
> **安装说明：**
>
> 1. 安装必要的依赖：
> ```shell
> pip install ruff pre-commit
> ```
>
> 2. 提交前运行：
> ```shell
> pre-commit run --all-files
> ```

感谢您为LightX2V的改进做出贡献！

## 🤝 致谢

我们向所有启发和促进LightX2V开发的模型仓库和研究社区表示诚挚的感谢。此框架基于开源社区的集体努力而构建。

## 🌟 Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=ModelTC/lightx2v&type=Timeline)](https://star-history.com/#ModelTC/lightx2v&Timeline)

## ✏️ 引用

如果您发现LightX2V对您的研究有用，请考虑引用我们的工作：

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

## 📞 联系与支持

如有任何问题、建议或需要支持，欢迎通过以下方式联系我们：
- 🐛 [GitHub Issues](https://github.com/ModelTC/lightx2v/issues) - 错误报告和功能请求
- 💬 [GitHub Discussions](https://github.com/ModelTC/lightx2v/discussions) - 社区讨论和问答

---

<div align="center">
由 LightX2V 团队用 ❤️ 构建
</div>


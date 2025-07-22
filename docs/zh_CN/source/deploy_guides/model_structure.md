# 模型结构介绍

## 📖 概述

本文档介绍 Lightx2v 项目的模型目录结构，帮助用户正确组织模型文件，实现便捷的使用体验。通过合理的目录组织，用户可以享受到"一键启动"的便利，无需手动配置复杂的路径参数。

## 🗂️ 模型目录结构

### Lightx2v官方模型列表

查看所有可用模型：[Lightx2v官方模型仓库](https://huggingface.co/lightx2v)


### 标准目录结构

以 `Wan2.1-I2V-14B-480P-Lightx2v` 为例：

```
模型根目录/
├── Wan2.1-I2V-14B-480P-Lightx2v/
│   ├── config.json                                    # 模型配置文件
│   ├── Wan2.1_VAE.pth                                # VAE变分自编码器
│   ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth  # CLIP视觉编码器 (FP16)
│   ├── models_t5_umt5-xxl-enc-bf16.pth               # T5文本编码器 (BF16)
│   ├── taew2_1.pth                                   # 轻量级VAE (可选)
│   ├── fp8/                                          # FP8量化版本 (DIT/T5/CLIP)
│   ├── int8/                                         # INT8量化版本 (DIT/T5/CLIP)
│   ├── original/                                     # 原始精度版本 (DIT)
│   ├── xlm-roberta-large/
│   └── google/
```

### 💾 存储建议

**强烈建议将模型文件存储在SSD固态硬盘上**，可以显著提升模型加载速度和推理性能。

**推荐存储路径**：
```bash
/mnt/ssd/models/          # 独立SSD挂载点
/data/ssd/models/         # 数据SSD目录
/opt/models/              # 系统优化目录
```

## 🔧 模型文件说明


### 量化版本目录

每个模型都包含多个量化版本，用于不同硬件配置：

```
模型目录/
├── fp8/                         # FP8量化版本 (H100/A100等高端GPU)
├── int8/                        # INT8量化版本 (通用GPU)
└── original/                    # 原始精度版本 (DIT)
```

**💡 使用全精度模型**：如需使用全精度模型，只需将官方权重文件复制到 `original/` 目录即可。

## 🚀 使用方法

### Gradio界面启动

使用Gradio界面时，只需指定模型根目录路径：

```bash
# 图像到视频 (I2V)
python gradio_demo_zh.py \
    --model_path /path/to/Wan2.1-I2V-14B-480P-Lightx2v \
    --model_size 14b \
    --task i2v

# 文本到视频 (T2V)
python gradio_demo_zh.py \
    --model_path /path/to/models/Wan2.1-T2V-14B-Lightx2v \
    --model_size 14b \
    --task t2v
```

### 配置文件启动

使用配置文件启动时, 如[配置文件](https://github.com/ModelTC/LightX2V/tree/main/configs/offload/disk/wan_i2v_phase_lazy_load_480p.json)中的以下路径配置可以省略：

- `dit_quantized_ckpt` 无需指定，代码会自动在模型目录下查找
- `tiny_vae_path`：无需指定，代码会自动在模型目录下查找
- `clip_quantized_ckpt`：无需指定，代码会自动在模型目录下查找
- `t5_quantized_ckpt`：无需指定，代码会自动在模型目录下查找

**💡 简化配置**：按照推荐的目录结构组织模型文件后，大部分路径配置都可以省略，代码会自动处理。


### 手动下载

1. 访问 [Hugging Face模型页面](https://huggingface.co/lightx2v)
2. 选择需要的模型版本
3. 下载所有文件到对应目录

**💡 下载建议**：建议使用SSD存储，并确保网络连接稳定。对于大文件，可使用 `git lfs` 或下载工具如 `aria2c`。



## 💡 最佳实践

- **使用SSD存储**：显著提升模型加载速度和推理性能
- **统一目录结构**：便于管理和切换不同模型版本
- **预留足够空间**：确保有足够的存储空间（建议至少200GB）
- **定期清理**：删除不需要的模型版本以节省空间
- **网络优化**：使用稳定的网络连接和下载工具

## 🚨 常见问题

### Q: 模型文件太大，下载很慢怎么办？
A: 使用国内镜像源、下载工具如 `aria2c`，或考虑使用云存储服务

### Q: 启动时提示模型路径不存在？
A: 检查模型是否已正确下载，验证路径配置是否正确

### Q: 如何切换不同的模型版本？
A: 修改启动命令中的模型路径参数，支持同时运行多个模型实例

### Q: 模型加载速度很慢？
A: 确保模型存储在SSD上，启用延迟加载功能，使用量化版本模型

### Q: 配置文件中的路径如何设置？
A: 按照推荐目录结构组织后，大部分路径配置可省略，代码会自动处理

## 📚 相关链接

- [Lightx2v官方模型仓库](https://huggingface.co/lightx2v)
- [Gradio部署指南](./deploy_gradio.md)

---

通过合理的模型文件组织，用户可以享受到"一键启动"的便捷体验，无需手动配置复杂的路径参数。建议按照本文档的推荐结构组织模型文件，并充分利用SSD存储的优势。

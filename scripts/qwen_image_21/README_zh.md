# Qwen-Image-2.1 部署指南

[English](README.md) | [简体中文](README_zh.md)

## 1. 模型下载

官方模型权重：[Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)。下载命令：

```bash
hf download Qwen/Qwen-Image-2.1 --local-dir Qwen/Qwen-Image-2.1
```

## 2. 环境安装

根据 GPU 型号选择 [LightX2V Docker 镜像](https://hub.docker.com/r/lightx2v/lightx2v/tags)。在宿主机的工作目录中拉取对应镜像：

```bash
# H100, A100等
docker pull lightx2v/lightx2v:26062001

# RTX 5090
docker pull lightx2v/lightx2v:26062001-cu130-5090-fix-260921
```

使用所选镜像启动容器：

```bash
docker run --gpus all -itd --ipc=host --name [容器名] -v [挂载设置] --entrypoint /bin/bash [镜像id]
```

在容器内下载 LightX2V 源码：

```bash
git clone https://github.com/ModelTC/LightX2V.git
```

后续命令均在容器内的 LightX2V 仓库根目录执行。

## 3. 离线推理

### 3.1 通用案例

先修改对应 bash 脚本中的 `lightx2v_path`（仓库路径）、`model_path`（模型根目录）和 `CUDA_VISIBLE_DEVICES`（GPU 编号）。

```bash
# 文生图
bash scripts/qwen_image_21/qwen_image_21_t2i.sh

# 图生图
bash scripts/qwen_image_21/qwen_image_21_i2i.sh
```

直接在脚本中修改提示词 `--prompt`、参考图 `--image_path`、尺寸 `--size`、种子 `--seed` 和输出路径 `--save_result_path`。

使用 `--size 高 宽`。宽高均须不小于 32，非 32 倍数会自动向下对齐。

图生图可删除脚本中的 `--size` 参数，按参考图比例自动确定输出尺寸。总像素数约为配置中的 `resolution²`（`resolution` 默认为 `1024`），宽高就近对齐到 32 的倍数。

默认配置已经启用通用优化路径：FlashAttention3、FlashInfer RoPE、Triton LayerNorm 与 modulation、融合 QK RMSNorm、融合 Transformer block 算子、关闭 CFG 的推理路径及 OpenCV 结果保存。

### 3.2 RTX 5090 案例

示例采用的优化技术包括：

- DiT FP8 linear 与 FP16 累加、QwenVL 语言栈 FP8。
- SageAttention2、FlashInfer RoPE。
- Triton LayerNorm/modulation、融合 QK RMSNorm 与 Transformer block 算子。
- 文本编码器 offload、VAE torch.compile。
- Ulysses SP、FP8 通信、分组 head parallel、文本 TP 和 VAE 并行。

转换 DiT 权重：

```bash
python tools/convert/converter.py \
    --source /path/to/Qwen-Image-2.1/transformer \
    --output /path/to/Qwen-Image-2.1-fp8-f16-accum \
    --output_name qwen_image_21_fp8_f16_accum \
    --model_type qwen_image_21_dit \
    --quantization_profile qwen-image-21-fp8-f16-accum \
    --quantized --linear_type fp8 --device cuda:0 --single_file
```

QwenVL FP8 转换命令见 [converter.py](../../tools/convert/converter.py) 中 `convert_qwen_image_21_text_encoder_fp8` 的示例。在对应 JSON 中设置 `dit_quantized_ckpt`、`text_encoder_quantized_ckpt`；在脚本中设置仓库路径、原始模型路径和 GPU 编号。

```bash
# 单卡 1K：文生图 / 图生图
bash scripts/qwen_image_21/qwen_image_21_t2i_5090_1k.sh
bash scripts/qwen_image_21/qwen_image_21_i2i_5090_1k.sh

# 单卡 2K：文生图 / 图生图
bash scripts/qwen_image_21/qwen_image_21_t2i_5090_2k.sh
bash scripts/qwen_image_21/qwen_image_21_i2i_5090_2k.sh

# 双卡 2K：文生图 / 图生图
bash scripts/qwen_image_21/qwen_image_21_t2i_5090_2k_sp2.sh
bash scripts/qwen_image_21/qwen_image_21_i2i_5090_2k_sp2.sh
```

| GPU | 任务 | 输出分辨率 | 端到端耗时 |
| --- | --- | ---: | ---: |
| RTX 5090 ×1 | T2I | 1024×1024 | **5.586 s** |
| RTX 5090 ×1 | I2I | 1024×1024 | **6.804 s** |
| RTX 5090 ×1 | T2I | 2048×2048 | **30.333 s** |
| RTX 5090 ×1 | I2I | 2048×2048 | **42.787 s** |
| RTX 5090 ×2 | T2I | 2048×2048 | **16.868 s** |
| RTX 5090 ×2 | I2I | 2048×2048 | **25.212 s** |

测试使用 PyTorch 2.11.0+cu130，40 步、seed 42、关闭 CFG，提示词与示例脚本一致。I2I 使用 `assets/inputs/imgs/girl.png`，参考图按对应分辨率处理。

同一实例中首次请求后，连续三次请求取中位数；双卡每次取较慢 rank 的耗时。端到端耗时包含图片保存，不包含模型加载和一次性初始化。

运行时在日志中搜索 **`RUN pipeline cost`**，查看单次 pipeline 耗时，单位为秒。

## 4. 服务化部署与 API 调用

先修改 `server/start_server.sh` 中的仓库路径、模型路径和 GPU 编号，然后启动：

```bash
bash scripts/qwen_image_21/server/start_server.sh
```

默认端口为 `8000`，同一个服务支持文生图和图生图。

服务启动完成后，在同一容器新开终端并发送请求：

```bash
# 文生图
python scripts/qwen_image_21/server/post_t2i.py

# 图生图
python scripts/qwen_image_21/server/post_i2i.py
```

直接修改代码中的 `url`、`message`、`output_path`；图生图还需设置 `image_path`。两个脚本已分别设置 `task: "t2i"` 和 `task: "i2i"`。

服务请求使用相同的尺寸规则。图生图需自动确定尺寸时，删除 `message` 中的 `size` 字段。

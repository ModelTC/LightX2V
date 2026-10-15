# Qwen Image

[English](README.md) | [简体中文](README_zh.md)

设置所选脚本中的 `lightx2v_path`、`model_path` 和 `CUDA_VISIBLE_DEVICES`。以下命令使用仓库相对路径，也可使用 sh 的绝对路径。

## 基础任务

| 任务 | 权重 | 脚本 |
| --- | --- | --- |
| 文生图 | Qwen-Image-2512 | [qwen_image_t2i_2512.sh](qwen_image_t2i_2512.sh) |
| 图像编辑 | Qwen-Image-Edit-2511 | [qwen_image_i2i_2511.sh](qwen_image_i2i_2511.sh) |
| 图像分层 | Qwen-Image-Layered | [qwen_image_i2i_layered.sh](qwen_image_i2i_layered.sh) |

```bash
bash scripts/qwen_image/qwen_image_t2i_2512.sh
```

在命令中设置提示词、输入图片、seed 和输出路径，通过 `--config_json` 选择 [配置](../../configs/qwen_image)。Layered 脚本还需设置其中的 Diffusers 源码路径。

图像编辑支持多个输入，按顺序对应提示词中的 Image 1、Image 2 等：

```bash
--image_path "/path/to/1.png,/path/to/2.png,/path/to/3.png"
```

## 服务与请求

| 服务 | 启动脚本 |
| --- | --- |
| 单卡文生图 | [start_server_t2i.sh](../server/start_server_t2i.sh) |
| 单卡图像编辑 | [start_server_i2i.sh](../server/start_server_i2i.sh) |
| 八卡图像编辑（CFG2 × SP4） | [start_server_i2i_dist.sh](server/start_server_i2i_dist.sh) |

例如，启动文生图服务后，在另一个终端发送请求：

```bash
bash scripts/server/start_server_t2i.sh
```

```bash
python scripts/server/post_sync_t2i_base64.py --prompt "A cat in a garden" --output result.png
```

图像编辑服务使用以下客户端：

```bash
python scripts/server/post_sync_i2i_base64.py --prompt "Turn the image into a watercolor painting" --image_path /path/to/image.png --output result.png
```

客户端默认连接 `http://127.0.0.1:8000`，可用 `--url` 修改。图片客户端将本地图片转为 Base64 上传；自行发送 POST 时，也可将 `image_path` 设为服务端可访问的绝对路径。

## 多卡

| 示例 | 并行方式 | 脚本 |
| --- | --- | --- |
| 图像编辑 | 八卡 CFG2 × SP4 | [qwen_image_i2i_2511_dist.sh](qwen_image_i2i_2511_dist.sh) |
| Lightning 文生图 | 四卡 SP4 | [qwen_image_t2i_2512_distill_dist.sh](inference/distill/qwen_image_t2i_2512_distill_dist.sh) |

调整并行配置时，同步设置 `CUDA_VISIBLE_DEVICES` 和 `--nproc_per_node`；进程数为 `cfg_p_size × seq_p_size`，未设置的维度按 1 计算。

## 进阶功能

### 编译与复用

文生图编译使用 [qwen_image_t2i_2512_compile.json](../../configs/qwen_image/qwen_image_t2i_2512_compile.json)。图像编辑可在基础 JSON 中添加：

```json
{
    "use_compile": true,
    "rms_norm_type": "sgl-kernel"
}
```

图像编辑的编码结果复用示例：

```bash
bash scripts/qwen_image/server/start_server_i2i_reuse.sh
```

设置客户端的提示词和图片路径后，在另一个终端运行：

```bash
python scripts/qwen_image/server/post_i2i_reuse.py
```

首次请求设置 `reuse=False`，复用请求设为 `True`；保持提示词、图片路径和内容不变，可修改 seed。该服务示例同时启用了编译。

文生图可在上述编译配置中添加以下字段，并在请求中设置 `reuse`。不同模型和任务使用独立缓存目录：

```json
{
    "enable_reuse": true,
    "reuse_cache_path": "save_results/reuse_cache/qwen_image_t2i_2512"
}
```

### 蒸馏与 FP8

| 示例 | 脚本 | 可选 FP8 配置 |
| --- | --- | --- |
| Lightning 文生图，8 步 | [qwen_image_t2i_2512_distill.sh](inference/distill/qwen_image_t2i_2512_distill.sh) | [qwen_image_t2i_2512_distill_fp8.json](../../configs/qwen_image/qwen_image_t2i_2512_distill_fp8.json) |
| Lightning 图像编辑，8 步 | [qwen_image_i2i_2511_distill.sh](inference/distill/qwen_image_i2i_2511_distill.sh) | [qwen_image_i2i_2511_distill_fp8.json](../../configs/qwen_image/qwen_image_i2i_2511_distill_fp8.json) |
| Zoe 文生图，4 步 | [qwen_image_t2i_2512_distill_zoe.sh](inference/distill/qwen_image_t2i_2512_distill_zoe.sh) | [qwen_image_t2i_2512_distill_zoe_fp8.json](../../configs/qwen_image/qwen_image_t2i_2512_distill_zoe_fp8.json) |

脚本默认使用 BF16 LoRA 或蒸馏权重。FP8 使用同一脚本，将 `--config_json` 切换到对应配置。`model_path` 保持为完整基础模型目录，按所选 JSON 设置 `lora_configs`、`dit_original_ckpt` 或 `dit_quantized_ckpt` 中的本地权重路径。`enable_cfg: false` 的配置不传 `negative_prompt`，包括空字符串。

### 其他加速方式

| 功能 | 用法 |
| --- | --- |
| TensorRT VAE | 在基础脚本中选择 [T2I 配置](../../configs/qwen_image/qwen_image_t2i_2512_trt.json) 或 [I2I 配置](../../configs/qwen_image/qwen_image_i2i_2511_trt.json)，设置 `trt_vae_config.trt_engine_path`。两者分别使用静态与多 profile 引擎。 |
| LightLLM 文本编码器 | [kernel 脚本](inference/text_encoder/qwen_image_i2i_2511_kernel.sh) 或 [service 脚本](inference/text_encoder/qwen_image_i2i_2511_service.sh)。service 模式需先启动 LightLLM 服务并设置 `lightllm_config.service_url`，见[部署说明](../../examples/BeginnerGuide/ZH_CN/QwenImageEncoderSplitDeploy.md)。 |
| 共享 CPU offload | [使用说明](inference/shared_offload/README_CN.md)，配置位于 `configs/qwen_image/shared_offload/`。 |
| baseline / contiguous 布局 | [使用说明](inference/offload_layout/README_CN.md)，提供 GPU 和 NPU 示例。 |

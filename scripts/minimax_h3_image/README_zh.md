# MiniMax-H3-Image

[English](README.md) | [简体中文](README_zh.md)

支持文生图（`t2i`）和参考图生成／编辑（`i2i`），共用 MiniMax-H3-Image 权重和六步 PDD 调度。`i2i` 支持单张及多张参考图。

## 运行前准备

模型权重：[MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)。

先安装 LightX2V 及对应设备的依赖。Qwen3-VL 视觉编码器要求 `transformers>=5.15.0`，需要包含 `get_vision_interpolation_indices_and_weights` 接口。

下文命令均从仓库根目录执行。新终端先设置路径并加载基础环境：

```bash
lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3-Image

cd "$lightx2v_path"
source "$lightx2v_path/scripts/base/base.sh"
```

`model_path` 指向完整的模型根目录，包含 Transformer、Qwen3-VL、VAE、Tokenizer／Processor，以及 `pdd_config.json`、`pdd_fused_heads.safetensors`。

使用推理脚本、服务启动脚本或缓存生成脚本前，先修改脚本内的 `lightx2v_path`、`model_path` 和 `CUDA_VISIBLE_DEVICES`。

## AdaLN 缓存

**示例配置默认启用 AdaLN 缓存；首次推理或启动服务前，需要先生成匹配的缓存。已有匹配缓存时可跳过。**

```bash
bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_image_adaln.sh
```

[缓存生成脚本](../../tools/cache_minimax_h3_adaln/run_cache_minimax_h3_image_adaln.sh) 只需单卡。日志出现 `AdaLN cache saved to` 表示生成完成。推理和服务会根据模型与调度自动查找缓存。

默认配置为：

```json
{
  "use_adaln_cache": true,
  "adaln_cache_dir": "~/.cache/lightx2v/adaln"
}
```

当前模型的默认缓存目录为：

```text
~/.cache/lightx2v/adaln/minimax_h3_image/
└── image_06steps_1.000_0.9675_0.8875_0.7500_0.5328_0.2320/
    ├── manifest.json
    └── adaln_cache.safetensors
```

目录名包含推理步数和每一步的 sigma，每个 sigma 保留四位有效数字，包含末尾零；调度末尾用于结束积分的 `0.0` 不拼入目录名。`~` 为运行进程所属用户的主目录。

相同模型和调度的缓存可在 T2I、I2I、不同提示词、参考图、seed、输出尺寸及 TP／SP 配置间复用。更改调度时间步或生成缓存所用的 AdaLN 相关权重后，需要重新生成。`manifest.json` 校验精确时间步及缓存结构，不记录或检查 checkpoint 文件指纹、PDD 配置哈希和运行环境信息。

生成工具遇到同名目录会报 `FileExistsError`；需要重建时，先移走旧缓存目录，或修改 `adaln_cache_dir`，并让生成缓存与推理使用相同的缓存根目录。

## CLI 离线推理

### 使用示例脚本

| 任务 | 单卡 | 四卡 SP4 | 四卡 TP2 × SP2 |
| --- | --- | --- | --- |
| 文生图 | [run_minimax_h3_image_t2i.sh](run_minimax_h3_image_t2i.sh) | [run_minimax_h3_image_t2i_sp4.sh](run_minimax_h3_image_t2i_sp4.sh) | [run_minimax_h3_image_t2i_tp_sp.sh](run_minimax_h3_image_t2i_tp_sp.sh) |
| 参考图生成／编辑 | [run_minimax_h3_image_i2i.sh](run_minimax_h3_image_i2i.sh) | [run_minimax_h3_image_i2i_sp4.sh](run_minimax_h3_image_i2i_sp4.sh) | [run_minimax_h3_image_i2i_tp_sp.sh](run_minimax_h3_image_i2i_tp_sp.sh) |

按任务选择一个脚本运行：

```bash
# 单卡文生图
bash scripts/minimax_h3_image/run_minimax_h3_image_t2i.sh

# 单卡参考图生成／编辑
bash scripts/minimax_h3_image/run_minimax_h3_image_i2i.sh

# 四卡 SP4 文生图
bash scripts/minimax_h3_image/run_minimax_h3_image_t2i_sp4.sh

# 四卡 SP4 参考图生成／编辑
bash scripts/minimax_h3_image/run_minimax_h3_image_i2i_sp4.sh

# 四卡 TP2 × SP2 文生图
bash scripts/minimax_h3_image/run_minimax_h3_image_t2i_tp_sp.sh

# 四卡 TP2 × SP2 参考图生成／编辑
bash scripts/minimax_h3_image/run_minimax_h3_image_i2i_tp_sp.sh
```

六个脚本当前均输出 `4096 × 4096` PNG，保存到仓库的 `save_results/` 目录；文件名与任务及并行方式对应。I2I 默认参考图为 `$model_path/examples/reference1.png`。

两种四卡 I2I 脚本均支持通过 `IMAGE_PATH` 覆盖参考图。多张参考图使用英文逗号分隔，并相应修改脚本中的 `prompt`：

```bash
IMAGE_PATH="/path/to/reference1.png,/path/to/reference2.png" \
  bash scripts/minimax_h3_image/run_minimax_h3_image_i2i_sp4.sh
```

### 并行配置

以下为 [configs/minimax_h3_image](../../configs/minimax_h3_image) 中 CUDA 配置的默认行为，T2I、I2I 和服务均可使用：

| 配置 | GPU 数量 | DiT | Qwen3-VL | VAE |
| --- | --- | --- | --- | --- |
| [minimax_h3_image.json](../../configs/minimax_h3_image/minimax_h3_image.json) | 1 | 单卡 | 单卡 | 单卡 |
| [minimax_h3_image_sp.json](../../configs/minimax_h3_image/minimax_h3_image_sp.json) | 2 | SP2（Ulysses） | rank 0 | rank 0 |
| [minimax_h3_image_sp4.json](../../configs/minimax_h3_image/minimax_h3_image_sp4.json) | 4 | SP4（Ulysses） | TP4 | 编码、解码均按空间 tile 并行 |
| [minimax_h3_image_tp_sp.json](../../configs/minimax_h3_image/minimax_h3_image_tp_sp.json) | 4 | TP2 × SP2（Ulysses） | TP4 | 编码、解码均按空间 tile 并行 |

两个四卡配置均默认打开 `text_encoder_tensor_parallel`、`vae_encode_parallel` 和 `vae_decode_parallel`。SP4 的 DiT 仅使用序列并行；两个四卡配置均关闭 CPU offload，开启 DiT 和 VAE 编译。VAE Encoder 将单张或多张参考图的 tile 分配给各 rank，计算后通过 all-gather 让各 rank 获得完整编码结果；Decoder 的 tile 由各 rank 并行计算，最终由 rank 0 汇总、拼接和输出图片。T2I 没有参考图，不执行 VAE 编码。

切换配置时，同时调整 `CUDA_VISIBLE_DEVICES` 和 `--nproc_per_node`，使进程数等于 DiT 的 TP × SP 规模。Qwen3-VL 的 TP 使用整个推理进程组，因此两个四卡配置均为 TP4。

## 服务与 POST 请求

### 启动服务

先准备 AdaLN 缓存，再修改所选启动脚本中的 `lightx2v_path`、`model_path` 和 `CUDA_VISIBLE_DEVICES`。单卡及两种四卡服务均支持 T2I 和 I2I，默认端口为 8000。

单卡启动（[start_server.sh](server/start_server.sh)）：

```bash
bash scripts/minimax_h3_image/server/start_server.sh
```

四卡 SP4 启动（[start_server_sp4.sh](server/start_server_sp4.sh)）：

```bash
bash scripts/minimax_h3_image/server/start_server_sp4.sh
```

四卡 TP2 × SP2 启动（[start_server_parallel.sh](server/start_server_parallel.sh)）：

```bash
bash scripts/minimax_h3_image/server/start_server_parallel.sh
```

### POST 客户端

等待模型加载、编译和预热完成后，在另一终端运行对应任务的客户端。按服务地址修改 [post_t2i.py](server/post_t2i.py) 或 [post_i2i.py](server/post_i2i.py) 中的 `url`。I2I 还需将 `image_path` 改为客户端本地参考图路径，客户端会将图片编码为 Base64 发送。

```bash
python scripts/minimax_h3_image/server/post_t2i.py
python scripts/minimax_h3_image/server/post_i2i.py
```

客户端提交请求后返回任务 ID，默认输出为 `2048 × 2048` PNG。生成完成后，图片保存在服务端的 `$lightx2v_path/save_results/server_cache/outputs/` 下。可在客户端中修改 `prompt`、`size`、`seed` 和 `save_result_path`。

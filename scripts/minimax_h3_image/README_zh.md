# MiniMax-H3-Image

[English](README.md) | [简体中文](README_zh.md)

支持文生图（`t2i`）和参考图生成／编辑（`i2i`），共用 MiniMax-H3-Image 权重和六步 PDD 调度。`i2i` 支持单张及多张参考图。

## 运行前准备

先安装 LightX2V 及对应设备的依赖。Qwen3-VL 视觉编码器要求 `transformers>=5.15.0`，需要包含 `get_vision_interpolation_indices_and_weights` 接口。

下文命令均从仓库根目录执行。新终端先设置路径并加载基础环境：

```bash
lightx2v_path=/data/nvme1/yongyang/dan/x2v_infer/h3-image/LightX2V
model_path=/data/nvme1/models/MiniMaxAI/MiniMax-H3-Image

cd "$lightx2v_path"
source "$lightx2v_path/scripts/base/base.sh"
```

`model_path` 指向完整的模型根目录，包含 Transformer、Qwen3-VL、VAE、Tokenizer／Processor，以及 `pdd_config.json`、`pdd_fused_heads.safetensors`。

使用推理脚本或缓存生成脚本前，先修改脚本内的 `lightx2v_path`、`model_path` 和 `CUDA_VISIBLE_DEVICES`。当前单卡推理脚本使用 GPU 7，四卡脚本使用 GPU 4、5、6、7，缓存生成脚本使用 GPU 0。

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

| 任务 | 单卡 | 四卡 TP2 × SP2 |
| --- | --- | --- |
| 文生图 | [run_minimax_h3_image_t2i.sh](run_minimax_h3_image_t2i.sh) | [run_minimax_h3_image_t2i_tp_sp.sh](run_minimax_h3_image_t2i_tp_sp.sh) |
| 参考图生成／编辑 | [run_minimax_h3_image_i2i.sh](run_minimax_h3_image_i2i.sh) | [run_minimax_h3_image_i2i_tp_sp.sh](run_minimax_h3_image_i2i_tp_sp.sh) |

按任务选择一个脚本运行：

```bash
# 单卡文生图
bash scripts/minimax_h3_image/run_minimax_h3_image_t2i.sh

# 单卡参考图生成／编辑
bash scripts/minimax_h3_image/run_minimax_h3_image_i2i.sh

# 四卡文生图
bash scripts/minimax_h3_image/run_minimax_h3_image_t2i_tp_sp.sh

# 四卡参考图生成／编辑
bash scripts/minimax_h3_image/run_minimax_h3_image_i2i_tp_sp.sh
```

四个脚本当前均输出 `4096 × 4096` PNG，保存到仓库的 `save_results/` 目录；文件名与任务及并行方式对应。I2I 默认参考图为 `$model_path/examples/reference1.png`。

四卡 I2I 脚本支持通过 `IMAGE_PATH` 覆盖参考图。多张参考图使用英文逗号分隔，并相应修改脚本中的 `prompt`：

```bash
IMAGE_PATH="/absolute/path/reference1.png,/absolute/path/reference2.png" \
  bash scripts/minimax_h3_image/run_minimax_h3_image_i2i_tp_sp.sh
```

### 直接调用 CLI

单卡 T2I，可直接修改提示词、尺寸、seed 和输出路径：

```bash
CUDA_VISIBLE_DEVICES=7 python -m lightx2v.infer \
  --model_cls minimax_h3_image \
  --task t2i \
  --model_path "$model_path" \
  --config_json "$lightx2v_path/configs/minimax_h3_image/minimax_h3_image.json" \
  --prompt "A photorealistic photograph of a cat sitting by a sunlit window." \
  --size 2048 2048 \
  --seed 42 \
  --save_result_path "$lightx2v_path/save_results/minimax_h3_image_t2i.png"
```

四卡 I2I：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  -m lightx2v.infer \
  --model_cls minimax_h3_image \
  --task i2i \
  --model_path "$model_path" \
  --config_json "$lightx2v_path/configs/minimax_h3_image/minimax_h3_image_tp_sp.json" \
  --image_path "$model_path/examples/reference1.png" \
  --prompt "Keep the scene in <Picture 1> and change the lighting to warm sunset light." \
  --size 2048 2048 \
  --seed 1234 \
  --save_result_path "$lightx2v_path/save_results/minimax_h3_image_i2i_tp_sp.png"
```

CLI 的 `--image_path` 同样接受逗号分隔的多张图片路径，最多 16 张。参考图按传入顺序对应提示词中的 `<Picture 1>`、`<Picture 2>` 等标记。

### 并行配置

以下为 [configs/minimax_h3_image](../../configs/minimax_h3_image) 中 CUDA 配置的默认行为，T2I、I2I 和服务均可使用：

| 配置 | GPU 数量 | DiT | Qwen3-VL | VAE |
| --- | --- | --- | --- | --- |
| [minimax_h3_image.json](../../configs/minimax_h3_image/minimax_h3_image.json) | 1 | 单卡 | 单卡 | 单卡 |
| [minimax_h3_image_tp.json](../../configs/minimax_h3_image/minimax_h3_image_tp.json) | 2 | TP2 | rank 0 | rank 0 |
| [minimax_h3_image_sp.json](../../configs/minimax_h3_image/minimax_h3_image_sp.json) | 2 | SP2（Ulysses） | rank 0 | rank 0 |
| [minimax_h3_image_tp_sp.json](../../configs/minimax_h3_image/minimax_h3_image_tp_sp.json) | 4 | TP2 × SP2（Ulysses） | TP4 | 编码、解码均按空间 tile 并行 |

四卡配置默认打开 `text_encoder_tensor_parallel`、`vae_encode_parallel` 和 `vae_decode_parallel`。VAE Encoder 将单张或多张参考图的 tile 分配给各 rank，计算后通过 all-gather 让各 rank 获得完整编码结果；Decoder 的 tile 由各 rank 并行计算，最终由 rank 0 汇总、拼接和输出图片。T2I 没有参考图，不执行 VAE 编码。

切换配置时，同时调整 `CUDA_VISIBLE_DEVICES` 和 `--nproc_per_node`，使进程数等于 DiT 的 TP × SP 规模。Qwen3-VL 的 TP 使用整个推理进程组，因此四卡混合配置下为 TP4。

## 服务与 POST 请求

### 启动服务

先完成路径设置和 AdaLN 缓存准备。服务启动时不传 `--task`，即可在同一服务中通过请求的 `task` 切换 T2I 和 I2I。若启动时指定 `--task t2i` 或 `--task i2i`，服务只接受对应任务。H3 Image 无需设置 `--model-variant`。

以下设置将服务输入缓存和异步输出统一放在仓库的 `save_results/server_cache/` 下：

```bash
export LIGHTX2V_CACHE_DIR="$lightx2v_path/save_results/server_cache"
```

单卡启动：

```bash
CUDA_VISIBLE_DEVICES=7 python -m lightx2v.server \
  --model_cls minimax_h3_image \
  --model_path "$model_path" \
  --config_json "$lightx2v_path/configs/minimax_h3_image/minimax_h3_image.json" \
  --host 0.0.0.0 \
  --port 8000
```

四卡启动，使用与 CLI 相同的 Qwen3-VL、DiT 和 VAE 并行配置：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  -m lightx2v.server \
  --model_cls minimax_h3_image \
  --model_path "$model_path" \
  --config_json "$lightx2v_path/configs/minimax_h3_image/minimax_h3_image_tp_sp.json" \
  --host 0.0.0.0 \
  --port 8000
```

单卡与四卡选择一种启动方式即可。等待模型加载、编译和预热完成，HTTP 服务开始监听后，在另一终端发送请求。多卡模式仅 rank 0 提供 HTTP 接口，其余 rank 参与推理。

```bash
curl --fail http://127.0.0.1:8000/v1/service/status
```

交互式接口文档位于 `http://127.0.0.1:8000/docs`。远程访问时，将下文的 `127.0.0.1` 替换为服务所在主机地址。

### 同步请求：直接返回 PNG

`POST /v1/tasks/image/sync` 等待生成完成后返回 `image/png` 二进制，用 `-o` 保存即可。以下示例不要求服务端保存输出文件，图片写入运行 curl 的客户端目录。

T2I：

```bash
mkdir -p save_results
curl --fail --max-time 610 \
  http://127.0.0.1:8000/v1/tasks/image/sync \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "t2i",
    "prompt": "A photorealistic photograph of a cat sitting by a sunlit window.",
    "seed": 42,
    "size": [2048, 2048]
  }' \
  -o save_results/minimax_h3_image_t2i_http.png
```

I2I，使用服务端本地参考图：

```bash
mkdir -p save_results
curl --fail --max-time 610 \
  http://127.0.0.1:8000/v1/tasks/image/sync \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "i2i",
    "prompt": "Keep the scene in <Picture 1> and change the lighting to warm sunset light.",
    "image_path": "/data/nvme1/models/MiniMaxAI/MiniMax-H3-Image/examples/reference1.png",
    "seed": 1234,
    "size": [2048, 2048]
  }' \
  -o save_results/minimax_h3_image_i2i_http.png
```

`image_path` 是字符串。单图也可使用服务端可访问的图片 URL，或图片文件的 Base64 字符串；客户端本地图片必须先将文件内容转成 Base64，或将文件放到服务端可访问的位置。多参考图使用逗号分隔的**服务端本地路径**，例如 `"image_path": "/absolute/path/a.png,/absolute/path/b.png"`，并在提示词中按顺序引用 `<Picture 1>`、`<Picture 2>`。

同步接口默认等待上限为 600 秒。需要更长时间时，可通过 URL 查询参数 `?timeout_seconds=1200` 调整，同时增大 curl 的 `--max-time`。

### 异步请求：提交、查询、下载

异步接口立即返回任务 ID。需要后续下载结果时，请显式设置 `save_result_path`：

```bash
curl --fail http://127.0.0.1:8000/v1/tasks/image/ \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "t2i",
    "prompt": "A photorealistic photograph of a cat sitting by a sunlit window.",
    "seed": 42,
    "size": [2048, 2048],
    "save_result_path": "minimax_h3_image_t2i_async.png"
  }'
```

将返回的 `task_id` 填入下方变量，再查询状态：

```bash
task_id="替换为返回的 task_id"
curl --fail "http://127.0.0.1:8000/v1/tasks/$task_id/status"
```

状态依次为 `pending`、`processing`、`completed`；失败时为 `failed`，错误信息也在状态响应中。等待状态为 `completed` 后下载：

```bash
mkdir -p save_results
curl --fail "http://127.0.0.1:8000/v1/tasks/$task_id/result" \
  -o save_results/minimax_h3_image_t2i_async.png
```

相对 `save_result_path` 保存于服务缓存目录的 `outputs/` 下；按上述环境变量设置，示例文件位于 `$lightx2v_path/save_results/server_cache/outputs/minimax_h3_image_t2i_async.png`。I2I 异步请求使用相同接口，设置 `task: "i2i"` 并增加 `image_path` 即可。

## 请求参数与启动配置

| 设置 | 位置与含义 |
| --- | --- |
| 模型、并行、卸载、算子、编译和预热 | 启动参数及 `--config_json`；CLI 和服务共用这些配置。 |
| `task` | CLI 通过 `--task` 选择；上述服务通过每次请求的 `task` 选择 `t2i`／`i2i`。 |
| `prompt`、`image_path`、`seed` | 每次请求设置；省略 seed 或设为 null 时使用 42，显式 0 是有效 seed。 |
| `size` | `[高度, 宽度]`；CLI 写作 `--size 高度 宽度`。请求中显式设置的尺寸覆盖 JSON，省略时默认使用配置中的 `[2048, 2048]`。 |
| `save_result_path` | CLI 保存 PNG 时需要指定；同步 HTTP 可直接接收 PNG，异步下载需要指定服务端保存路径。 |

本目录对应的四份 CUDA 配置均默认启用 `use_adaln_cache`、`use_compile` 和 `warmup`。单卡、TP2 和 SP2 配置默认启用 CPU offload；四卡 TP2 × SP2 配置默认将 `cpu_offload`、`text_encoder_cpu_offload` 和 `vae_cpu_offload` 设为 `false`，让 DiT、Qwen3-VL 和 VAE 权重常驻 GPU。预热使用 `[2048, 2048]` 和 `[3552, 4736]` 两组尺寸；服务同时支持两种任务时，两种任务都会预热。预热尺寸与单次请求的输出尺寸独立。

当前推理使用六步 PDD 调度与 BF16 权重，并在实现中保留 FP32 敏感计算。该模型为 guidance-distilled 模型，配置保持 `enable_cfg: false`、`sample_guide_scale: 1.0`，无需传入负面提示词。

# 训练数据与预处理

本文说明训练数据格式，以及 WAN 和 LTX2 的 latent 数据集构建方法。预处理结果包含统一的 `metadata.jsonl`，可由 `latent_dataset` 直接读取。

```text
data_process/
├── wan/build_latent_dataset.py
└── ltx/build_latent_dataset.py
```

## 数据集选择

| 训练方式 | 数据集 | 输入 |
| --- | --- | --- |
| Flow | `video_dataset` | 视频路径和 caption 元数据，训练时在线编码 |
| Flow | `latent_dataset` | 预先生成的视频、音频和文本缓存 |
| DMD / AR-DMD | `prompt_dataset` | TXT prompt，训练时在线编码 |
| DMD / AR-DMD | `latent_dataset` | 预先生成的文本条件缓存 |
| Teacher Forcing | `latent_dataset` | PT 缓存元数据或 LMDB |

WAN 和 LTX2 使用相同的数据集入口。所有数据集均返回统一结构：

```python
{
    "inputs": {},
    "conditioning": {},
    "meta": {},
}
```

- `inputs`：图片、参考图片、视频、视频 latent 和音频 latent 等模型输入。
- `conditioning`：prompt、正向文本条件和负向文本条件。
- `meta`：文件路径、分辨率和帧数等样本信息。

## video_dataset

`video_dataset` 读取 JSON、JSONL 或 CSV 元数据，并在训练时加载视频。目录输入必须包含 `metadata.jsonl`。

```json
{"video":"videos/000001.mp4","caption":"A person walks through a park."}
```

```yaml
data:
  train:
    name: video_dataset
    data_path: /path/to/datasets/metadata.jsonl
    video_root: /path/to/datasets/videos
    video_column: video
    prompt_column: caption
    height: 480
    width: 832
    num_frames: 81
    frame_rate: 24
    batch_size: 1
```

视频路径可以是绝对路径，也可以是相对元数据文件或 `video_root` 的路径。

## prompt_dataset

`prompt_dataset` 读取 TXT 或 LIST 文件，每个非空行表示一个 prompt。

```yaml
data:
  train:
    name: prompt_dataset
    data_path: /path/to/prompts/train.txt
    batch_size: 1
```

## latent_dataset

`latent_dataset` 读取 JSON、JSONL、CSV 缓存元数据、包含 `metadata.jsonl` 的目录或 LMDB。缓存元数据支持以下字段：

| 字段 | 含义 |
| --- | --- |
| `video_latent_path` | 视频 latent 的 PT 文件 |
| `audio_latent_path` | 音频 latent 的 PT 文件，LTX2 T2AV 使用 |
| `condition_path` | 正向文本条件的 PT 文件 |
| `negative_condition_path` | 当前样本的负向文本条件，可选 |
| `caption` | 原始文本，可选 |

WAN 示例：

```json
{"video_latent_path":"latents/000001.pt","condition_path":"conditions/000001.pt"}
```

LTX2 T2AV 示例：

```json
{"video_latent_path":"latents/000001.pt","audio_latent_path":"audio_latents/000001.pt","condition_path":"conditions/000001.pt"}
```

```yaml
data:
  train:
    name: latent_dataset
    data_path: /path/to/datasets/latent_cache
    batch_size: 1
```

相对路径以元数据文件所在目录为基准。缓存目录中的 `negative_condition.pt` 会被自动加载；`negative_condition_path` 仅用于指定其他位置的缓存。包含 `data.mdb` 或 `lock.mdb` 的目录会被识别为 LMDB，并支持 WAN causal LMDB 和统一的 `sample_pt` LMDB。

常用 DataLoader 配置包括 `dataset_repeat`、`max_samples`、`batch_size`、`num_workers`、`shuffle`、`drop_last` 和 `pin_memory`。

## 预处理输入元数据

脚本接受 JSON、JSONL 或 CSV。至少需要视频路径和 caption 两列：

```json
{"video":"videos/000001.mp4","caption":"A person walks through a park."}
```

CSV 示例：

```csv
video,caption
videos/000001.mp4,A person walks through a park.
```

通过 `--video-column` 和 `--prompt-column` 指定自定义列名。相对路径可通过 `--video-root` 解析。

## WAN

以下命令同时生成视频 latent、正向文本条件和负向文本条件：

```bash
cd /path/to/LightX2V

python lightx2v_train/data_process/wan/build_latent_dataset.py \
  /path/to/datasets/metadata.csv \
  --video-root /path/to/datasets/videos \
  --output-dir /path/to/datasets/wan_latent_cache \
  --model-dir /path/to/models/Wan2.2-TI2V-5B \
  --cache-components all \
  --video-column video \
  --prompt-column caption \
  --height 704 \
  --width 1280 \
  --latent-frames 96 \
  --max-samples 1000
```

`--cache-components` 可选值：

| 值 | 输出 |
| --- | --- |
| `all` | 视频 latent 和文本条件 |
| `video` | 仅视频 latent |
| `prompt` | 仅文本条件和负向文本条件 |

输出结构：

```text
wan_latent_cache/
├── metadata.jsonl
├── negative_condition.pt
├── latents/
└── conditions/
```

构建文本缓存时，脚本使用固定的 `WAN_NEGATIVE_PROMPT` 生成 `negative_condition.pt`。仅生成视频 latent 时不会处理文本条件。

脚本支持连续片段拼接。文件名形如 `sample_0.mp4`、`sample_1.mp4` 时，会按序拼接到目标帧数；拼接后仍不足目标帧数的样本会被跳过。对应 caption 使用 `--prompt-separator` 连接。

## LTX2

LTX2 预处理调用官方 LTX-2 工具，生成视频 latent、音频 latent 和文本条件：

```bash
cd /path/to/LightX2V

python lightx2v_train/data_process/ltx/build_latent_dataset.py \
  /path/to/datasets/ltx_t2av/metadata.jsonl \
  --output-dir /path/to/datasets/ltx_t2av/cache \
  --model-path /path/to/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
  --text-encoder-path /path/to/models/LTX-2 \
  --ltx2-repo /path/to/LTX-2 \
  --resolution-buckets 768x512x241 \
  --batch-size 1 \
  --device cuda
```

输出结构：

```text
cache/
├── metadata.jsonl
├── negative_condition.pt
├── latents/
├── audio_latents/
└── conditions/
```

脚本会先使用固定的 `LTX2_NEGATIVE_PROMPT` 生成 `negative_condition.pt`。该缓存用于 DMD 和 AR-DMD；仅训练 Flow 或 Teacher Forcing 时可通过 `--skip-negative-cache` 跳过。T2AV 的 Flow 和 Teacher Forcing 需要音频 latent；仅处理无音频任务时使用 `--skip-audio`。

## 训练

训练任务通过配置中的 `data.train` 选择数据集。WAN 和 LTX2 的写法一致。

### Flow

使用原始视频和 caption 时选择 `video_dataset`。视频和文本编码在训练时执行：

```yaml
data:
  train:
    name: video_dataset
    data_path: /path/to/datasets/metadata.jsonl
    video_root: /path/to/datasets/videos
    video_column: video
    prompt_column: caption
    height: 480
    width: 832
    num_frames: 81
    frame_rate: 24
    batch_size: 1
    num_workers: 4
    shuffle: true
```

使用预计算缓存时选择 `latent_dataset`：

```yaml
data:
  train:
    name: latent_dataset
    data_path: /path/to/datasets/latent_cache/metadata.jsonl
    batch_size: 1
    num_workers: 4
    shuffle: true
```

WAN 的缓存记录包含 `video_latent_path` 和 `condition_path`。LTX2 T2AV 还需要 `audio_latent_path`。

### DMD / AR-DMD

使用 TXT prompt 并在线运行 text encoder 时选择 `prompt_dataset`：

```yaml
data:
  train:
    name: prompt_dataset
    data_path: /path/to/prompts/train.txt
    batch_size: 1
    num_workers: 4
    shuffle: true
    drop_last: true
```

使用预计算文本条件时选择 `latent_dataset`。元数据中的每条记录需要包含 `condition_path`：

```yaml
data:
  train:
    name: latent_dataset
    data_path: /path/to/datasets/prompt_cache/metadata.jsonl
    batch_size: 1
    num_workers: 4
    shuffle: true
    drop_last: true
```

`latent_dataset` 会自动加载缓存目录中的 `negative_condition.pt`。

### Teacher Forcing

Teacher Forcing 使用 `latent_dataset`。PT 缓存通过 `metadata.jsonl` 加载：

```yaml
data:
  train:
    name: latent_dataset
    data_path: /path/to/datasets/tf_cache/metadata.jsonl
    batch_size: 1
    num_workers: 4
    shuffle: true
    drop_last: true
```

LMDB 直接将 `data_path` 指向包含 `data.mdb` 或 `lock.mdb` 的目录：

```yaml
data:
  train:
    name: latent_dataset
    data_path: /path/to/datasets/tf_cache.lmdb
    batch_size: 1
    num_workers: 4
    shuffle: true
    drop_last: true
```

WAN Teacher Forcing 需要视频 latent 和文本条件。LTX2 T2AV Teacher Forcing 需要视频 latent、音频 latent 和文本条件。

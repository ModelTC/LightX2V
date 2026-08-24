# 训练数据与预处理

本文说明训练数据格式，以及 WAN 的 latent 数据集构建方法。预处理结果包含统一的 `metadata.jsonl`，可由 `latent_dataset` 直接读取。

```text
data_process/
└── wan/build_latent_dataset.py
```

## 图像训练缓存

Qwen-Image、LongCat-Image 和 Flux2 的生成/编辑训练可以预先缓存 VAE latent
与文本、参考图条件。构建缓存时只加载 VAE 和条件编码器，不加载 DiT：

```bash
cd /path/to/LightX2V

python lightx2v_train/cache_data.py \
  --config lightx2v_train/configs/train/flow/qwen_image_lora.yaml \
  --output_dir /path/to/qwen_image_cache \
  --save_dtype bf16
```

输出格式对所有图像模型保持一致：

```text
qwen_image_cache/
├── cache_data.jsonl
└── cache/
    ├── 00000000.pt
    └── 00000001.pt
```

`cache_data.jsonl` 保留原始记录，仅增加 `training_cache`：

```json
{"prompt":"A cat.","target_image":"images/cat.png","training_cache":"cache/00000000.pt"}
```

训练时把训练集改为 `image_dataset` 并指向该文件：

```yaml
data:
  use_training_cache: true

  train:
    name: image_dataset
    data_path: [/path/to/qwen_image_cache/cache_data.jsonl]
    num_workers: 8
    persistent_workers: true
    prefetch_factor: 2
    pin_memory: true

inference:
  method: none
  infer_every_iters: null
```

Dataset 会自动读取 PT，并跳过图片解码与模型侧数据处理；训练启动时也只加载
DiT。Flow Matching、Consistency、DMD 和 DOPSD 通过各自的训练能力接口生成并
消费统一格式的缓存。DOPSD 缓存训练还需将
`training.dopsd.trajectory_every_iters` 设为 `null`，因为轨迹可视化需要 VAE 解码。
缓存按算法需要保存目标 latent 以及正向、无条件或负向条件；修改 prompt、负向
prompt、图片处理参数或预训练模型后应重新构建。物理 batch 仍固定为 1；多
worker、持久 worker 与预取用于隐藏单样本 PT 的读取开销。

缓存会校验模型、算法和预处理签名；旧版 schema 1 缓存需要重新构建。

## 数据集选择

| 训练方式 | 数据集 | 输入 |
| --- | --- | --- |
| 图像 DMD | `prompt_dataset` | prompt、目标高度和目标宽度 |
| Flow | `video_dataset` | 视频路径和 caption 元数据，训练时在线编码 |
| Flow | `latent_dataset` | 预先生成的视频和文本缓存 |
| DMD / AR-DMD | `prompt_dataset` | TXT prompt，训练时在线编码 |
| DMD / AR-DMD | `latent_dataset` | 预先生成的文本条件缓存 |
| Teacher Forcing | `latent_dataset` | PT 缓存元数据或 LMDB |

所有数据集均返回统一结构：

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
```

视频路径可以是绝对路径，也可以是相对元数据文件或 `video_root` 的路径。

## prompt_dataset

`prompt_dataset` 支持 TXT、LIST、JSON 和 JSONL；TXT/LIST 中每个非空行表示一个 prompt。

```yaml
data:
  train:
    name: prompt_dataset
    data_path: /path/to/prompts/train.txt
```

图像 DMD 使用 JSON 或 JSONL。每条记录必须显式指定目标尺寸，
不需要 `target_image`：

```json
{"prompt":"A cat walking through snow.","target_height":1024,"target_width":1024}
```

多尺寸训练时，每条记录的 `(target_height, target_width)` 表示它所属的
精确尺寸桶。同一 batch 的样本会由 `bucket_by_size` 按尺寸分组：

```jsonl
{"prompt":"A square composition.","target_height":1024,"target_width":1024}
{"prompt":"A wide landscape.","target_height":768,"target_width":1344}
{"prompt":"A tall portrait.","target_height":1344,"target_width":768}
```

```yaml
data:
  train:
    name: prompt_dataset
    data_path: /path/to/image_dmd/train.jsonl
    bucket_by_size: true
    shuffle: true

training:
  method: dmd
  dmd:
    # 不配置采样比例：保持数据集中各尺寸桶的数量分布。
    image_sizes:
      - {value: [1024, 1024]}
      - {value: [16, 768, 1344]}
      - {value: [16, 1344, 768]}
```

如果需要显式控制尺寸桶的 global-batch 采样比例，为每个条目添加
`ratio`：

```yaml
training:
  dmd:
    image_sizes:
      - {value: [1024, 1024], ratio: 5}
      - {value: [16, 768, 1344], ratio: 3}
      - {value: [16, 1344, 768], ratio: 2}
```

`ratio` 是相对权重，会自动归一化；上例三个桶的期望采样占比为
`50% / 30% / 20%`。权重在分布式训练步级别生效，所有分布式 rank
在同一训练步使用同一尺寸。`value` 只支持 `[height, width]` 或
`[prefix, height, width]`，图像 DMD 使用最后两个值作为像素高宽。一份配置必须
全部带 `ratio` 或全部不带，不能混用。旧的裸数组格式（例如
`- [1024, 1024]`）不再支持。

如果配置了 `training.dmd.image_sizes`，每条样本的尺寸必须属于该列表；
如果不配置，则接受能被模型图像尺寸倍数整除的任意正尺寸（当前图像模型为
VAE 空间压缩倍率的 2 倍）。每个样本都必须
同时提供 `target_height` 和 `target_width`。物理 batch 固定为 1；
`bucket_by_size: true` 在分布式训练时保证同一步所有 rank 使用相同尺寸。
默认 `drop_last: false` 会在每个尺寸桶内重复少量样本，使样本数能被数据并行
rank 数整除；设置 `drop_last: true` 则丢弃不足一个分布式训练步的尾部样本。

## latent_dataset

`latent_dataset` 读取 JSON、JSONL、CSV 缓存元数据、包含 `metadata.jsonl` 的目录或 LMDB。缓存元数据支持以下字段：

| 字段 | 含义 |
| --- | --- |
| `video_latent_path` | 视频 latent 的 PT 文件 |
| `audio_latent_path` | 音频 latent 的 PT 文件，音视频模型可选 |
| `condition_path` | 正向文本条件的 PT 文件 |
| `negative_condition_path` | 当前样本的负向文本条件，可选 |
| `caption` | 原始文本，可选 |

WAN 示例：

```json
{"video_latent_path":"latents/000001.pt","condition_path":"conditions/000001.pt"}
```

```yaml
data:
  train:
    name: latent_dataset
    data_path: /path/to/datasets/latent_cache
```

相对路径以元数据文件所在目录为基准。缓存目录中的 `negative_condition.pt` 会被自动加载；`negative_condition_path` 仅用于指定其他位置的缓存。包含 `data.mdb` 或 `lock.mdb` 的目录会被识别为 LMDB，并支持 WAN causal LMDB 和统一的 `sample_pt` LMDB。

常用 DataLoader 配置包括 `dataset_repeat`、`max_samples`、`num_workers`、`shuffle`、`drop_last` 和 `pin_memory`。物理 batch 固定为 1，数据配置中不存在 `batch_size` 选项。

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

## 训练

训练任务通过配置中的 `data.train` 选择数据集。

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
    num_workers: 4
    shuffle: true
```

使用预计算缓存时选择 `latent_dataset`：

```yaml
data:
  train:
    name: latent_dataset
    data_path: /path/to/datasets/latent_cache/metadata.jsonl
    num_workers: 4
    shuffle: true
```

WAN 的缓存记录包含 `video_latent_path` 和 `condition_path`。

### DMD / AR-DMD

使用 TXT prompt 并在线运行 text encoder 时选择 `prompt_dataset`：

```yaml
data:
  train:
    name: prompt_dataset
    data_path: /path/to/prompts/train.txt
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
    num_workers: 4
    shuffle: true
    drop_last: true
```

WAN Teacher Forcing 需要视频 latent 和文本条件。

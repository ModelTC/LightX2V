# LightX2V Train Dataset

训练侧数据入口现在只保留三类：

- `video_dataset`: 按 metadata json/jsonl/csv 读取原始视频和 caption。
- `prompt_dataset`: 按 txt/list 读取 prompt，在线 text encoder。
- `latent_dataset`: 读取预处理好的 latent/condition cache，数据路径只能是 metadata json/jsonl/csv、包含 `metadata.jsonl` 的目录，或 LMDB 目录。

`data.format` 已删除。不要再配置 `format: prompt/pt/manifest/lmdb/auto`。

## 任务和 Dataset

WAN / LTX2 统一按下面的结构使用。

```text
flow:
  video_dataset:
    按 json/jsonl/csv 读取 raw video + caption，训练时在线 VAE/text encoder。

  latent_dataset:
    提前处理好 video latent 和 prompt latent 的 pt 文件。
    用 metadata.jsonl 描述每个样本的 video_latent_path / condition_path。
    LTX2 T2AV 额外需要 audio_latent_path。

dmd / dmd_ar:
  prompt_dataset:
    按 txt/list 读取 prompt，训练时在线 text encoder。

  latent_dataset:
    提前处理好 prompt latent。
    用 metadata.jsonl 描述每个样本的 condition_path。
    如果启用 CFG，建议提供 negative_condition.pt 或 negative_condition_path。

tf:
  latent_dataset:
    用 LMDB，或用 metadata.jsonl 指向提前处理好的 video latent 和 prompt latent pt 文件。
    WAN 旧开源 causal forcing LMDB 保留支持。
    LTX2 T2AV teacher forcing 需要 video_latent_path / audio_latent_path / condition_path。
```

## 通用配置

所有 dataset 都走 `data.train` / `data.val`：

```yaml
data:
  train:
    name: latent_dataset
    data_path: /path/to/dataset
    batch_size: 1
    num_workers: 8
    pin_memory: true
    shuffle: true
    drop_last: true
```

通用字段：

- `name`: `video_dataset`、`prompt_dataset`、`latent_dataset`。
- `data_path`: 单个路径或路径列表。
- `batch_size`: DataLoader batch size。
- `num_workers`: DataLoader worker 数。
- `dataset_repeat`: 数据重复倍数。
- `max_samples`: 调试时限制样本数。
- `shuffle/drop_last/pin_memory`: DataLoader 选项。

禁止字段：

```yaml
format: prompt
format: pt
format: manifest
format: lmdb
format: auto
```

路径类型由 `data_path` 自动决定：

- 目录下有 `data.mdb` 或 `lock.mdb`: LMDB。
- 目录下有 `metadata.jsonl`: metadata dataset。
- 文件后缀是 `.jsonl/.json/.csv`: metadata dataset。
- `prompt_dataset` 只接受 `.txt/.list`。

## Raw Metadata

`video_dataset` 和预处理工具使用 raw metadata。推荐统一成 LTX2 当前 jsonl 这种格式：

```json
{
  "id": "00274986-83a7-49aa-aac9-a380d2090d5e",
  "video": "/data/videos/00274986/output_video.mp4",
  "caption": "画面中的人物按照音频自然表达，口型与语音准确同步。",
  "width": 720,
  "height": 1280,
  "fps": 16.0,
  "duration": 1.4375,
  "frames": 23
}
```

可选字段可以保留，例如 `original_caption`、`audio_text`、`asr_status`。dataset 不强依赖这些字段。

`video_dataset` 配置示例：

```yaml
data:
  train:
    name: video_dataset
    data_path: /data/dataset/metadata.jsonl
    video_column: video
    prompt_column: caption
    height: 480
    width: 832
    num_frames: 81
    frame_rate: 16
    random_start: true
    batch_size: 1
```

输出 sample：

```python
{
    "inputs": {
        "video": video_tensor,
    },
    "conditioning": {
        "prompt": caption,
    },
    "meta": {
        "video_path": "...",
        "id": "...",
        "width": 720,
        "height": 1280,
        "fps": 16.0,
    },
}
```

## Prompt Dataset

`prompt_dataset` 用于 DMD / DMD-AR 在线 text encoder。每行一个 prompt：

```text
a person speaking naturally
a city street at night
```

配置：

```yaml
data:
  train:
    name: prompt_dataset
    data_path: /data/prompts.txt
    batch_size: 1
    num_workers: 0
```

输出 sample：

```python
{
    "inputs": {},
    "conditioning": {
        "prompt": "...",
    },
    "meta": {
        "prompt_path": "...",
        "row_index": 0,
    },
}
```

## Latent Metadata

`latent_dataset` 读取预处理 cache。metadata 推荐在 raw metadata 基础上增加 latent 路径：

```json
{
  "id": "00274986-83a7-49aa-aac9-a380d2090d5e",
  "video": "/data/videos/00274986/output_video.mp4",
  "caption": "画面中的人物按照音频自然表达，口型与语音准确同步。",
  "width": 720,
  "height": 1280,
  "fps": 16.0,
  "frames": 23,
  "video_latent_path": "latents/00274986.pt",
  "audio_latent_path": "audio_latents/00274986.pt",
  "condition_path": "conditions/00274986.pt",
  "negative_condition_path": "negative_condition.pt"
}
```

路径可以是绝对路径，也可以是相对路径。相对路径按 metadata 文件所在目录解析。

推荐目录：

```text
latent_dataset/
  metadata.jsonl
  latents/
    00274986.pt
  audio_latents/
    00274986.pt
  conditions/
    00274986.pt
  negative_condition.pt
```

配置：

```yaml
data:
  train:
    name: latent_dataset
    data_path: /data/latent_dataset
    negative_condition_path: /data/latent_dataset/negative_condition.pt
    batch_size: 1
    num_workers: 8
```

如果 `negative_condition_path` 不写，`latent_dataset` 会在 `data_path` 目录下查找 `negative_condition.pt`。

### 字段含义

- `video_latent_path`: video/VAE latent 的 pt 文件。
- `audio_latent_path`: audio latent 的 pt 文件，LTX2 T2AV 需要。
- `condition_path`: prompt/text condition 的 pt 文件。
- `negative_condition_path`: 可选，单条样本级 negative condition。
- `caption`: 可选，主要用于 debug 或 fallback 在线编码。
- `video`: 可选，原始视频路径，主要用于追溯样本。

### 输出 Key

`latent_dataset` 输出统一的 canonical key。缓存 pt/sample 也必须使用这些 key，不再兼容 `latent`、`clean_latents`、`clean_latent` 等旧名字。

```python
{
    "inputs": {
        "video_latents": video_payload,  # LTX2 读取
        "audio_latents": audio_payload,  # LTX2 T2AV 读取
        "latents": latent_tensor,        # WAN 读取，语义是干净 x0 latent
    },
    "conditioning": {
        "prompt": caption,
        "positive": prompt_condition,     # WAN prompt cache 需要 positive.prompt_embed
        "negative": negative_condition,
    },
    "meta": {...},
}
```

实际哪些 key 存在，取决于 metadata 里给了哪些路径。

## LMDB Dataset

`latent_dataset` 支持 LMDB 目录：

```text
clean_data/
  data.mdb
  lock.mdb
```

配置：

```yaml
data:
  train:
    name: latent_dataset
    data_path: /data/causal_forcing_data/dataset/clean_data
```

支持两种 LMDB：

- WAN 开源 causal forcing LMDB：`latents_shape`、`latents_0_data`、`prompts_0_data` 等。
- 新 `sample_pt` LMDB：`__format__=sample_pt`、`sample_count=N`、`sample_00000000=torch.save(sample)`。

LMDB 不需要 `format: lmdb`。
旧 WAN LMDB 物理上仍然读取 `latents_*`，dataset 输出统一为 `inputs.latents`。

## WAN 用法

### WAN Flow

在线读视频：

```yaml
data:
  train:
    name: video_dataset
    data_path: /data/wan/raw/metadata.jsonl
    video_column: video
    prompt_column: caption
```

读预处理 latent：

```yaml
data:
  train:
    name: latent_dataset
    data_path: /data/wan/latent_dataset
```

WAN latent metadata 至少需要：

```json
{"video_latent_path": "latents/000001.pt", "condition_path": "conditions/000001.pt", "caption": "..."}
```

### WAN DMD / AR-DMD

在线 text encoder：

```yaml
data:
  train:
    name: prompt_dataset
    data_path: /data/wan/prompts.txt
```

读 prompt cache：

```yaml
data:
  train:
    name: latent_dataset
    data_path: /data/wan/prompt_cache
    negative_condition_path: /data/wan/prompt_cache/negative_condition.pt
```

prompt cache metadata：

```json
{"condition_path": "conditions/000001.pt", "caption": "..."}
```

### WAN TF

旧开源 LMDB：

```yaml
data:
  train:
    name: latent_dataset
    data_path: /data/causal_forcing_data/dataset/clean_data
```

或 pt cache metadata：

```json
{"video_latent_path": "latents/000001.pt", "condition_path": "conditions/000001.pt", "caption": "..."}
```

## LTX2 用法

### LTX2 Flow

LTX2 raw jsonl 可以直接给预处理工具：

```json
{"id": "...", "video": "/path/to/video.mp4", "caption": "...", "width": 720, "height": 1280, "fps": 16, "frames": 23}
```

预处理后训练用 latent metadata：

```yaml
data:
  train:
    name: latent_dataset
    data_path: /data/ltx2/latent_dataset
```

LTX2 T2AV 至少需要：

```json
{
  "video_latent_path": "latents/000001.pt",
  "audio_latent_path": "audio_latents/000001.pt",
  "condition_path": "conditions/000001.pt",
  "caption": "..."
}
```

### LTX2 DMD / AR-DMD

在线 text encoder：

```yaml
data:
  train:
    name: prompt_dataset
    data_path: /data/ltx2/prompts.txt
```

读 prompt cache：

```yaml
data:
  train:
    name: latent_dataset
    data_path: /data/ltx2/prompt_cache
    negative_condition_path: /data/ltx2/prompt_cache/negative_condition.pt
```

注意：`data_process/ltx/build_latent_dataset.py` 只处理正向 caption，不会生成 `negative_condition.pt`。DMD/AR-DMD 如果启用 CFG，需要保持 text encoder 在线并配置正确的 `training.teacher.negative_prompt`，或额外准备 negative condition cache。

### LTX2 TF

LMDB 或 latent metadata 都可以：

```yaml
data:
  train:
    name: latent_dataset
    data_path: /data/ltx2/tf_dataset
```

metadata 至少需要：

```json
{"video_latent_path": "latents/000001.pt", "audio_latent_path": "audio_latents/000001.pt", "condition_path": "conditions/000001.pt"}
```

## 预处理入口

WAN latent dataset：

```bash
python lightx2v_train/data_process/wan/build_latent_dataset.py \
  /data/nvme5/gushiqiao/datatets/OpenVid-1M.csv \
  --video-root /data/nvme5/gushiqiao/datatets/video \
  --output-dir /data/wan/latent_dataset \
  --model-dir /path/to/Wan2.2-TI2V-5B \
  --cache-components all \
  --video-column video \
  --prompt-column caption \
  --height 704 \
  --width 1280 \
  --latent-frames 96 \
  --max-samples 1000
```

LTX2 T2AV latent dataset：

```bash
python lightx2v_train/data_process/ltx/build_latent_dataset.py \
  /data/ltx2/raw/dataset.jsonl \
  --output-dir /data/ltx2/latent_dataset \
  --model-path /path/to/ltx2.safetensors \
  --text-encoder-path /path/to/text_encoder \
  --ltx2-repo /data/nvme5/gushiqiao/codes/LTX-2 \
  --video-column video \
  --caption-column caption \
  --resolution-buckets 768x768x49 \
  --batch-size 1 \
  --device cuda \
  --overwrite
```

这个工具会写：

```text
latents/
audio_latents/
conditions/
metadata.jsonl
```

它不会写 `negative_condition.pt`。

## 迁移旧配置

旧写法：

```yaml
name: prompt_video_dataset
```

改成：

```yaml
name: video_dataset
```

旧写法：

```yaml
name: latent_dataset
format: prompt
data_path: /path/to/prompts.txt
```

改成：

```yaml
name: prompt_dataset
data_path: /path/to/prompts.txt
```

旧写法：

```yaml
name: latent_dataset
format: manifest
data_path: /path/to/cache
```

改成：

```yaml
name: latent_dataset
data_path: /path/to/cache
```

旧写法：

```yaml
name: latent_dataset
format: lmdb
data_path: /path/to/clean_data
```

改成：

```yaml
name: latent_dataset
data_path: /path/to/clean_data
```

旧写法：

```yaml
name: latent_dataset
format: pt
data_path: /path/to/pt_dir
```

需要改成 metadata：

```text
pt_dir/
  metadata.jsonl
  latents/
  conditions/
```

## 检查数据

metadata 快速检查：

```bash
head -n 3 /path/to/dataset/metadata.jsonl
find /path/to/dataset -maxdepth 2 -type f | head
```

训练前建议先：

```yaml
max_samples: 8
num_workers: 0
```

确认能跑通后再放开。

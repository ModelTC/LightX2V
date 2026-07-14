# Data Process

`data_process` 只放训练数据预处理入口，按模型拆目录：

```text
data_process/
  wan/build_latent_dataset.py
  ltx/build_latent_dataset.py
```

训练侧只保留三类 dataset：

```text
video_dataset   读 json/jsonl/csv 里的 video + caption，训练时在线编码视频和文本
prompt_dataset  读 txt/list，一行一个 prompt，训练时在线编码文本
latent_dataset  读预处理后的 metadata.jsonl 或 lmdb，直接加载 latent / condition
```

不要再写旧字段 `format: prompt`、`format: manifest`；旧的 `prompt_video_dataset` 也改成 `video_dataset`。

## WAN 预处理

从 CSV/JSON/JSONL 生成 `latent_dataset`：

```bash
cd /data/nvme5/gushiqiao/codes/LightX2V

python lightx2v_train/data_process/wan/build_latent_dataset.py \
  /data/nvme5/gushiqiao/datatets/OpenVid-1M.csv \
  --video-root /data/nvme5/gushiqiao/datatets/video \
  --output-dir /data/nvme5/gushiqiao/datatets/cache_data/wan2_2_ti2v_5b_96f \
  --model-dir /data/nvme0/gushiqiao/models/official_models/wan2.2/Wan2.2-TI2V-5B \
  --cache-components all \
  --video-column video \
  --prompt-column caption \
  --height 704 \
  --width 1280 \
  --latent-frames 96 \
  --max-samples 1000
```

`--cache-components`：

- `all`: 写 `latents/`、`conditions/`、`negative_condition.pt`、`metadata.jsonl`，可用于 WAN flow/tf/dmd。
- `video`: 只写 video latent，适合只想缓存 flow/tf 视频部分时使用。
- `prompt`: 只写 prompt condition 和 `negative_condition.pt`，适合 DMD/AR-DMD。

视频帧数不够时，脚本会按文件名末尾连续编号拼接同组视频，例如：

```text
celebv___f2KtcXAxI_0.mp4
celebv___f2KtcXAxI_1.mp4
```

拼接后仍不足 `--raw-frames` 的尾段会被丢弃。

## WAN Flow

Flow 可以直接读原始视频，也可以读预处理 latent。

原始视频：

```yaml
data:
  train:
    name: video_dataset
    data_path:
      - /data/nvme5/gushiqiao/datatets/OpenVid-1M.csv
    video_root: /data/nvme5/gushiqiao/datatets/video
    video_column: video
    prompt_column: caption
    height: 480
    width: 832
    num_frames: 81
```

预处理 latent：

```yaml
data:
  train:
    name: latent_dataset
    data_path: /data/nvme5/gushiqiao/datatets/cache_data/wan2_2_ti2v_5b_96f
```

启动：

```bash
cd /data/nvme5/gushiqiao/codes/LightX2V

CONFIG=configs/train/flow/wan2_1_t2v_14b_lora.yaml \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash lightx2v_train/scripts/run_wan_t2v.sh
```

## WAN DMD / AR-DMD

DMD 可以读 prompt txt 在线编码，也可以读 prompt cache。

在线 prompt：

```yaml
data:
  train:
    name: prompt_dataset
    data_path:
      - /data/nvme4/gushiqiao/new/Causal-Forcing/prompts/vidprom_filtered_extended.txt
```

Prompt cache：

```yaml
data:
  train:
    name: latent_dataset
    data_path: /data/nvme5/gushiqiao/datatets/cache_data/wan2_2_ti2v_5b_prompt_cache
    negative_condition_path: /data/nvme5/gushiqiao/datatets/cache_data/wan2_2_ti2v_5b_prompt_cache/negative_condition.pt
```

启动普通 DMD：

```bash
cd /data/nvme5/gushiqiao/codes/LightX2V

CONFIG=configs/train/dmd/wan2_2_ti2v_5b_dmd.yaml \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash lightx2v_train/scripts/run_wan_t2v.sh
```

启动 AR-DMD：

```bash
cd /data/nvme5/gushiqiao/codes/LightX2V

CONFIG=configs/train/dmd/wan2_2_ti2v_5b_ar_dmd.yaml \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash lightx2v_train/scripts/run_wan_t2v.sh
```

## WAN TF

TF 需要 latent 数据。新数据用 `wan/build_latent_dataset.py --cache-components all` 重新生成；旧开源 causal LMDB 仍可直接作为 `latent_dataset` 读取。

PT latent：

```yaml
data:
  train:
    name: latent_dataset
    data_path: /data/nvme5/gushiqiao/datatets/cache_data/wan2_2_ti2v_5b_96f
```

LMDB：

```yaml
data:
  train:
    name: latent_dataset
    data_path: /data/nvme5/gushiqiao/datatets/causal_forcing_data/dataset/clean_data
```

启动：

```bash
cd /data/nvme5/gushiqiao/codes/LightX2V

CONFIG=configs/train/tf/wan2_2_ti2v_5b_tf_cache_96f_sp4.yaml \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash lightx2v_train/scripts/run_wan_t2v.sh
```

## LTX2 预处理

LTX2 T2AV 需要先调用 LTX2 自身工具生成 video latent、audio latent 和 prompt condition：

```bash
cd /data/nvme5/gushiqiao/codes/LightX2V

python lightx2v_train/data_process/ltx/build_latent_dataset.py \
  /data/nvme5/gushiqiao/datatets/x2v_online/lightx2v_ltx_t2av/dataset_asr_enhanced_multilingual.jsonl \
  --output-dir /data/nvme5/gushiqiao/datatets/x2v_online/lightx2v_ltx_t2av/.precomputed \
  --model-path /data/nvme4/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
  --text-encoder-path /data/nvme0/gushiqiao/models/official_models/LTX-2 \
  --ltx2-repo /data/nvme5/gushiqiao/codes/LTX-2 \
  --resolution-buckets 512x768x241 \
  --batch-size 1 \
  --device cuda
```

输出：

```text
latents/
audio_latents/
conditions/
metadata.jsonl
```

## LTX2 Flow

LTX2 flow 读预处理后的 `latent_dataset`：

```yaml
data:
  train:
    name: latent_dataset
    data_path: /data/nvme5/gushiqiao/datatets/x2v_online/lightx2v_ltx_t2av/.precomputed
```

启动：

```bash
cd /data/nvme5/gushiqiao/codes/LightX2V

CONFIG=configs/train/flow/ltx_t2av_lora.yaml \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=6,7 \
bash lightx2v_train/scripts/run_ltx_t2av_train.sh
```

## LTX2 DMD / AR-DMD

LTX2 DMD 也读 `latent_dataset`。如果开启 CFG，数据里需要有 `conditioning.negative`，或在 config 里显式指定 `data.train.negative_condition_path`。只有 `model.load_text_encoder=true` 时，trainer 才能用 config 里的 `training.teacher.negative_prompt` 在线编码 negative condition。

```yaml
data:
  train:
    name: latent_dataset
    data_path: /data/nvme5/gushiqiao/datatets/x2v_online/lightx2v_ltx_t2av/.precomputed
```

启动普通 DMD：

```bash
cd /data/nvme5/gushiqiao/codes/LightX2V

CONFIG=configs/train/dmd/ltx_t2av_dmd_lora.yaml \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=6,7 \
bash lightx2v_train/scripts/run_ltx_t2av_dmd_lora.sh
```

启动 AR-DMD：

```bash
cd /data/nvme5/gushiqiao/codes/LightX2V

CONFIG=configs/train/dmd/ltx_t2av_ar_dmd_lora.yaml \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=6,7 \
bash lightx2v_train/scripts/run_ltx_t2av_dmd_lora.sh
```

## LTX2 TF

LTX2 TF 读同一套 `latent_dataset`，要求每条样本都有：

```text
inputs.video_latents
inputs.audio_latents
conditioning.positive
```

Config：

```yaml
data:
  train:
    name: latent_dataset
    data_path: /data/nvme5/gushiqiao/datatets/x2v_online/lightx2v_ltx_t2av/.precomputed
```

启动：

```bash
cd /data/nvme5/gushiqiao/codes/LightX2V

CONFIG=configs/train/tf/ltx_t2av_teacher_forcing_lmdb.yaml \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=6,7 \
bash lightx2v_train/scripts/run_ltx_t2av_train.sh
```

## 原始 JSON / CSV 是否能直接训练

`OpenVid-1M.csv` 和 `dataset_asr_enhanced_multilingual.jsonl` 都可以被 `video_dataset` 读取，只要 config 里指定好列名和 root：

- OpenVid: 常见字段是 `video`、`caption`，视频通常是相对路径，需要 `video_root` 或 `media_root`。
- LTX2 jsonl: 常见字段也是 `video`、`caption`，video 多数是绝对路径，通常不用额外 root。

WAN flow 可以直接使用 raw `video_dataset`。WAN tf、WAN latent-flow、WAN cached-DMD、LTX2 flow、LTX2 dmd、LTX2 tf 都建议先跑对应 `data_process` 生成 `latent_dataset`。

# 训练数据与统一缓存

LightX2V-Train 的原始数据集和训练缓存都返回同一种样本结构：

```python
{
    "inputs": {},
    "conditioning": {},
    "meta": {},
}
```

Qwen-Image、LongCat-Image、Flux2、Wan 和 MiniMax-H3 共用
`cache_data.py -> TrainingCacheTrainer -> TrainingCacheDataset` 这一条链路。模型能力
只负责把一个原始样本编码为静态训练 payload；数据集不包含模型分支，也不再使用
Wan/H3 专用的 latent 清单、LMDB 格式或独立缓存脚本。

## 构建缓存

使用与训练相同的配置运行：

```bash
cd /path/to/LightX2V

python lightx2v_train/cache_data.py \
  --config lightx2v_train/configs/train/flow/wan2_1_t2v_1_3b_lora.yaml \
  --output_dir /path/to/wan_cache \
  --save_dtype bf16
```

H3 SFT 同样直接使用原始音视频配置：

```bash
python lightx2v_train/cache_data.py \
  --config lightx2v_train/configs/train/flow/minimax_h3_t2av_lora.yaml \
  --output_dir /path/to/minimax_h3_cache \
  --save_dtype bf16
```

输出始终为：

```text
cache_output/
├── cache_data.jsonl
└── cache/
    ├── 00000000.pt
    └── 00000001.pt
```

`cache_data.jsonl` 保留源记录，并补充规范化的 `prompt` 和
`training_cache`：

```json
{"video":"videos/000001.mp4","audio":"audio/000001.wav","prompt":"A person walks through a park.","training_cache":"cache/00000000.pt"}
```

每个 PT 文件均使用公共 schema：

```python
{
    "cache_info": {...},
    "inputs": {...},
    "conditioning": {
        "prompt": "...",
        "positive": {...},
        # 算法需要时还会有 unconditional / negative / shared
    },
    "meta": {...},
}
```

缓存会校验 schema、模型名称、模型路径、训练算法和数据处理签名。修改预训练模型、
prompt、负向 prompt、分辨率、帧数或处理参数后，需要重新构建。

## 使用缓存训练

保持原配置的模型、预处理和训练算法不变，只修改训练数据路径并开启
`use_training_cache`：

```yaml
data:
  use_training_cache: true

  train:
    # name 保持构建缓存时的值，签名校验会使用它；运行时框架会自动切换到
    # model-agnostic training_cache_dataset。
    name: video_dataset
    data_path:
      - /path/to/wan_cache/cache_data.jsonl
    num_workers: 8
    persistent_workers: true
    prefetch_factor: 2
    pin_memory: true
    shuffle: true

inference:
  method: none
  infer_every_iters: null
```

此时训练入口只加载 DiT，不加载 VAE 和文本编码器；worker 只读取 PT，不会再次解码
图片、视频或音频。物理 batch 固定为 1。

## 原始数据集

### image_dataset

图像生成样本：

```json
{"target_image":"images/cat.png","prompt":"A cat."}
```

编辑模型还可以提供 `source_images`。

### video_dataset

Wan SFT/Consistency/Teacher Forcing 的基础格式：

```json
{"video":"videos/000001.mp4","prompt":"A person walks through a park."}
```

```yaml
data:
  train:
    name: video_dataset
    data_path: [/path/to/train.jsonl]
    video_column: video
    prompt_column: prompt
    height: 480
    width: 832
    num_frames: 81
    frame_rate: 24
```

MiniMax-H3 T2AV 每条记录还必须有音频：

```json
{"video":"videos/000001.mp4","audio":"audio/000001.wav","prompt":"A dog runs through snow."}
```

H3 的 `num_frames` 必须满足 `17*n+5`。处理器会把音频重采样到 32 kHz、转换为
双声道，并按视频帧数裁剪或补零。推荐使用 `video_flow_shift=6`、
`audio_flow_shift=3`；正数自定义值仍然允许。

### prompt_dataset

DMD、AR-DMD、Phased-DMD 和 SGMD 使用 TXT/LIST/JSON/JSONL prompt：

```yaml
data:
  train:
    name: prompt_dataset
    data_path: [/path/to/prompts.txt]
```

图像 DMD 的 JSONL 还需要 `target_height` 与 `target_width`。H3 DMD 的生成尺寸和
帧数由 `training.dmd.height/width/num_frames` 指定。

## 支持范围

| 模型 | 可生成统一缓存的训练方法 |
| --- | --- |
| Qwen-Image / LongCat-Image / Flux2 | Flow、Consistency、DMD、DOPSD（按模型能力） |
| Wan | Flow、Consistency、DMD、AR-DMD、Phased-DMD、SGMD、Teacher Forcing |
| MiniMax-H3 | Flow、DMD |

缓存构建要求 `dataset_repeat: 1`，并始终处理配置中的全部源记录。`save_dtype` 只转换
浮点缓存；token ID、mask 和 tag 等离散张量会保留原 dtype。

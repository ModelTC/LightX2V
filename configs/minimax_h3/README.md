# MiniMax-H3 原生音视频任务

该集成支持 `t2av`、`i2av`、`l2av`、`fl2av` 和 `ref2av`。运行时不导入 Diffusers，也不需要转换或重新下载权重；DiT、Qwen3-VL 前 50 层与视觉塔、视频/音频 VAE 和 scheduler 都由 LightX2V 原生执行。Transformers 只用于 tokenizer 与官方 Qwen3-VL 像素 processor。

## 模型目录

推荐继续使用 `/data/nvme6/gushiqiao/models/MiniMax-H3` 作为 `model_path`。根目录至少需要以下组件：

```text
MiniMax-H3/
├── transformer/
├── transformer_ref/
├── vae/
├── audio_vae/
├── text_encoder/
├── tokenizer/
└── processor/
```

`text_encoder`、`tokenizer` 和 `processor` 可以是指向 `FL2VA/` 对应目录的软链接，不需要复制权重。

基础四任务读取 `transformer/`；`ref2av` 自动读取结构相同但数值不同的 `transformer_ref/`。

原生 loader 保留发布权重中的混合精度：主体权重为 BF16，原有的 FP32 投影、时间嵌入和输出头保持 FP32。运行时要求 `DTYPE=BF16`；不要求设置 `SENSITIVE_LAYER_DTYPE`，随附脚本设置它只是为了显式表达精度策略。

## 任务与输入

- `t2av`：仅 prompt。
- `i2av`：`--image_path`，首帧条件。
- `l2av`：`--last_frame_path`，尾帧条件。
- `fl2av`：同时传首帧与尾帧。
- `ref2av`：重复传 `--reference image=/path`、`video=/path` 或 `audio=/path`，顺序会被保留。最多 9 图、3 视频、3 个带音频 reference、总计 12 项；不能只有音频。

- 当前支持单 prompt、batch size 1。
- 固定 24 fps，官方时长范围为 5–15 秒。
- 帧数必须满足 `17 * n + 5`；默认是 124 帧。不满足时会先向上对齐，再检查是否仍在时长范围内。
- 高和宽都必须是 32 的倍数；默认分辨率为 768×1344。
- 模型是 guidance-distilled，不使用 CFG，`negative_prompt` 会被忽略。
- 保存路径必须以 `.mp4` 结尾；输出为 24 fps H.264 视频和 32 kHz AAC 立体声音频。
- 默认 attention 后端是 `torch_sdpa`。只有环境已经安装并支持 FlashAttention 3 时，才应把配置中的 `attn_type` 改为 `flash_attn3`。

当前实现面向单卡，并要求按组件顺序执行 CPU offload：文本编码器 → DiT → 视频 VAE → 音频 VAE。配置中应保持 `cpu_offload`、`text_encoder_cpu_offload` 和 `vae_cpu_offload` 为 `true`，`offload_granularity` 为 `model`。

## CLI

在 LightX2V 仓库根目录运行：

```bash
MODEL_PATH=/data/nvme6/gushiqiao/models/MiniMax-H3 \
PROMPT='A cinematic wide shot of ocean waves at sunset, with synchronized natural ambience.' \
OUTPUT_PATH=outputs/minimax_h3_t2av.mp4 \
bash scripts/minimax_h3/run_minimax_h3_t2av.sh
```

可通过 `CONFIG_JSON`、`SEED`、`LIGHTX2V_PATH` 覆盖相应默认值。默认配置是 `configs/minimax_h3/minimax_h3_t2av.json`。

```bash
IMAGE_PATH=first.png bash scripts/minimax_h3/run_minimax_h3_i2av.sh
LAST_FRAME_PATH=last.png bash scripts/minimax_h3/run_minimax_h3_l2av.sh
IMAGE_PATH=first.png LAST_FRAME_PATH=last.png bash scripts/minimax_h3/run_minimax_h3_fl2av.sh
REFERENCES='image=person.png,video=motion.mp4,audio=voice.wav' \
  bash scripts/minimax_h3/run_minimax_h3_ref2av.sh
```

首尾帧任务在未显式传 `target_shape` 时由第一张实际输入图决定 768p 画布。`ref2av` 的参考图使用自身 2048 短边画布，参考视频重采样到 24 fps，参考音频统一为 32 kHz 立体声。

## Python pipeline

使用同一份 `config_json` 即可初始化程序化入口：

```python
import os

os.environ.setdefault("DTYPE", "BF16")

from lightx2v.pipeline import LightX2VPipeline

pipe = LightX2VPipeline(
    task="t2av",
    model_cls="minimax_h3",
    model_path="/data/nvme6/gushiqiao/models/MiniMax-H3",
)
pipe.create_generator(
    config_json="configs/minimax_h3/minimax_h3_t2av.json",
)
result = pipe.generate(
    seed=42,
    prompt="A cinematic wide shot of ocean waves at sunset, with synchronized natural ambience.",
    save_result_path="outputs/minimax_h3_t2av.mp4",
)
```

设置 `return_result_tensor=True` 时返回视频、立体声音频和采样率，不写输出文件。

Python API 的条件参数与 CLI 对应。每个 pipeline 在初始化时固定权重分区；`ref2av` 应单独初始化，并接收有序列表：

```python
ref_pipe = LightX2VPipeline(
    task="ref2av",
    model_cls="minimax_h3",
    model_path="/data/nvme6/gushiqiao/models/MiniMax-H3",
)
ref_pipe.create_generator(
    config_json="configs/minimax_h3/minimax_h3_ref2av.json",
)
ref_pipe.generate(
    prompt="Keep the character and camera motion, with matching ambience.",
    references=["image=person.png", "video=motion.mp4", "audio=voice.wav"],
    save_result_path="outputs/ref2av.mp4",
)
```

## 当前未支持

- CFG、negative prompt 条件分支和 CFG parallel
- sequence parallel、tensor parallel 及多卡推理
- DiT 量化、LoRA 和 feature caching
- block offload、`lazy_load`、`unload_modules` 和 warmup
- 转换后的 checkpoint 或非官方 safetensors 权重格式

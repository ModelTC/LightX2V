# MiniMax-H3 Roll：低成本抽卡

[English](README.md) | [简体中文](README_zh.md)

先用低分辨率草片筛选提示词和 seed，再对选中的草片做高分辨率重生成。可用 `draft_steps` 进一步减少草片的计算量。以下命令均从仓库根目录执行。

重生成统一使用 `resample_source_video`：解码草片的 RGB 帧，按目标尺寸做 Lanczos 缩放，再进行 video/audio VAE 编码，按 `resample_start_step` 对应的 video/audio sigma 加噪，执行原时间表后缀去噪和解码。不读写 latent 文件，不对 latent 做空间插值；同尺寸 RGB 输入跳过缩放。

接口支持离线 CLI、Python `LightX2VPipeline.generate()`、推理 JSON 和 HTTP server 的 JSON 请求。`fl2av` 的首尾帧条件与草片视频是独立输入，重生成时仍可传入首尾图。

## T2AV 启动脚本

两个独立脚本默认使用八卡 SP8、8 步 LoRA 配置：

- [run_minimax_h3_t2av_draft.sh](run_minimax_h3_t2av_draft.sh)：生成 480p 草片。
- [run_minimax_h3_t2av_resample.sh](run_minimax_h3_t2av_resample.sh)：将选中的草片重生成为 768p 视频。

脚本运行前直接修改两个脚本中的 `lightx2v_path`、`model_path` 和 `CUDA_VISIBLE_DEVICES`，设置 `--config_json` 对应配置中的本地 LoRA 路径，并准备匹配的 AdaLN 缓存。

```bash
bash scripts/minimax_h3_roll/run_minimax_h3_t2av_draft.sh
bash scripts/minimax_h3_roll/run_minimax_h3_t2av_resample.sh
```

默认输出到 `save_results/minimax_h3_roll_t2av/draft.mp4` 和 `resample.mp4`。草片脚本显式设置 `--draft_steps 3`，生成 K3 草片；删除该参数则执行配置的完整 8 步。重生成从 `resample_start_step=1` 开始（当前为后 7 步）。两个阶段使用相同提示词、seed 和帧数，各自启动一次 `lightx2v.infer`，会分别加载模型；需要复用模型时使用下文的 Python 示例或 HTTP server。

输入和生成参数直接在脚本中修改：

| 脚本变量或参数 | 示例值 | 用途 |
| --- | --- | --- |
| `lightx2v_path` / `model_path` | `/path/to/LightX2V` / `/path/to/MiniMax-H3` | 仓库和基础模型目录，运行前替换为本机路径。 |
| `--config_json` | `configs/minimax_h3_roll/minimax_h3_bf16_8step.json`（相对仓库根目录） | 启动配置；LoRA、shift 和步数由配置决定。 |
| `CUDA_VISIBLE_DEVICES` / `--nproc_per_node` | `0,1,2,3,4,5,6,7` / `8` | GPU 和进程数，须与配置的并行规模匹配。 |
| `--prompt` / `--seed` / `--num_frames` | 雨中黄色电车示例 / `42` / `124` | 两阶段共用的提示词、随机种子和帧数。 |
| `--draft_steps` | `3` | 脚本生成 3 步草片；删除该参数则沿用配置的完整步数。 |
| `--size` | 草片 `480 864`；重生成 `768 1344` | `[高度, 宽度]`，高清重生成可改为 `1088 1920`。 |
| `--resample_start_step` | `1` | 重生成在原时间表中的起点索引。 |
| `--resample_source_video` | `save_results/minimax_h3_roll_t2av/draft.mp4`（相对仓库根目录） | 要重生成的草片，可指定已选中的带音轨视频。 |
| `--save_result_path` | `save_results/minimax_h3_roll_t2av/{draft,resample}.mp4`（相对仓库根目录） | 结果文件；再次使用相同路径会覆盖对应结果。 |
| `DTYPE` / `SENSITIVE_LAYER_DTYPE` | `BF16` / `BF16` | 推理精度，与现有 H3 离线脚本一致。 |

草片脚本已设置 `--draft_steps 3`。生成 K3 草片并查看结果后，再运行重生成脚本：

```bash
bash scripts/minimax_h3_roll/run_minimax_h3_t2av_draft.sh
# 查看 save_results/minimax_h3_roll_t2av/draft.mp4 后，重生成选中的草片。
bash scripts/minimax_h3_roll/run_minimax_h3_t2av_resample.sh
```

换 seed 时修改 `--save_result_path` 保留候选草片，并把重生成脚本的 `--resample_source_video` 指向选中的草片。15 秒例子将两个脚本中的 `--num_frames` 都改为 `362`。重生成已有视频时，直接修改 `--resample_source_video`，同时设置与源视频匹配的帧数和所需提示词。

## FL2AV 启动脚本

FL2AV 同时使用首帧和尾帧图片。两个独立脚本沿用上文的八卡 SP8、BF16/BF16 和 8 步 LoRA 配置：

- [run_minimax_h3_fl2av_draft.sh](run_minimax_h3_fl2av_draft.sh)：根据首尾帧生成 480p K3 草片。
- [run_minimax_h3_fl2av_resample.sh](run_minimax_h3_fl2av_resample.sh)：保留首尾帧条件，将选中的草片重生成为 768p 视频。

运行前修改两个脚本中的 `lightx2v_path`、`model_path` 和 `CUDA_VISIBLE_DEVICES`，设置 `configs/minimax_h3_roll/minimax_h3_bf16_8step.json` 中的本地 LoRA 路径，并准备匹配的 AdaLN 缓存。

示例复用仓库内的小鸟首尾帧，无需另外下载图片：

| 首帧：`--image_path` | 尾帧：`--last_frame_path` |
| --- | --- |
| ![站立的小鸟](../../assets/inputs/imgs/flf2v_input_first_frame-fs8.png) | ![飞行的小鸟](../../assets/inputs/imgs/flf2v_input_last_frame-fs8.png) |

两个脚本使用同一提示词，描述小鸟起飞并配上鸟鸣、振翅和微风声：

```text
A fluffy blue bird stands on a wooden ledge, looks up, spreads its wings and takes off into a bright blue sky. The camera follows the bird upward in one continuous shot. Sound: soft chirping, fluttering wings and a gentle breeze.
```

```bash
bash scripts/minimax_h3_roll/run_minimax_h3_fl2av_draft.sh
# 查看 save_results/minimax_h3_roll_fl2av/draft.mp4 后，重生成选中的草片。
bash scripts/minimax_h3_roll/run_minimax_h3_fl2av_resample.sh
```

默认输出到 `save_results/minimax_h3_roll_fl2av/draft.mp4` 和 `resample.mp4`。两个阶段都传 `--task fl2av`，使用相同的 `--image_path`、`--last_frame_path`、提示词、seed `42` 和帧数 `124`。草片为 `--size 480 864 --draft_steps 3`；重生成为 `--size 768 1344 --resample_start_step 1`，执行原 8 步时间表的后 7 步。

替换输入时，两个脚本都应修改首尾图和提示词；重生成的 `--resample_source_video` 指向选中的带音轨草片，同时保留首尾帧参数。删除草片脚本中的 `--draft_steps 3` 可生成完整 8 步草片。15 秒例子将两个脚本的 `--num_frames` 都改为 `362`。每个脚本分别加载模型。

## Ref2AV 启动脚本

Ref2AV 使用参考素材引导生成。两个独立脚本默认八卡 SP8、BF16/BF16，使用基础模型的 `transformer_ref/` 权重和单独的 [Ref2AV 8 步配置](../../configs/minimax_h3_roll/minimax_h3_ref2av_bf16_8step.json)：

- [run_minimax_h3_ref2av_draft.sh](run_minimax_h3_ref2av_draft.sh)：根据参考图生成 480p K3 草片。
- [run_minimax_h3_ref2av_resample.sh](run_minimax_h3_ref2av_resample.sh)：保留参考图条件，将选中的草片重生成为 768p 视频。

运行前修改两个脚本中的 `lightx2v_path`、`model_path` 和 `CUDA_VISIBLE_DEVICES`。基础模型目录需包含 `transformer_ref/`；将配置中的 LoRA 路径改为本地的 [minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors)。此示例使用 video/audio shift `12/3`、`alpha=8` 和 `reference_image_resize_mode="match"`。

首次运行前，用 Ref2AV 配置和 `--model-variant ref2av` 准备匹配的 AdaLN 缓存；已有匹配缓存时跳过这一步：

```bash
lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3

export CUDA_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"

python "${lightx2v_path}/tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py" \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3_roll/minimax_h3_ref2av_bf16_8step.json" \
  --model-variant ref2av
```

示例复用仓库内的小鸟图片作为外观参考，不固定首尾帧，无需另外下载素材：

![小鸟参考图](../../assets/inputs/imgs/flf2v_input_first_frame-fs8.png)

两个脚本使用同一提示词，描述参考小鸟在花园木栅栏上跳跃、歪头和振翅：

```text
The fluffy blue bird from the reference image hops along a wooden garden fence, tilts its head and flaps its small wings. Keep its round body, blue feathers and expressive eyes consistent. A gentle camera pan follows the bird in warm morning sunlight. Sound: cheerful chirping, soft wingbeats and a light breeze.
```

```bash
bash scripts/minimax_h3_roll/run_minimax_h3_ref2av_draft.sh
# 查看 save_results/minimax_h3_roll_ref2av/draft.mp4 后，重生成选中的草片。
bash scripts/minimax_h3_roll/run_minimax_h3_ref2av_resample.sh
```

默认输出到 `save_results/minimax_h3_roll_ref2av/draft.mp4` 和 `resample.mp4`。两个阶段都传 `--model-variant ref2av --task ref2av`，使用相同的 `--image_path`、提示词、seed `42` 和帧数 `124`。草片为 `--size 480 864 --draft_steps 3`；重生成为 `--size 768 1344 --resample_start_step 1`，执行原 8 步时间表的后 7 步。删除草片脚本中的 `--draft_steps 3` 可生成完整 8 步草片；15 秒例子将两个脚本的 `--num_frames` 都改为 `362`。每个脚本分别加载模型。

替换参考输入时，两个脚本都应同步修改 `--image_path` 和提示词；也可添加 `--video_path` / `--audio_path`，同类多个文件用逗号分隔，音频参考需同时提供图片或视频参考。重生成时保留原始参考输入，将选中的带音轨草片传给 `--resample_source_video`；`--video_path` 表示参考视频，不能替代该参数。

## 参数与固定流程

| 字段 | 默认值 | 用途 |
| --- | --- | --- |
| `resample_source_video` | 无 | 带音轨的草片文件路径；RGB 缩放到目标尺寸后编码。省略时按草片或普通生成配置执行。 |
| `resample_start_step` | `1` | 原时间表的起点索引，整数且位于 `[0,infer_steps]`。`1` 只跳过第一步，`0` 完整重生成，`infer_steps` 只做 VAE 编解码。 |
| `draft_steps` | 当前配置的 `infer_steps` | 仅显式指定正整数 K 且 `K < infer_steps` 时，执行前 K 次前向并以最后一次 velocity 估计 clean AV。省略或 `K >= infer_steps` 时走原完整采样流程。 |

字段可放入启动配置，也可按请求传给 `generate()`。请求的非 `None` 值覆盖启动配置，包括 `resample_start_step=0`。CLI 使用对应的 `--字段名`。工作流配置不预设 `draft_steps`：例如当前 `infer_steps=8` 时，默认完整跑 8 步；换成 4 步或 29 步配置则默认跑相应步数。只有 `1 <= draft_steps < infer_steps` 才开启少步生成，这时不能与 `resample_source_video` 同时使用。少步生成支持 `ref2av`：仅对目标音视频行做 clean 投影，图像、视频及音频参考行保留原条件。大于或等于配置步数不会扩展时间表，也不会执行提前 clean 投影。

Ref2AV 的两个阶段都应传入原始 `image_path` / `video_path` / `audio_path` 参考条件。先用 `draft_steps=3` 生成少步草片，再省略 `draft_steps` 并设置 `resample_source_video` 做重生成；不能把草片只作为 `video_path` 传入来替代重生成。提前结束也会投影目标音频，少步草片的音频质量需单独试听。

重生成固定为 **suffix sampling**：用 `resample_start_step` 直接索引原时间表，执行其余步骤，不重新铺设时间表。video/audio 分别取同一索引下各自的 sigma 和 timestep。日志记录实际 sigma、起点索引和剩余 NFE。旧参数 `resample_strength` 已移除。

```text
resample_source_video → RGB 解码 → Lanczos 缩放到目标宽高 → Video VAE encode
                      + 源音轨 → Audio VAE encode
                      → 分别按 video/audio sigma 加噪
                      → 原时间表剩余步骤 → AV VAE decode → 输出视频
```

video VAE 保持现有 seed-42 posterior sample，audio VAE 固定编码草片音轨并使用 posterior mean。源视频必须带音轨；无音轨会报错，不补充静音。

## 一次加载模型：草片 → RGB 重生成

以下 Python 脚本需在与配置匹配的 8 卡 `torchrun` 进程中运行。`size` 顺序为 `[height,width]`。

```python
from lightx2v import LightX2VPipeline

pipe = LightX2VPipeline(
    model_path="/root/lightx2v_h3_world/models/MiniMax-H3-native",
    model_cls="minimax_h3", model_variant="fl2av",
)
pipe.create_generator(config_json="configs/minimax_h3_roll/minimax_h3_bf16_8step.json")
prompt = "A red fox walks through a snowy clearing, one locked-off shot. Sound: paws on snow and soft wind."
pipe.generate(
    task="t2av", prompt=prompt, seed=42, size=[480,864], num_frames=124,
    save_result_path="/root/h3_example/draft.mp4",  # 默认按配置完整跑 8 步
)
pipe.generate(
    task="t2av", prompt=prompt, seed=2026, size=[1088,1920], num_frames=124,
    resample_source_video="/root/h3_example/draft.mp4",  # 默认 resample_start_step=1，跑后 7 步
    save_result_path="/root/h3_example/rgb.native.mp4",
)
```

本配置下省略第一条请求的 `draft_steps` 即生成完整 8 步草片；需要少步草片时显式传入 `draft_steps=3` 等小于 8 的正整数。`draft_steps=8` 或更大时仍走原完整 8 步流程。高清草片也可使用 `size=[1088,1920]`，此时第二阶段不缩放 RGB，但仍需 VAE 重编码和解码。`fl2av` 两次请求均传入 `task="fl2av"`、`image_path` 和 `last_frame_path`，并显式指定各自的 `size`。

## 服务化：串行 draft → resample

启动方式参考 [H3 服务文档](../minimax_h3/README_zh.md#服务与-post-请求)。下面直接使用仓库已提交的 `lightx2v.server` 入口、Roll 配置和图片，客户端代码完整写在文档中。服务启动时加载一次模型，后续草片和重生成请求复用它。

先设置仓库、基础模型和配置中的本地 LoRA 路径，并准备匹配的 AdaLN 缓存，再启动八卡服务。启动时指定 `--model-variant fl2av`，每条请求通过 `task` 选择 T2AV 或 FL2AV：

```bash
lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source "${lightx2v_path}/scripts/base/base.sh"

torchrun --standalone --nproc_per_node=8 -m lightx2v.server \
  --model_cls minimax_h3 \
  --model-variant fl2av \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3_roll/minimax_h3_bf16_8step.json" \
  --host 0.0.0.0 \
  --port 8000
```

四卡运行时，将 `CUDA_VISIBLE_DEVICES` 改为四张卡、`--nproc_per_node` 改为 `4`，并将所用配置的 `parallel.seq_p_size` 改为 `4`。

等待服务启动完成，在另一个终端的仓库根目录运行以下客户端。默认生成 T2AV 雨中电车；将 `task` 改为 `"fl2av"` 即使用上文的小鸟首尾帧示例，图片通过 Base64 上传。客户端与服务端不在同一台机器时，修改 `url`。

```bash
python - <<'PY'
import base64
import time
from pathlib import Path

import requests

url = "http://127.0.0.1:8000"
task = "t2av"  # 改为 "fl2av" 使用首尾帧示例。

def submit_and_wait(message):
    response = requests.post(f"{url}/v1/tasks/video/", json=message, timeout=30)
    response.raise_for_status()
    task_id = response.json()["task_id"]
    print(f"Submitted {task_id}", flush=True)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        response = requests.get(f"{url}/v1/tasks/{task_id}/status", timeout=30)
        response.raise_for_status()
        result = response.json()
        if result["status"] == "completed":
            if not result.get("save_result_path"):
                raise RuntimeError(f"Task {task_id} completed without a saved video")
            print(result, flush=True)
            return result["save_result_path"]
        if result["status"] in {"failed", "cancelled"}:
            raise RuntimeError(f"Task {task_id}: {result}")
        time.sleep(1)
    raise TimeoutError(f"Task {task_id}: check its status before submitting another request")

common = {
    "task": task,
    "prompt": "A yellow tram passes along a rainy street. Sound: rainfall and a tram bell.",
    "seed": 42,
    "num_frames": 124,
}
if task == "fl2av":
    common.update(
        prompt="A fluffy blue bird stands on a wooden ledge, looks up, spreads its wings and takes off into a bright blue sky. The camera follows the bird upward in one continuous shot. Sound: soft chirping, fluttering wings and a gentle breeze.",
        image_path=base64.b64encode(Path("assets/inputs/imgs/flf2v_input_first_frame-fs8.png").read_bytes()).decode("ascii"),
        last_frame_path=base64.b64encode(Path("assets/inputs/imgs/flf2v_input_last_frame-fs8.png").read_bytes()).decode("ascii"),
    )
output_dir = f"minimax_h3_roll/{task}_{time.time_ns()}"
source_video = submit_and_wait({
    **common,
    "size": [480, 864],
    "draft_steps": 3,
    "save_result_path": f"{output_dir}/draft.mp4",
})
submit_and_wait({
    **common,
    "size": [768, 1344],
    "resample_source_video": source_video,
    "resample_start_step": 1,
    "save_result_path": f"{output_dir}/resample.mp4",
})
PY
```

这两条请求严格串行：草片状态达到 `completed` 后才提交重生成；失败或取消则停止。草片执行 K3，重生成执行后 7 步；需要 K8 草片时删除第一条请求中的 `draft_steps`。FL2AV 两次请求保留相同首尾帧、提示词、seed 和帧数。

`save_result_path` 的相对路径按服务输出目录解析；重生成使用草片完成后返回的服务端绝对路径。`resample_source_video` 必须指向服务端可读取的带音轨视频。客户端会打印任务 ID 和完成结果，可用 `GET /v1/tasks/{task_id}/result` 下载对应视频。

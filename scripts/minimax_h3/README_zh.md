# MiniMax-H3

[English](README.md) | [简体中文](README_zh.md)

MiniMax-H3 可以生成带有同步立体声音频的视频。以下命令均在 LightX2V 仓库根目录执行。运行前，请先设置所选 shell 脚本中的 `lightx2v_path` 和 `model_path`。

## 模型权重与任务

`model_path` 下需要采用发布的 Diffusers 组件目录结构：

```text
MiniMax-H3/
├── transformer/       # t2av, i2av, l2av, fl2av
├── transformer_ref/   # ref2av
├── text_encoder/
├── tokenizer/
├── processor/
├── vae/
└── audio_vae/
```

每个 transformer 目录都需要包含 `config.json`、权重索引和权重分片。请按所需任务下载对应组件；只下载原始的 `FL2VA/` 或 `Ref2VA/` 权重目录，并不能满足上述目录要求。FP8/INT8 配置仍需使用原始组件配置、tokenizer 和 VAE，并在 JSON 中另行指定本地量化权重。

| 任务 | 请求输入 | Transformer |
| --- | --- | --- |
| `t2av` | `prompt` | `transformer` |
| `i2av` | `prompt`、`image_path`（首帧） | `transformer` |
| `l2av` | `prompt`、`last_frame_path` | `transformer` |
| `fl2av` | `prompt`、`image_path`、`last_frame_path` | `transformer` |
| `ref2av` | `prompt`、参考图片和/或视频，以及可选的参考音频 | `transformer_ref` |

基础 transformer 加载一次即可处理前四种任务。参考生成使用独立的 transformer 和服务。这些权重已经完成 CFG 蒸馏，请勿传入 `negative_prompt`，包括空字符串。

## 离线推理

五种任务脚本共用 `configs/minimax_h3/minimax_h3.json`：单 GPU、BF16 权重、模型级 CPU 卸载，默认输出 124 帧，`[高度, 宽度] = [544, 960]`。脚本中的 `--task` 决定加载哪组 transformer 以及如何处理输入，JSON 文件名不决定任务。

```bash
bash scripts/minimax_h3/run_minimax_h3_t2av.sh
bash scripts/minimax_h3/run_minimax_h3_i2av.sh
bash scripts/minimax_h3/run_minimax_h3_l2av.sh
bash scripts/minimax_h3/run_minimax_h3_fl2av.sh
bash scripts/minimax_h3/run_minimax_h3_ref2av.sh
```

普通配置使用 SageAttention2 和 SGL 算子。请根据已安装的算子和设备，选择合适的注意力、量化和并行配置。模型路径、prompt、seed 和输出路径直接写在脚本中；每个 JSON 都是一份完整的启动配置。

切换运行方式时，将下表中脚本的 `--config_json` 改为 `configs/minimax_h3/` 下对应的文件。单 GPU 脚本也可运行 FP8、compile 和四步 LoRA 配置；`run_minimax_h3_t2av_parallel.sh` 统一用于 SP、TP 和混合并行。下表列出各配置设置的并行规模。文件名中的 `encoder` 指文本编码器，`vae` 指视频 VAE。

| `configs/minimax_h3/` 下的配置 | 启动脚本 | 运行方式 |
| --- | --- | --- |
| `minimax_h3.json` | 上述五种任务脚本中的任意一个 | 单 GPU BF16 |
| `minimax_h3_compile.json` | 上述五种任务脚本中的任意一个 | 编译并在启动时预热 |
| `minimax_h3_block_offload.json` | `run_minimax_h3_t2av.sh` | 单 GPU BF16，分块卸载 |
| `minimax_h3_sp.json` | `run_minimax_h3_t2av_parallel.sh` | SP4 |
| `minimax_h3_tp.json` | `run_minimax_h3_t2av_parallel.sh` | TP2 |
| `minimax_h3_tp_sp.json` | `run_minimax_h3_t2av_parallel.sh` | TP2 × SP2 |
| `minimax_h3_sol_block_offload.json` | `run_minimax_h3_t2av.sh` | 单 GPU Sol-Attn |
| `fp8/minimax_h3.json` | `run_minimax_h3_t2av.sh` | 单 GPU DiT FP8 |
| `fp8/minimax_h3_encoder_fp8.json` | `run_minimax_h3_t2av.sh` | 单 GPU DiT + 文本编码器 FP8 |
| `fp8/minimax_h3_vae_fp8.json` | `run_minimax_h3_t2av.sh` | 单 GPU DiT + 视频 VAE FP8 |
| `fp8/minimax_h3_sp_5090.json` | `run_minimax_h3_t2av_parallel.sh` | SP8，5090 FP8 配置 |
| `dmd/minimax_h3_bf16_4step.json` | `run_minimax_h3_t2av.sh` | 单 GPU BF16，四步 LoRA |
| `dmd/minimax_h3_bf16_4step_sol.json` | `run_minimax_h3_t2av.sh` | 单 GPU 四步 Sol-Attn |
| `dmd/minimax_h3_fp8_4step.json` | `run_minimax_h3_t2av_parallel.sh` | SP4，FP8 + 四步 LoRA |
| `dmd/minimax_h3_int8_4step.json` | `run_minimax_h3_t2av_parallel.sh` | SP4，INT8 + 四步 LoRA |
| `dmd/minimax_h3_fp8_8step.json` | `run_minimax_h3_t2av_parallel.sh` | SP8，FP8 + 八步 LoRA |
| `dmd/minimax_h3_int8_convrot_8step.json` | `run_minimax_h3_t2av_parallel.sh` | SP8，INT8 ConvRot + 八步 LoRA |
| `dmd/minimax_h3_fp8_4step_5090.json` | `run_minimax_h3_t2av_parallel.sh` | SP8，5090，四步 LoRA |
| `dmd/minimax_h3_fp8_4step_5090_vae_fp8.json` | `run_minimax_h3_t2av_parallel.sh` | SP8，FP8 VAE + 四步 LoRA |
| `dmd/minimax_h3_fp8_4step_5090_vae_fp8_sla.json` | `run_minimax_h3_t2av_parallel.sh` | SP8，FP8 VAE + 配套的 SLA LoRA |
| `dmd/minimax_h3_fp8_4step_5090_vae_fp8_sol.json` | `run_minimax_h3_t2av_parallel.sh` | SP8，FP8 文本编码器/VAE + Sol-Attn + 四步 LoRA |
| `dmd/minimax_h3_ref2av_4step.json` | `run_minimax_h3_ref2av.sh`，按下文说明使用 8 个进程 | 参考任务四步 LoRA |

`run_minimax_h3_t2av_parallel.sh` 默认使用 SP4。切换并行配置时，需要修改 `--config_json`，并让 `torchrun --nproc_per_node` 和 `CUDA_VISIBLE_DEVICES` 与 JSON 保持一致。这些配置所需的进程数为 `tensor_p_size × seq_p_size`；未设置的并行维度按 1 计算。

| 所选 JSON 中的并行方式 | `CUDA_VISIBLE_DEVICES` | `--nproc_per_node` |
| --- | --- | --- |
| TP2 | `0,1` | `2` |
| SP4 或 TP2 × SP2 | `0,1,2,3` | `4` |
| SP8，包括 5090 配置 | `0,1,2,3,4,5,6,7` | `8` |

使用 8 GPU 参考任务 LoRA 配置时，在 `run_minimax_h3_ref2av.sh` 中设置 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`，并将 `python -m lightx2v.infer` 改为 `torchrun --standalone --nproc_per_node=8 -m lightx2v.infer`。保留其中的 `--task ref2av` 和参考输入参数。

两个单 GPU Sol 配置均默认输出 362 帧，尺寸为 `[768, 1344]`。在 `run_minimax_h3_t2av.sh` 中通过 `--config_json` 选择对应的 Sol JSON，即可使用这些默认规格。

所有启用编译的 MiniMax-H3 配置都同时设置了 `warmup: true`。普通配置与编译配置的默认输出规格一致，两份服务启动脚本也都可以使用。

### 视频编码

如需使用原先的 362 帧视频编码示例，在 `minimax_h3.json` 的副本中修改以下字段，再让启动脚本指向这份完整的 JSON：

```json
{
  "target_video_length": 362,
  "video_codec_options": {
    "preset": "ultrafast",
    "crf": "18"
  }
}
```

`video_codec_options` 属于启动配置，由现有的 MP4 编码器使用。尺寸和帧数仍可由请求覆盖，无需为每种输出规格单独保留一份配置。

## LoRA 与 Python 调用

将所选配置中量化权重和 LoRA 字段的 `/path/to/...` 替换为实际存在的本地文件。加载器不会根据仓库 ID 自动下载 LoRA。请使用对应任务的 LoRA，并让 `alpha` 与该权重的训练配置保持一致；基础任务、参考任务、SLA 和不同版本的 LoRA 不能仅因形状相同就相互替换。

基础任务四步 v1.0 LoRA 使用 `alpha=128`、`video_flow_shift=6` 和 `audio_flow_shift=3`。其他配置使用各自的 alpha 和 shift 设置。

根据 DiT 权重，在所选 JSON 中设置 `lora_dynamic_apply`：

| DiT 权重 | 支持的设置 | 行为 |
| --- | --- | --- |
| 原始 BF16 | `false` 或 `true` | `false` 在加载权重时合并 LoRA；`true` 在推理时动态应用 LoRA。 |
| 量化 FP8 或 INT8，包括 ConvRot | `true` | 在推理时动态应用 LoRA，不支持将 LoRA 合并进量化 DiT 权重。 |

只量化文本编码器或 VAE 不受上述合并限制。动态 LoRA 当前只接受一个适配器，并要求显式提供 `alpha`。现有模型加载器会拒绝不支持的组合。

共用配置 `dmd/minimax_h3_bf16_4step.json` 默认设置 `lora_dynamic_apply: true`，输出 362 帧，尺寸为 `[768, 1344]`。将该字段改为 `false` 即可使用加载时合并。在 `run_minimax_h3_t2av.sh` 中选择这份 JSON；如需更短、更小的输出，在推理命令中添加 `--target_shape 544 960` 和 `--num_frames 124`。

在 JSON 中将 `infer_steps` 设置为实际模型计算次数：四步推理填 `4`，八步推理填 `8`。Scheduler 会自动包含末尾零点。迁移旧版 MiniMax-H3 配置时，将原 `infer_steps` 减一（`5` → `4`、`9` → `8`、`30` → `29`），即可保持原有采样过程。仓库中的配置已完成同步。

使用 Python 时，先设置[示例](../../examples/minimax_h3/minimax_h3_t2av_dmd.py)中的 `MODEL_PATH`，以及所选 JSON 中的本地 LoRA 路径，再执行：

```bash
python examples/minimax_h3/minimax_h3_t2av_dmd.py
```

## 服务与 POST 请求

先启动基础任务服务，再从另一个终端发送请求：

```bash
bash scripts/minimax_h3/server/start_server.sh
```

```bash
python scripts/minimax_h3/server/post_t2av.py
python scripts/minimax_h3/server/post_i2av.py
python scripts/minimax_h3/server/post_l2av.py
python scripts/minimax_h3/server/post_fl2av.py
```

基础任务请求必须指定 `task`，因为服务加载的 transformer 支持四种任务。图片示例会将客户端本地文件编码为 Base64 后发送。

参考生成需要单独启动参考任务服务：

```bash
bash scripts/minimax_h3/server/start_server_ref2av.sh
```

```bash
python scripts/minimax_h3/server/post_ref2av.py
```

两份启动脚本默认都使用 8000 端口。如需同时运行两个服务，请分别设置 GPU、`--port` 和 `--metric_port`，并修改 POST 示例中的 URL。参考任务服务只接受 `ref2av`，因此请求可以省略 `task`；示例为便于理解仍显式填写。

参考请求可以同时提供 `image_path`、`video_path` 和 `audio_path`。同一类型包含多个文件时，使用逗号分隔的服务端本地路径，例如：

```json
{
  "task": "ref2av",
  "prompt": "Generate an audio-video scene following the references.",
  "image_path": "/path/to/character.jpg,/path/to/scene.jpg",
  "video_path": "/path/to/motion.mp4",
  "audio_path": "/path/to/voice.wav",
  "seed": 42,
  "save_result_path": "./minimax_h3_references.mp4"
}
```

音频必须与图片或视频一起提供。Runner 最多接受 9 张图片、3 个视频、共计 12 个参考素材；参考音频最多 3 个，视频中自带的音轨也计入此限制。单张图片还支持 Base64 和 HTTP(S) URL；输入视频使用服务端本地路径。参考图片默认按发布的 Diffusers 尺寸规则预处理；也可在 JSON 中显式设置 `reference_image_resize_mode: "match"`，将参考图片面积限制在输出画布面积以内。

POST 会在推理完成前返回任务 ID。随后可查询状态，并在任务完成后下载结果：

```bash
curl http://localhost:8000/v1/tasks/TASK_ID/status
curl --fail http://localhost:8000/v1/tasks/TASK_ID/result -o minimax_h3.mp4
```

`save_result_path` 建议使用 `./minimax_h3_t2av.mp4` 这样的相对路径。服务会相对于其输出目录解析该路径，并通过下载接口返回文件。省略路径或传入 `null` 都会跳过保存，该请求将没有可下载的结果。

## 请求参数与启动配置

- 启动配置：模型路径、默认任务、权重/LoRA、算子、卸载、并行、编译和预热。Prompt、媒体路径、seed 和输出路径应放在 Python 调用、CLI 命令或 POST 请求体中。
- 输出默认值：JSON 设置 `target_video_length` 和 `target_height`/`target_width`。请求可通过 `num_frames` 和 `target_shape`（`[高度, 宽度]`）覆盖。宽高必须是 32 的倍数。帧数向上对齐到 `17*n+5`，支持的对齐后帧数范围为 124 到 362；例如，125 会调整为 141。
- Seed：省略或传入 `null` 时默认使用 42；显式提供的非负整数按原值使用，包括 0。
- 保存：每个 CLI 示例都显式提供输出路径，移除该参数即跳过文件保存，服务采用相同规则。输出为 MP4，包含 24 FPS 视频和 32 kHz 立体声音频。

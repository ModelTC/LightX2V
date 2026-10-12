# MiniMax-H3

[English](README.md) | [简体中文](README_zh.md)

运行前，设置所选脚本中的 `lightx2v_path`、`model_path` 和 `CUDA_VISIBLE_DEVICES`。

## 模型权重与任务

从官方 Hugging Face 仓库 [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) 下载 Diffusers 格式权重，`model_path` 指向本地模型目录。

| 任务 | 请求输入 | Transformer |
| --- | --- | --- |
| `t2av` | `prompt` | `transformer` |
| `i2av` | `prompt`、`image_path`（首帧） | `transformer` |
| `l2av` | `prompt`、`last_frame_path` | `transformer` |
| `fl2av` | `prompt`、`image_path`、`last_frame_path` | `transformer` |
| `ref2av` | `prompt`、参考图片和/或视频，以及可选的参考音频 | `transformer_ref` |

启动时须指定 `--model-variant`：`fl2av` 支持表中前四种任务，`ref2av` 支持参考任务。请求通过 `task` 选择任务，基础分支切换任务无需重新加载。两个分支可共用推理 JSON，LoRA 和量化权重须与分支匹配。

## AdaLN 缓存

通过 `use_adaln_cache: true` 启用 AdaLN 缓存。启用缓存时，先用相同模型、配置、步数和 flow shift 生成持久化缓存：

```bash
bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh
```

缓存脚本中的 `lightx2v_path`、`model_path`、`--config_json` 和 `--model-variant` 与推理保持一致。JSON 中须设置 `adaln_cache_dir`，示例配置使用 `~/.cache/lightx2v/adaln`。

已有匹配缓存可复用。

## 离线推理

五个任务脚本共用 `configs/minimax_h3/minimax_h3.json`：单卡 BF16，默认输出 124 帧、`[高度, 宽度] = [544, 960]`。

```bash
bash scripts/minimax_h3/run_minimax_h3_t2av.sh
bash scripts/minimax_h3/run_minimax_h3_i2av.sh
bash scripts/minimax_h3/run_minimax_h3_l2av.sh
bash scripts/minimax_h3/run_minimax_h3_fl2av.sh
bash scripts/minimax_h3/run_minimax_h3_ref2av.sh
```

基础配置使用 SageAttention2 和 SGL；注意力、量化和并行配置须与设备及已安装算子匹配。

通过 `--config_json` 选择配置，下表列出 `configs/minimax_h3/` 中的代表配置。单卡使用对应任务脚本；SP、TP 和混合并行参考 `run_minimax_h3_t2av_parallel.sh`，其他任务同步设置分支、任务及输入。

| `configs/minimax_h3/` 下的配置 | 运行方式 |
| --- | --- |
| `minimax_h3.json` | 单 GPU BF16 |
| `minimax_h3_compile.json` | 编译并在启动时预热 |
| `minimax_h3_block_offload.json` | 单 GPU BF16，分块卸载 |
| `minimax_h3_sp.json` | SP4 |
| `minimax_h3_tp.json` | TP2 |
| `minimax_h3_tp_sp.json` | TP2 × SP2 |
| `fp8/minimax_h3.json` | 单 GPU DiT FP8 |
| `dmd/minimax_h3_bf16_4step.json` | 单 GPU BF16，四步 LoRA |
| `dmd/minimax_h3_fp8_4step.json` | SP4，FP8 + 四步 LoRA |
| `dmd/minimax_h3_ref2av_4step.json` | SP8，参考任务四步 LoRA |

多卡脚本默认 SP4，切换配置时同步设置 `CUDA_VISIBLE_DEVICES` 和 `--nproc_per_node`，与配置中的并行规模保持一致。

编译使用 `minimax_h3_compile.json`，启用 `use_compile`、`warmup`，并设置 `rope_type: "torch_real_rope"`；服务也可选用。

在推理 JSON 中添加 `video_codec_options` 设置 MP4 编码选项，例如：

```json
{
  "video_codec_options": {
    "preset": "ultrafast",
    "crf": "18"
  }
}
```

## LoRA

| DiT 权重 | `lora_dynamic_apply` | 行为 |
| --- | --- | --- |
| 原始 BF16 | `false` 或 `true` | `false` 在加载权重时合并 LoRA；`true` 在推理时动态应用 LoRA。 |
| 量化 FP8 或 INT8，包括 ConvRot | `true` | 在推理时动态应用 LoRA，不支持将 LoRA 合并进量化 DiT 权重。 |

基础任务四步 v1.0 LoRA 示例：在 `minimax_h3.json` 中添加或修改以下字段，`path` 填本地 LoRA 路径：

```json
{
  "infer_steps": 4,
  "video_flow_shift": 6.0,
  "audio_flow_shift": 3.0,
  "h3_step_update": "training_euler",
  "lora_dynamic_apply": true,
  "lora_configs": [
    {
      "path": "/path/to/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
      "strength": 1.0,
      "alpha": 128
    }
  ]
}
```

## 服务与 POST 请求

基础服务设置 `--model-variant fl2av`，启动时不传 `--task`。按卡数选择入口，模型加载后在另一终端提交请求。

单卡：

```bash
bash scripts/minimax_h3/server/start_server.sh
```

四卡 SP4 使用 `minimax_h3_sp.json`：

```bash
bash scripts/minimax_h3/server/start_server_parallel.sh
```

单卡和多卡服务共用以下 POST 客户端：

```bash
python scripts/minimax_h3/server/post_t2av.py
python scripts/minimax_h3/server/post_i2av.py
python scripts/minimax_h3/server/post_l2av.py
python scripts/minimax_h3/server/post_fl2av.py
```

基础请求须指定 `task`，输入见任务表。图片示例上传客户端本地文件的 Base64，也可传服务端可访问的绝对路径。

## 请求参数与启动配置

- **启动与请求**：启动时设置模型路径、分支、权重／LoRA、算子、卸载、并行、编译和预热；通过 Python、CLI 或 POST 传入 `prompt`、媒体路径、`seed` 和输出路径。
- **输出规格**：JSON 提供 `num_frames`、`size` 的默认值，请求可覆盖。`size` 为 `[高度, 宽度]`，宽高须为 32 的倍数；帧数向上对齐到 `17*n+5`，对齐后须在 124–362 之间，如 125 → 141。输出为 MP4，包含 24 FPS 视频和 32 kHz 立体声音频。
- **随机种子**：`seed` 省略或传 `null` 时使用 42；可指定包含 0 在内的非负整数。
- **结果保存**：CLI 不指定输出路径时跳过保存。服务的 `save_result_path` 建议使用 `./minimax_h3_t2av.mp4` 等相对路径，按服务输出目录解析；省略或传 `null` 时跳过保存，已保存的结果可下载。

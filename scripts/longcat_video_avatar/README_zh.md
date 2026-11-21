# LongCat-Video-Avatar 1.6

[English](README.md)

`longcat_video_avatar` 使用 `s2v` 任务生成单人或双人说话视频，默认每段 93 帧、25 fps。

## 准备

将两份权重放在同一父目录下。`model_path` 指向 `LongCat-Video-Avatar-1.6`。同级的 `LongCat-Video` 提供 `text_encoder/`、`tokenizer/` 和 `vae/`。[Avatar 权重](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.6) · [共享权重](https://huggingface.co/meituan-longcat/LongCat-Video)

H800 示例使用 BF16、FlashAttention 3 和 FFmpeg。在仓库根目录运行 `pip install -e '.[longcat-avatar]'` 安装音频依赖，并在所用 Bash 脚本顶部设置 `lightx2v_path`、`model_path` 和 `CUDA_VISIBLE_DEVICES`。

## HTTP 服务

启动服务（双卡用 [start_server_sp2.sh](server/start_server_sp2.sh)）。

```bash
bash scripts/longcat_video_avatar/server/start_server.sh
```

在另一终端发送单人请求（双人用 [post_s2v_multi.py](server/post_s2v_multi.py)）。

```bash
python scripts/longcat_video_avatar/server/post_s2v_single.py
```

## 命令行推理

从仓库根目录选择一个示例运行。

```bash
# 单人
bash scripts/longcat_video_avatar/run_longcat_video_avatar_single.sh
# 双人同框
bash scripts/longcat_video_avatar/run_longcat_video_avatar_multi.sh
# 双卡 Ulysses 序列并行
bash scripts/longcat_video_avatar/run_longcat_video_avatar_s2v_sp2.sh
```

## 共用参数

同一个参数在命令行中写作 `--num_segments 2`，在 HTTP 请求中写作 `"num_segments": 2`。

| 需求 | 参数 |
| --- | --- |
| 单人说话 | 提供提示词和一路音频。示例默认带参考图，省略 `image_path` 即可只用提示词和音频生成。 |
| 双人说话 | 提供一张合照，按默认的左、右人物顺序将两条音轨传入 `audio_path` 数组。`audio_type` 默认为 `para`，同时播放至较短音轨结束。`add` 按顺序播放。 |
| 续写 | 三种输入均可将 `num_segments` 设为 2。后续每段复用前段末尾 13 帧，默认两段共 173 帧（6.92 秒）。 |
| 按档位选尺寸 | `resolution` 可选 `480p`（默认）或 `720p`。有图时选档内最接近参考图宽高比的尺寸。无图时分别输出 832 × 480 或 1280 × 768（宽 × 高）。 |
| 指定高宽 | `--size 512 768` 或 HTTP 的 `"size": [512, 768]` 表示高 512、宽 768。高宽须为 16 的倍数，双卡时宽须为 32 的倍数。 |

尺寸优先级为请求中的 `size`、请求中的 `resolution`、部署配置中的 `request_defaults`。[单卡配置](../../configs/longcat_video_avatar/h800/s2v_bf16.json) · [双卡配置](../../configs/longcat_video_avatar/h800/s2v_bf16_sp2.json)

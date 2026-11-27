# H3 API 视频整理与 8 卡 VAE 编码

当前默认流程：**重新纳入已有 API 视频（包括原始 prompt）→ 整理统一 JSON → 仅补视频/音频 VAE 编码**。不运行 DiT、不重新生成 teacher 视频、不编码 text/image condition。原有 `prepare_minimax_h3_dataset_8gpu.sh` 入口也转发到这条流程。

只有同一条件、实际 prompt 和尺寸可对应的素材才会关联；不会把 original 请求的视频强行配给 enhanced condition。没有匹配视频的记录保留，teacher 字段仍可能为 null，并在最终报告列出。

## 停止旧数据任务

在新机器执行，只匹配本数据清单的补齐任务，不停止训练：

```bash
pkill -TERM -f '[p]repare_minimax_h3_dataset_8gpu.sh' || true
pkill -TERM -f '[p]repare_minimax_h3_dataset_vae_8gpu.sh' || true
pkill -TERM -f '[c]omplete_h3_dataset.py.*--manifest /mnt/miaohua/gushiqiao/datasets/fl2av_dataset/h3_self_gen/dataset.json' || true
while pgrep -f '[c]omplete_h3_dataset.py.*--manifest /mnt/miaohua/gushiqiao/datasets/fl2av_dataset/h3_self_gen/dataset.json' >/dev/null; do sleep 2; done
```

## 更新代码并启动

将新版 `h3_dataset_tools.zip` 上传到 `/mnt/miaohua/gushiqiao/h3_dataset_tools.zip` 后：

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate
unzip -o /mnt/miaohua/gushiqiao/h3_dataset_tools.zip -d .

export H3_DATA_ROOT=/mnt/miaohua/gushiqiao/datasets
export H3_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3
export H3_GPUS=0,1,2,3,4,5,6,7
nohup bash lightx2v_train/scripts/prepare_minimax_h3_dataset_vae_8gpu.sh \
  > "$H3_DATA_ROOT/h3_dataset_vae_pipeline.log" 2>&1 < /dev/null &
echo $! > "$H3_DATA_ROOT/h3_dataset_vae_pipeline.pid"
```

首次运行自动备份 `dataset.json`、`report.json` 为 `*.before_api_video_vae.json`，即使已有旧清单也会重新整理。已有文件利用复制索引复用，不删除源数据或旧 DiT 中间结果。成功整理后写入 `.pipeline/api_video_vae_v1.organized`，重跑同一命令即可续传。

如果后续补充了新的源数据，需要再次扫描：先停止数据任务，再删除这一个 marker 后重跑；不要删除 data 目录。

依赖：原 H3 Python 环境、`unzip`、`flock`、`ffprobe`、`ffmpeg`。仅需模型的 `vae/` 与 `audio_vae/`；不需要 FA3 或加载 DiT。

## 行为

| 数据现状 | 操作 |
|---|---|
| 视频和两个 AV latent 都有 | 直接复用 |
| 视频有，某个 latent 缺失 | 只编码缺失项 |
| 没有匹配视频 | 保留 null 和现有字段，跳过，不做 teacher 采样或视频重建 |
| condition 缺失 | 不补 condition，仍保留缺失标记 |

VAE 使用 `posterior.mode()`，保存归一化 BF16 latent；video/audio 对应每条记录自己的尺寸与帧数。视频 latent 为 `[24,37,H/16,W/16]`（124 帧），音频 latent 为 `[2,32,207]`。已存在且关联正确的 latent 不重复编码。

新进度记录：`data/vae_completion/`。新增 latent：`data/vae_teacher_video_latent/` 和 `data/vae_teacher_audio_latent/`。旧 `data/completion/` 的 DiT receipts、packed latent 不读取、不覆盖、不删除。

## 日志和结果

```bash
tail -f /mnt/miaohua/gushiqiao/datasets/h3_dataset_vae_pipeline.log
tail -f /mnt/miaohua/gushiqiao/datasets/fl2av_dataset/h3_self_gen/logs/vae_completion/gpu0.log
cat /mnt/miaohua/gushiqiao/datasets/fl2av_dataset/h3_self_gen/vae_completion_report.json
```

GPU 日志会记录 `video_latent`、`audio_latent`、`complete`。启动摘要应为 `mode: encode_only`、`model_evaluations: 0`。`skipped_missing_video` 单独统计无视频记录，不应误认为编码失败。八个 worker 结束后才将 receipts 合并回 `dataset.json`；运行中看每卡日志，不根据主 JSON 中尚未回写的 null 判断进度。

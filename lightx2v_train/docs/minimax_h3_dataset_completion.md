# H3 数据迁移、整理与 8 卡补齐

先用单独的 rsync 命令复制旧目录，再用一个 bash 入口自动完成：**整理统一 JSON → 8 卡补齐缺失视频和 AV latent**。后续数据全部使用 `/mnt/miaohua/gushiqiao/datasets/`。这里只处理 T2AV / I2AV，不启动训练；后续 PDMD 学生使用 4 步。

## 单独复制旧目录

```bash
rsync -aH --no-owner --no-group --partial --info=progress2 \
  /mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_teacher_latents_5s \
  /mnt/miaohua/gushiqiao/datasets/
```

源目录末尾不加 `/`，复制后得到 `datasets/minimax_h3_teacher_latents_5s/`。原目录保留；中断后重复同一 rsync 命令续传。复制成功后再运行下面的 bash。

## Bash 自动整理与补齐

先同步新版 `lightx2v_train/scripts/prepare_minimax_h3_dataset_8gpu.sh` 及配套 Python 脚本。在新机器运行：

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate

nohup bash lightx2v_train/scripts/prepare_minimax_h3_dataset_8gpu.sh \
  > /mnt/miaohua/gushiqiao/datasets/h3_dataset_pipeline.log 2>&1 < /dev/null &
echo $! > /mnt/miaohua/gushiqiao/datasets/h3_dataset_pipeline.pid
```

复制需要 `rsync`；bash 流程需要 `unzip`、`flock`、`ffprobe` 和原有 H3 Python 环境。默认使用 0–7 卡，每卡独立 worker，单卡需容纳 BF16 H3 DiT。请在这 8 张卡可用时启动。

## 自动执行的步骤

1. **整理**：自动解压 I2AV 图片和 teacher 视频 ZIP，关联 condition、图片、视频及 AV latent，检查 latent shape，复制引用的素材到 `fl2av_dataset/h3_self_gen/data/`，生成 `dataset.json` 和 `report.json`。

   所有训练资产路径必须位于 miaohua 内；历史 provenance 可以保留旧路径作为追溯信息。关联不明确的记录不会强行配对。出现被拒绝的源记录时停止在整理阶段，报告位置会写入日志。

2. **补齐**：整理成功后自动启动 8 个 GPU worker。完成后回写同一个 JSON，不需要手动启动第二次命令。

| 当前记录 | 操作 |
| --- | --- |
| condition 缺失 | 使用记录中的实际 prompt 编码；I2AV 同时编码首帧 |
| 视频有，AV latent 缺一个或两个 | 只补缺失的 VAE 编码 |
| 视频缺失，两个 AV latent 都有 | 从现有 latent 重建视频，保留已有 latent |
| 视频缺失，AV latent 不完整 | 使用本地 H3 生成视频，再编码完整 AV latent |
| 全部齐全 | 跳过 |

每条记录保留自己的 `width`、`height`、`num_frames`、`fps`。API 使用非增强 prompt 生成的记录仍保留，仅暂时清空 teacher 视频及两个 latent 字段，再由本地 H3 补齐；官方 selfgen 数据不按此 API 规则清空。

Teacher 使用原始 dense FL2AV H3，不加载 DCM / PDMD / MoE / LoRA。沿用旧出片设置：**30 个 sigma 点、29 次模型前向，视频／音频 shift=12／3**。4 步是后续学生训练参数。DiT 和 VAE 分阶段加载，不同时常驻显存。

## 日志与输出

```bash
# 整理与补齐阶段进度
 tail -f /mnt/miaohua/gushiqiao/datasets/h3_dataset_pipeline.log

# 每卡详细进度，gpu0 可换成 gpu1～gpu7
 tail -f /mnt/miaohua/gushiqiao/datasets/fl2av_dataset/h3_self_gen/logs/completion/gpu0.log

# 最终汇总
 cat /mnt/miaohua/gushiqiao/datasets/fl2av_dataset/h3_self_gen/completion_report.json
```

最终数据清单：

```text
/mnt/miaohua/gushiqiao/datasets/fl2av_dataset/h3_self_gen/dataset.json
```

阶段日志都在 `h3_self_gen/logs/`：`extract_images.log`、`organize.log`、`completion.log`，以及 `completion/gpu*.log`。最终报告中 `remaining` 为空、所有 worker 退出码为 0 才表示全部完成。

## 续跑

复制中断时重跑 rsync；整理或补齐中断时重跑 bash。Bash 完全不访问旧数据目录：

- `dataset.json` 不存在：继续整理并复用已复制素材。
- `dataset.json` 已存在：保留，不重新整理覆盖。
- 补齐记录完成：跳过；部分完成：读取逐条进度继续。

首次补齐前自动备份 JSON；逐条进度写在 `data/completion/`。单条失败会记录 sample_id、阶段和 traceback，其他记录继续；主进程合并已完成部分并返回非零退出码。修复失败原因后重跑即可。

不要在运行期间手动编辑 JSON，也不要绕过入口再次运行 organizer 覆盖已补齐的数据。数据文件路径已填写但文件丢失时明确报错，不自动改用其他文件。

## 可选路径配置

默认代码和 Python 环境沿用旧位置；**迁移的是数据目录**，不会搬动模型和环境。以下变量只在与你机器实际路径不符时覆盖：

```bash
export H3_DATA_ROOT=/mnt/miaohua/gushiqiao/datasets
export H3_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3
export H3_FA3_KERNEL_PATH=/mnt/lm_data_afs/gushiqiao/cache/kernels/kernels--kernels-community--flash-attn3/snapshots/43f0bd269777115d94ff826e0d113ce9c1c9087b
export H3_GPUS=0,1,2,3,4,5,6,7
```

若文本编码器独立存放，设置 `H3_CONDITION_MODEL_PATH` 为包含 `text_encoder/`、`tokenizer/`、`processor/` 的目录；默认使用 `H3_MODEL_PATH`。FA3 使用显式本地内核路径，避免联网加载。

本机已用真实权重验证 T2AV / I2AV 小尺寸生成链路、1344×768/124 帧视频的补编码、已有资产复用；全量 8 卡任务需要在新机器启动。

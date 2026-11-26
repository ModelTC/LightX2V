# H3 数据整理与 8 卡补齐

当前只处理 FL2AV 基座使用的 T2AV / I2AV 数据。训练配置不在本次流程内，后续 PDMD 学生采用 4 步。

## 一次性整理

把更新后的 LightX2V 代码同步到新机器，进入现有 H3 Python 环境。需要 `torch`、`ffprobe`、`unzip`。下列路径沿用已有集群部署；代码目录与模型目录如果迁移了，修改对应变量。

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate

export H3_CODE_ROOT="$PWD"
export H3_DATA_ROOT=/mnt/miaohua/gushiqiao/datasets
export H3_DATASET="$H3_DATA_ROOT/fl2av_dataset/h3_self_gen"

unzip -n "$H3_DATA_ROOT/h3_teacher_i2av_images_9458.zip" -d "$H3_DATA_ROOT"

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u \
  lightx2v_train/data_process/minimax_h3/organize_h3_dataset.py \
  --datasets-root "$H3_DATA_ROOT" \
  --output "$H3_DATASET" \
  --teacher-latents /mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_teacher_latents_5s \
  --extract-teacher-zip \
  --check-latents \
  > "$H3_DATA_ROOT/organize_h3.log" 2>&1
```

`--teacher-latents` 指向之前的 127 GB 缓存，直接复制其中引用的文件进新目录，不必先整体搬运一次。如果不再有这个目录，删除这一整项参数；缺少的 latent 会保留为 `null` 并由下一步补齐。

输出：

- `h3_self_gen/dataset.json`：JSON 数组，训练资产均为绝对路径。
- `h3_self_gen/data/`：condition、图片、视频和 AV latent。
- `h3_self_gen/report.json`：缺失情况、关联歧义、尺寸和 prompt 检查结果。

原始输入文件保留，不删除、不搬走。数据关联不明确时保留独立记录并标注，不能按行号强行配对。每条记录保留自己的 `width`、`height`、`num_frames`、`fps`，没有统一缩放到一个分辨率。

API 使用非增强 prompt 生成的视频暂不使用：**记录、prompt、condition、图片仍保留**，仅将 teacher 视频和两个 latent 字段置为 `null`。官方 selfgen 数据不应用这条 API 排除规则。

## 8 卡补齐

每个 GPU 一个独立进程，无 FSDP 通信；单卡需容纳完整 BF16 H3 DiT。原始 H3 参数不加载 DCM / PDMD / MoE / LoRA。不同阶段卸载前一个模型后再加载下一个，不同时保留 DiT 和 VAE。

| 当前记录 | 处理 |
| --- | --- |
| condition 缺失 | 按该记录的实际 prompt 编码；I2AV 同时编码首帧 |
| 视频有，AV latent 缺一个或两个 | 只补缺失的 VAE 编码 |
| 视频缺失，两个 AV latent 都有 | 从现有 latent 重建视频，保留已有 latent |
| 视频缺失，AV latent 不完整 | 根据 condition 生成新视频，随后从该视频编码完整 AV latent |
| 全部齐全 | 跳过 |

Teacher 采样沿用已验证的 H3 出片入口：`MiniMaxH3Scheduler.set_timesteps(30)`，即 **30 个 sigma 点、29 次模型前向**，视频 shift=12、音频 shift=3，不对物理 sigma 二次 shift。4 步是后续学生训练参数，与 teacher 补齐无关。

在一台有 8 张空闲卡的机器运行一次：

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate

export H3_DATASET=/mnt/miaohua/gushiqiao/datasets/fl2av_dataset/h3_self_gen
export H3_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3
export H3_FA3_KERNEL_PATH=/mnt/lm_data_afs/gushiqiao/cache/kernels/kernels--kernels-community--flash-attn3/snapshots/43f0bd269777115d94ff826e0d113ce9c1c9087b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$H3_DATASET/logs"
nohup python -u lightx2v_train/data_process/minimax_h3/complete_h3_dataset.py \
  --manifest "$H3_DATASET/dataset.json" \
  --model-path "$H3_MODEL_PATH" \
  --gpus 0,1,2,3,4,5,6,7 \
  --schedule-points 30 \
  --attention-kernel-path "$H3_FA3_KERNEL_PATH" \
  > "$H3_DATASET/logs/completion.log" 2>&1 < /dev/null &
echo $! > "$H3_DATASET/logs/completion.pid"
```

FA3 路径指向之前训练使用的本地 Hub 内核 snapshot，需要该环境已安装匹配的 CUDA / PyTorch 构建。显式传入本地路径可避免运行时联网查找内核。可在启动前给相同命令追加 `--plan`，只输出缺失计数，不加载模型。

如果 DiT/VAE 与文本编码器分开存放，追加 `--condition-model-path /包含text_encoder与tokenizer和processor的模型根目录`。默认三者位于 `--model-path` 下；不会自动改用其他模型的文本编码器。

```bash
tail -f "$H3_DATASET/logs/completion/gpu0.log"
cat "$H3_DATASET/completion_report.json"
```

每张卡分别记录 `gpu0.log` 至 `gpu7.log`。末尾报告的 `remaining` 为空且所有 `worker_exit_codes` 为 0，才表示全部补齐。单条失败会记录 sample_id、阶段和 traceback，其他记录继续处理；最终命令返回非零状态，避免把部分成功当成全部完成。

## 回写与续跑

原 JSON 自动备份为 `dataset.before_completion.<hash>.json`。逐条进度写入 `data/completion/`；所有 worker 退出后，主进程将成功和部分完成的路径合并回同一个 `dataset.json`。失败记录的缺失字段仍为 `null`。

补齐后重复执行 **completion 命令** 可续跑，已完成记录不会重复生成。不要重跑最初的 organize 命令覆盖补齐后的 JSON，也不要在补齐期间手动编辑 JSON。文件路径存在于 JSON 但磁盘文件丢失时明确报错，不偷偷重生成。

新生成 teacher 的 `teacher_prompt` 使用实际 condition 文本，`teacher_origin` 为 `local_h3`；新增的 AV latent 来自保存后的 MP4，后验使用 mode、输出 BF16，记录生成参数和编码信息。旧视频仅补编码时保留其原有 prompt 来源，不能据此断言 API 内部没有改写过 prompt。

这一步不修改训练代码、不启动训练。待数据补齐后，再接入 4 步 PDMD + GAN。

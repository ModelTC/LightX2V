# 导入 LightX2V Ref teacher 最终 latent，供后续 GAN 使用

`data_process/minimax_h3/import_lightx2v_ref_teacher_latents.py` 将推理服务导出的 H3 Ref teacher 数据发布成现有 Ref GAN 可以直接读取的清单。脚本不运行模型或训练，也不复制 latent。输出路径指向推理服务器共享目录中的原文件。

正式采集使用 `schema=h3_teacher_final_v1`、`export_mode=final`：完成 29 次更新后，只保存 sigma=0 的最终 `video_latents.pt` / `audio_latents.pt` 和配对的 `condition.pt`，再写入 `manifest.json`；采集器另写完成记录。每条样本仅保存一个最终 AV 状态，不保存初始噪声或中间 `state_*.pt`。完整的双流 sigma 数组仍作为 30 个端点的小型元数据保留。

导入器也兼容旧的 `h3_teacher_trajectory_v1` 完整轨迹；旧格式仍必须有全部 30 个有序且非空的状态文件，不能拿缺失状态的旧导出冒充完整数据。两种格式发布的 GAN 数据都只读取条件和最终 AV 三条路径。

## 手工迁移

训练仓库需要复制以下三个新增文件，保留相对路径：

```text
lightx2v_train/data_process/minimax_h3/import_lightx2v_ref_teacher_latents.py
lightx2v_train/data_process/minimax_h3/test_import_lightx2v_ref_teacher_latents.py
lightx2v_train/docs/minimax_h3_ref_teacher_trajectory_import.md
```

不需要修改 `dmd2_gan_data.py`、训练目标或模型代码。推理端的导出支持和双节点采集脚本由推理仓库单独迁移；正式采集应输出 `schema=h3_teacher_final_v1` 和 `index.node0.jsonl` / `index.node1.jsonl`。

## 发布 5000 条正式清单

在能读取原始 JSONL、两个节点索引和所有导出目录的训练服务器执行。下面路径对应默认的采集目录；如果采集时换过输出目录，同步修改 `EXPORT_ROOT`。

```bash
cd /path/to/training/LightX2V/lightx2v_train
EXPORT_ROOT=/mnt/devsft_afs_2/gushiqiao/outputs/data/omni_ref_teacher29_decfg_5k
python data_process/minimax_h3/import_lightx2v_ref_teacher_latents.py \
  --raw-jsonl /mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/omni_r2v_image_only_100000_ir.jsonl \
  --export-index "$EXPORT_ROOT/index.node0.jsonl" "$EXPORT_ROOT/index.node1.jsonl" \
  --media-root /mnt/lm_data_afs/gushiqiao/datasets/omini-r2v \
  --output-dir "$EXPORT_ROOT/ref2av_imageonly" \
  --expected-count 5000
```

只有恰好 5000 条唯一、完整且相互一致的导出全部通过校验，才发布新清单。任意一条失败都会中止；不会静默跳过坏样本。重新导入时若校验失败，会保留原有合法清单。

少量 smoke 数据可以省略 `--expected-count 5000`，或改为实际预期数量。省略时，完成回执只声明本次索引中实际导入的数量，并单独记录原始数据总数和未选中数量，不会宣称已完成 5000 条。

输出：

| 文件 | 用途 |
| --- | --- |
| `metadata.jsonl` | 条件和最终干净 AV latent 的完整配对；供 GAN 使用 |
| `condition_metadata.jsonl` | 同一批条件，去除目标 AV 路径和形状；供条件流使用 |
| `metadata.complete.json` | 完整配对清单 SHA-256、真实数量、来源和验证范围 |
| `condition_metadata.complete.json` | 条件清单对应的完成回执 |

清单包含原始 `source_id` / `source_index`、`enhanced_prompt`、参考图顺序、条件/latent 指纹、124 帧/24 fps、实际方向与分辨率、2–6 张图片参考数量，以及现有采样器需要的成本字段。`packed_sequence_tokens_124` 使用实际目标尺寸计算：

```text
prompt_token_count + reference_video_rows + 37*(height/32)*(width/32) + 414
```

`teacher_export_schema` / `teacher_export_mode` 明确记录导出格式，`teacher_saved_states` 记录实际保存的状态数：最终格式为 1，旧完整轨迹为 30。兼容字段 `teacher_trajectory_states` 也使用实际保存数，不再固定为 30；`teacher_trajectory_manifest` 继续指向原导出的 `manifest.json`。`teacher_final_state_index=29` 与两个 `teacher_final_*_sigma=0` 记录 GAN 目标的最终位置。

`source_index` 与原训练 cache builder 一致，是**忽略空行后的零基 JSON 记录序号**。`source_id` 按原 builder 的 `sample_id` → `metadata_id` → `id` → `source_index` 字段回退规则取得。参考图解析沿用原 builder 的顺序语义：显式 `order` / `reference_index` 必须已经连续；不会重新排序或使用额外的 `valid_*` 字段删除参考图。caption 必须与原始 `enhanced_prompt.strip()` 完全匹配。

## 校验范围

默认导入只使用 Python 标准库，流式读取原始 JSONL，只保留选中的原始记录。它检查最终写入的非空 `manifest.json`、条件和最终 AV 文件；校验 schema、FP32/normalized 声明、124 帧/24 fps、实际双流 sigma 严格递减到 0、`reference_blend`、29 步、video12/audio3 shift、图片顺序和参考几何/成本。所有样本须具有一致的导出格式、模型路径、模型身份、DeCFG 来源和生成设置。不同节点可有不同的启动配置 SHA，它作为来源信息保留。

最终格式必须显式声明 `export_mode="final"`、`num_steps=29`、`num_states=1`、`state_paths=[]`、`final_state_index=29`、`final_video_sigma=0.0`、`final_audio_sigma=0.0`，并保留各 30 个端点的 `video_sigmas` / `audio_sigmas`。导入器不会寻找任何 `state_*.pt`；非最终步、非零终点、缺失最终文件或缺少必要字段均拒绝发布。旧完整轨迹格式继续逐一检查 `state_000.pt` 至 `state_029.pt`。

默认模式依赖导出器的 tensor 声明，**不读取 tensor 内容**。需要逐条检查时，在上述命令追加：

```text
--check-tensors
```

该选项需要 PyTorch，在 CPU 上逐条读取 `condition.pt` 和最终两个 AV 文件，检查身份、caption、参考路径/数量、packed reference shape、prompt embedding/tag、归一化、FP32 最终 AV 形状与有限值。导入旧格式时也不会反序列化 30 个轨迹状态。回执会明确记录实际验证范围。

最终视频目标为 `[24,37,H/16,W/16]`，音频目标为 `[2,32,207]`。目录和原始文件必须持续挂载在清单记录的绝对路径上。正式训练前可对少量 smoke 条目先执行 `--check-tensors`，再对正式清单使用默认的轻量校验或按需要完整检查最终 AV。

后续现有 Ref GAN 启动配置中的 `training.dmd.gan.real_cache`（或启动脚本的 `H3_REF_GAN_REAL_CACHE`）指向 `ref2av_imageonly/metadata.jsonl` 即可。导入完成不等于启动训练；原有模型、checkpoint、GPU 数量和采样配置的 preflight 仍按原训练流程检查。

## CPU 回归测试

在训练仓库根目录运行：

```bash
python -m unittest discover \
  -s lightx2v_train/data_process/minimax_h3 \
  -p test_import_lightx2v_ref_teacher_latents.py -v
```

无 PyTorch 时运行最终格式、旧完整轨迹、原始映射、重复/混合模型/损坏导出拒绝、数量门槛和完成回执测试；安装 PyTorch 后还会对两种格式验证最终 tensor，以及现有 `_deferred_dataset` / `DMD2GanRealStream` 接口，并验证最终格式的 `--check-tensors` CLI。测试使用微型 CPU fixture，不加载真实模型或运行训练。

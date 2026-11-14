# 导入 LightX2V Ref teacher 轨迹，供后续 GAN 使用

`data_process/minimax_h3/import_lightx2v_ref_teacher_latents.py` 将推理服务导出的 H3 Ref teacher 数据发布成现有 Ref GAN 可以直接读取的清单。脚本不运行模型或训练，也不复制 latent。输出路径指向推理服务器共享目录中的原文件。

每条轨迹包含 29 次更新、30 个 AV 状态：`state_000.pt` 是初始噪声，`state_001.pt` 至 `state_028.pt` 是中间状态，`state_029.pt` 是 sigma=0 的最终状态。30 个状态全部保留；当前 GAN 只使用单独导出的最终 `video_latents.pt` / `audio_latents.pt` 和配对的 `condition.pt`。此导入不会改变 PDMD 目标，也不会把中间噪声当作 GAN 正样本。

## 手工迁移

训练仓库需要复制以下三个新增文件，保留相对路径：

```text
lightx2v_train/data_process/minimax_h3/import_lightx2v_ref_teacher_latents.py
lightx2v_train/data_process/minimax_h3/test_import_lightx2v_ref_teacher_latents.py
lightx2v_train/docs/minimax_h3_ref_teacher_trajectory_import.md
```

不需要修改 `dmd2_gan_data.py`、训练目标或模型代码。推理端的导出支持和双节点采集脚本由推理仓库单独迁移；它们应输出 `schema=h3_teacher_trajectory_v1` 和 `index.node0.jsonl` / `index.node1.jsonl`。

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

`source_index` 与原训练 cache builder 一致，是**忽略空行后的零基 JSON 记录序号**。`source_id` 按原 builder 的 `sample_id` → `metadata_id` → `id` → `source_index` 字段回退规则取得。参考图解析沿用原 builder 的顺序语义：显式 `order` / `reference_index` 必须已经连续；不会重新排序或使用额外的 `valid_*` 字段删除参考图。caption 必须与原始 `enhanced_prompt.strip()` 完全匹配。

## 校验范围

默认导入只使用 Python 标准库，流式读取原始 JSONL，只保留选中的原始记录。它检查最终写入的 `manifest.json`、30 个有序且非空的状态文件、条件和最终 AV 文件；校验 schema、FP32/normalized 声明、124 帧/24 fps、实际双流 sigma 严格递减到 0、`reference_blend`、29 步、video12/audio3 shift、图片顺序和参考几何/成本。所有样本须具有一致的模型路径、模型身份、DeCFG 来源和生成设置。不同节点可有不同的启动配置 SHA，它作为来源信息保留。

默认模式依赖导出器的 tensor 声明，**不读取 tensor 内容**。需要逐条检查时，在上述命令追加：

```text
--check-tensors
```

该选项需要 PyTorch，在 CPU 上逐条读取 `condition.pt` 和最终两个 AV 文件，检查身份、caption、参考路径/数量、packed reference shape、prompt embedding/tag、归一化、FP32 最终 AV 形状与有限值。它仍不反序列化 30 个轨迹状态，避免为了发布清单读取约 TB 级轨迹。回执会明确记录实际验证范围。

最终视频目标为 `[24,37,H/16,W/16]`，音频目标为 `[2,32,207]`。目录和原始文件必须持续挂载在清单记录的绝对路径上。正式训练前可对少量 smoke 条目先执行 `--check-tensors`，再对正式清单使用默认的轻量校验或按需要完整检查最终 AV。

后续现有 Ref GAN 启动配置中的 `training.dmd.gan.real_cache`（或启动脚本的 `H3_REF_GAN_REAL_CACHE`）指向 `ref2av_imageonly/metadata.jsonl` 即可。导入完成不等于启动训练；原有模型、checkpoint、GPU 数量和采样配置的 preflight 仍按原训练流程检查。

## CPU 回归测试

在训练仓库根目录运行：

```bash
python -m unittest discover \
  -s lightx2v_train/data_process/minimax_h3 \
  -p test_import_lightx2v_ref_teacher_latents.py -v
```

无 PyTorch 时运行 manifest、原始映射、重复/混合模型/损坏轨迹拒绝、数量门槛和完成回执测试；安装 PyTorch 后还会验证最终 tensor，以及现有 `_deferred_dataset` / `DMD2GanRealStream` 接口。测试使用微型 CPU fixture，不加载真实模型或运行训练。

# OpenPI π0.5-LIBERO 训练复现

本文说明如何在 LightX2V 中从官方 π0.5 base 权重开始 LIBERO fine-tuning，如何
恢复被中断的训练，以及如何用 EMA 权重完成定量评测。推理、LIBERO rollout 和 ROS
的完整说明见项目级 [OpenPI README](../../../../scripts/openpi/support/README.md)。

下面的命令默认在项目根目录执行：

```bash
cd /data/liuhongda/lightx2v_openpi
conda activate base
```

训练直接使用当前环境的 `python`。开始前应确认：

```bash
command -v python
```

## 1. 两个训练入口

训练脚本只分为两种互斥语义：

| 脚本 | 输入权重 | 恢复 optimizer/step | 用途 |
| --- | --- | --- | --- |
| `run_pi05_finetune_ema.sh` | 官方 π0.5 base PyTorch 权重 | 否 | 开始一轮新的 LIBERO fine-tuning |
| `run_pi05_resume_ema.sh` | LightX2V 完整训练 checkpoint | 是 | 从中断位置严格续训 |

这里的“新 fine-tuning”不是随机初始化。模型参数来自官方 π0.5 base；optimizer、
scheduler 和 step 从零开始，EMA 从初始模型复制。当前接入不提供随机初始化训练。

调用链如下：

```text
run_pi05_finetune_ema.sh / run_pi05_resume_ema.sh
  -> support/launch_pi05_libero.sh
  -> python -m torch.distributed.run
  -> lightx2v_train/train.py
  -> OpenPILiberoDataset
  -> OpenPIPi05LiberoModel
  -> OpenPIFlowMatchingTrainer
```

两个公开入口共用同一个 helper、同一份 YAML 和同一个 trainer，只有初始化/恢复方式
不同。`support/launch_pi05_libero.sh` 是内部实现，不需要直接运行。

## 2. 默认文件组织

默认训练输入位于 `/data/liuhongda/openpi_data`：

```text
openpi_data/
├── openpi-assets/checkpoints/
│   ├── pi05_base_pytorch_fp32/
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   └── assets/paligemma_tokenizer.model
│   └── pi05_libero/
│       └── assets/physical-intelligence/libero/norm_stats.json
├── lerobot/physical-intelligence/libero/
│   ├── data/chunk-*/episode_*.parquet
│   └── meta/
│       ├── info.json
│       ├── episodes.jsonl
│       └── tasks.jsonl
└── python_deps/openpi_official_pytorch_runtime/
```

各项用途如下：

- `pi05_base_pytorch_fp32`：新 fine-tuning 的 model-only 初始权重，必须是 FP32。
- `physical-intelligence/libero`：官方 LIBERO-40 LeRobot 训练集。
- `norm_stats.json`：官方 LIBERO state/action q01、q99 归一化统计。
- `openpi_official_pytorch_runtime`：包含 OpenPI replacement 的 Transformers 4.53.2
  私有 overlay。

训练不读取 `pi05_libero_pytorch_fp32` specialist 权重，也不需要 MuJoCo、LIBERO
仿真或 ROS。仿真只在训练完成后的 rollout 评测阶段使用。

默认数据契约为：

| 项目 | 值 |
| --- | ---: |
| episodes | 1693 |
| frames | 273465 |
| language tasks | 40 |
| FPS | 10 |
| state dimension | 8 |
| action dimension | 7（模型内部补到 32） |
| action horizon | 10 |

### 2.1 下载 LeRobot 数据

如果本地还没有数据集，可以从 Hugging Face 下载：

```bash
hf download physical-intelligence/libero \
  --repo-type dataset \
  --local-dir /data/liuhongda/openpi_data/lerobot/physical-intelligence/libero
```

不要将 HDF5 LIBERO demonstrations 直接填到 `OPENPI_LEROBOT_ROOT`。当前 loader
读取的是上述 LeRobot v2.0 parquet 结构。

### 2.2 准备 LIBERO norm stats

默认从官方 `pi05_libero` checkpoint 读取已经用于该 recipe 的 quantile stats：

```text
/data/liuhongda/openpi_data/openpi-assets/checkpoints/pi05_libero/
└── assets/physical-intelligence/libero/norm_stats.json
```

如果缺少该文件，可使用 OpenPI 下载官方 checkpoint：

```bash
cd /data/liuhongda/openpi
export OPENPI_DATA_HOME=/data/liuhongda/openpi_data

uv run --no-sync python -c \
  'from openpi.shared import download; print(download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero"))'
```

如果其他 checkpoint 目录中已有同一份 stats，也可以通过
`OPENPI_NORM_STATS_PATH=/path/to/norm_stats.json` 指定。不要用 LeRobot
`meta/stats.json` 中的全局 mean/std 代替这里的 state/action q01、q99。

### 2.3 准备 FP32 base 权重

如果已经有以下目录，可以跳过本节：

```text
/data/liuhongda/openpi_data/openpi-assets/checkpoints/pi05_base_pytorch_fp32
```

从 JAX 权重重新准备时，先用 OpenPI 下载官方 base checkpoint 和 tokenizer：

```bash
cd /data/liuhongda/openpi
export OPENPI_DATA_HOME=/data/liuhongda/openpi_data

uv run --no-sync python -c \
  'from openpi.shared import download; print(download.maybe_download("gs://openpi-assets/checkpoints/pi05_base"))'

uv run --no-sync python -c \
  'from openpi.shared import download; print(download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}))'
```

先准备 Transformers overlay，再调用 OpenPI 的转换器输出 FP32 参数：

```bash
cd /data/liuhongda/lightx2v_openpi
bash scripts/openpi/1_setup_pytorch_runtime.sh prepare --component transformers

cd /data/liuhongda/openpi
PYTHONPATH=/data/liuhongda/openpi_data/python_deps/openpi_official_pytorch_runtime:/data/liuhongda/openpi/src \
.venv/bin/python examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir /data/liuhongda/openpi_data/openpi-assets/checkpoints/pi05_base \
  --config-name pi05_libero \
  --output-path /data/liuhongda/openpi_data/openpi-assets/checkpoints/pi05_base_pytorch_fp32 \
  --precision float32

mkdir -p /data/liuhongda/openpi_data/openpi-assets/checkpoints/pi05_base_pytorch_fp32/assets
cp -a /data/liuhongda/openpi_data/openpi-assets/checkpoints/pi05_base/assets/. \
  /data/liuhongda/openpi_data/openpi-assets/checkpoints/pi05_base_pytorch_fp32/assets/
cp /data/liuhongda/openpi_data/big_vision/paligemma_tokenizer.model \
  /data/liuhongda/openpi_data/openpi-assets/checkpoints/pi05_base_pytorch_fp32/assets/paligemma_tokenizer.model
```

输出目录应事先不存在或为空。训练前检查会验证 `config.json` 中的精度、812 个
safetensors tensor 的 dtype，以及 tokenizer 是否完整。

`scripts/openpi/2_convert_pi05_libero_to_pytorch.sh` 默认转换的是 fine-tuned
`pi05_libero` specialist，主要供推理复现使用，不要把它的默认输出误当成训练 base。

## 3. 准备和检查运行环境

训练只需要准备 Transformers overlay，不会修改 Python、PyTorch、CUDA 或 MuJoCo：

```bash
cd /data/liuhongda/lightx2v_openpi
bash scripts/openpi/1_setup_pytorch_runtime.sh prepare --component transformers
```

可以单独执行训练前检查：

```bash
python scripts/openpi/support/runtime.py train-check
```

没有可见 GPU、只想检查文件时使用：

```bash
python scripts/openpi/support/runtime.py train-check --no-cuda
```

两个训练入口在启动 DDP 前都会自动执行 `train-check`，因此正式启动时不需要重复
手工检查。检查范围包括：

- Transformers 4.53.2 及 OpenPI replacement 文件；
- FP32 base checkpoint、config 和 tokenizer；
- LIBERO-40 数据规模及 parquet 文件；
- state/action q01、q99 norm stats；
- PyTorch、LeRobot、Pillow、SentencePiece、Augmax 和 CUDA。

## 4. 开始新的 EMA fine-tuning

默认使用 4、5、6、7 号卡：

```bash
cd /data/liuhongda/lightx2v_openpi
bash lightx2v_train/scripts/openpi/run_pi05_finetune_ema.sh
```

使用 0、1、2、3 号卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash lightx2v_train/scripts/openpi/run_pi05_finetune_ema.sh
```

默认输出到：

```text
/data/liuhongda/lightx2v_openpi/output_train/openpi/pi05_libero
```

启动新的 fine-tuning 时，输出目录不能包含已有 `checkpoint-*`。如果目录里已有
训练 checkpoint，应使用 resume 脚本或指定新的输出目录：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OPENPI_TRAIN_OUTPUT=/data/liuhongda/lightx2v_openpi/output_train/openpi/pi05_libero_run2 \
bash lightx2v_train/scripts/openpi/run_pi05_finetune_ema.sh
```

## 5. 官方对齐的默认训练参数

配置文件为
[`pi05_libero.yaml`](../../../configs/train/openpi/pi05_libero.yaml)。默认值如下：

| 参数 | 值 |
| --- | ---: |
| global batch size | 256 |
| optimizer updates | 30000 |
| warmup steps | 10000 |
| peak/end learning rate | 5e-5 / 5e-5 |
| AdamW betas | 0.9 / 0.95 |
| AdamW epsilon | 1e-8 |
| weight decay | 1e-10 |
| global gradient clip | 1.0 |
| EMA decay | 0.999 |
| seed | 42 |

参数、梯度、AdamW state、loss 和 EMA master 保持 FP32；Gemma/SigLIP 的主要矩阵
计算使用 BF16。不开 GradScaler，TF32 关闭。默认四卡、无梯度累计时：

```text
per-GPU batch 64 × 4 GPUs × accumulation 1 = global batch 256
```

若显存不足，可在保持 global batch 256 的情况下增加梯度累计。例如四卡累计 8 次：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OPENPI_GRADIENT_ACCUMULATION_ITERS=8 \
bash lightx2v_train/scripts/openpi/run_pi05_finetune_ema.sh
```

对应每卡 micro batch 为 8。梯度累计会改变浮点求和顺序；追求与官方设置尽量一致
时优先使用 accumulation 1。

前 10000 step 的 learning rate 会从 0 线性增加到 `5e-5`，之后保持不变。因此训练
早期看到 LR 持续变大是正常 warmup，不是异常发散。

## 6. 监控训练

默认日志路径：

```bash
tail -f /data/liuhongda/lightx2v_openpi/output_train/openpi/pi05_libero/train.log
```

正常日志包括：

```text
[openpi:numerics] verified fp32 parameters, gradients, AdamW state, loss, and EMA masters
[openpi:train] step=... loss=... grad_norm=... lr=...
[openpi:checkpoint] saved complete checkpoint ...
```

判断训练是否正常时重点看：

- `loss`、`grad_norm` 和 `lr` 都是有限值；
- step 持续增加；
- step 1 完成 FP32 数值链检查；
- 每 1000 step 能成功生成完整 checkpoint；
- 不同 batch 的 loss 允许上下波动，不要求单调下降。

LeRobot 可能提示数据仍是 v2.0 global stats 格式。当前 loader 对该格式兼容，而且
它正是本接入验证过的数据格式，不要在正式复现过程中临时转换数据版本。

## 7. Checkpoint 保存逻辑

默认每 1000 step 保存一次，5000 的整数倍长期保留，其他 checkpoint 只保留最新
一个。完整 checkpoint 结构如下：

```text
checkpoint-000030000/
├── model.safetensors           # online 模型参数
├── ema/
│   ├── model.safetensors       # EMA 参数，用于推理和评测
│   ├── config.json
│   └── assets/
├── training_state.pt           # optimizer、scheduler、step、RNG、数据位置
├── manifest.json               # 训练契约和 provenance
├── _SUCCESS                    # 完整写入标记
├── config.yaml                 # 本次训练配置快照
├── config.json                 # 推理模型配置
└── assets/                     # tokenizer 和 norm stats
```

当前 3.6B FP32 完整 checkpoint 约为 52 GB；一次写入前会要求约 60 GB 可用空间。
使用默认保留策略跑满 30k 后，整个输出目录约为 359 GB。checkpoint 使用 staging
目录原子写入，保存期间训练会等待磁盘 I/O。

用途必须区分：

- 继续训练：传 `checkpoint-XXXXXXXXX` 根目录。
- 推理和成功率评测：使用 `checkpoint-XXXXXXXXX/ema`。
- 不要把 `ema` 子目录传给 resume 脚本，它没有 optimizer 等训练状态。

## 8. 恢复训练

从 5000 step 的完整 checkpoint 恢复并继续到默认目标 30000：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash lightx2v_train/scripts/openpi/run_pi05_resume_ema.sh \
  /data/liuhongda/lightx2v_openpi/output_train/openpi/pi05_libero/checkpoint-000005000
```

从 30000 step 继续到 35000：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OPENPI_MAX_TRAIN_ITERS=35000 \
bash lightx2v_train/scripts/openpi/run_pi05_resume_ema.sh \
  /data/liuhongda/lightx2v_openpi/output_train/openpi/pi05_libero/checkpoint-000030000
```

`OPENPI_MAX_TRAIN_ITERS` 表示目标总步数，不是额外训练步数。如果从 30000 恢复且
仍使用默认值 30000，程序会成功恢复后立即结束，不会更新参数。

严格恢复要求以下内容与保存时一致：

- DDP world size；
- global batch 和 gradient accumulation；
- 模型结构与数值精度；
- optimizer、scheduler、EMA 参数；
- 数据集、tokenizer、norm stats、shuffle 和 seed 契约。

可以更换物理 GPU 编号，但 GPU 数量必须一致。建议始终从输出目录中最新的完整
checkpoint 恢复；如果要从旧 checkpoint 创建分支训练，应同时指定新的
`OPENPI_TRAIN_OUTPUT`，避免后续 step 与已有 checkpoint 重名。

## 9. 评测训练结果

本 recipe 没有离线 validation split。训练过程中观察 loss、gradient norm 和 LR；
最终模型质量由 LIBERO rollout 成功率衡量。

使用 30k EMA 权重在 4 张卡上评测 4 个 suite、每任务 50 trials，共 2000 episodes：

```bash
cd /data/liuhongda/lightx2v_openpi

CUDA_VISIBLE_DEVICES=4,5,6,7 \
OPENPI_MODEL_PATH=/data/liuhongda/lightx2v_openpi/output_train/openpi/pi05_libero/checkpoint-000030000/ema \
OPENPI_PARALLEL_OUTPUT_ROOT=/data/liuhongda/lightx2v_openpi/save_results/pi05_libero_trained_ema \
bash scripts/openpi/run_libero_evaluate_parallel_i2va.sh
```

结果汇总位于：

```text
save_results/pi05_libero_trained_ema/parallel_summary.json
```

官方公开结果为：

| Suite | Success rate |
| --- | ---: |
| LIBERO-Spatial | 98.8% |
| LIBERO-Object | 98.2% |
| LIBERO-Goal | 98.0% |
| LIBERO-10 | 92.4% |
| Average | 96.85% |

## 10. 常用环境变量

日常通常只需要修改 GPU、输出路径或恢复目标步数：

| 环境变量 | 作用 | 默认值 |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | 参与 DDP 的 GPU 列表 | `4,5,6,7` |
| `OPENPI_TRAIN_OUTPUT` | 日志和 checkpoint 根目录 | `output_train/openpi/pi05_libero` |
| `OPENPI_MAX_TRAIN_ITERS` | 目标 optimizer 总步数 | `30000` |
| `OPENPI_GRADIENT_ACCUMULATION_ITERS` | 梯度累计次数 | `1` |
| `OPENPI_DATA_WORKERS` | 每个 rank 的 loader workers | `2` |

更换数据或资源路径时才需要：

| 环境变量 | 作用 | 默认值 |
| --- | --- | --- |
| `OPENPI_DATA_ROOT` | OpenPI 数据根目录 | `/data/liuhongda/openpi_data` |
| `OPENPI_INITIAL_CHECKPOINT` | FP32 π0.5 base 权重 | `.../pi05_base_pytorch_fp32` |
| `OPENPI_LEROBOT_ROOT` | LIBERO-40 LeRobot 数据 | `.../lerobot/physical-intelligence/libero` |
| `OPENPI_NORM_STATS_PATH` | LIBERO quantile stats | `.../pi05_libero/assets/.../norm_stats.json` |
| `OPENPI_TRANSFORMERS_RUNTIME_PATH` | Transformers overlay | `.../openpi_official_pytorch_runtime` |
| `OPENPI_GLOBAL_BATCH_SIZE` | optimizer global batch | `256` |
| `OPENPI_NPROC_PER_NODE` | DDP 进程数 | 从可见 GPU 数量推导 |
| `OPENPI_TRAIN_SEED` | 训练、shuffle 和采样 seed | `42` |
| `OPENPI_SAVE_EVERY_ITERS` | checkpoint 保存间隔 | `1000` |
| `OPENPI_KEEP_PERIOD` | 永久 checkpoint 周期 | `5000` |
| `OPENPI_SAVE_TOTAL_LIMIT` | 非周期 checkpoint 保留数 | `1` |
| `OPENPI_LOG_EVERY_ITERS` | 训练日志间隔 | `10` |

一般不需要手工设置 `OPENPI_NPROC_PER_NODE`。如果设置了，它应与实际可见 GPU 数量
一致；global batch 必须能被 `NPROC × gradient accumulation` 整除。

## 11. 常见问题

### 为什么启动后 learning rate 一直增加？

前 10000 step 是官方设置的线性 warmup，LR 会逐步升到 `5e-5`，之后保持不变。

### 为什么保存 checkpoint 很慢？

完整 checkpoint 包含 online FP32、EMA FP32 和 AdamW state，单个约 52 GB。保存时
需要把 staging 目录完整写入磁盘后再原子发布。

### 为什么官方 base 不能传给 resume 脚本？

官方 base 只有模型参数，没有 `training_state.pt`、EMA、optimizer、scheduler、
逐 rank RNG 和数据位置。它只能传给新 fine-tuning 入口。

### 为什么 `/ema` 不能续训？

`ema` 是面向推理的模型 artifact，不是完整训练 checkpoint。续训必须传它的父目录。

### 为什么换成两张卡后 resume 失败？

完整 checkpoint 保存了一份每 rank RNG 状态，并记录了 `world_size=4`。严格复现时
必须继续使用四个 DDP rank；GPU 编号可以改变。

### 如何确认 checkpoint 完整？

至少应存在：

```text
model.safetensors
ema/model.safetensors
training_state.pt
manifest.json
_SUCCESS
```

resume 脚本和 trainer 都会再次验证这些文件及训练契约。

算法、数据处理、数值精度和验收结果都记录在本文中。

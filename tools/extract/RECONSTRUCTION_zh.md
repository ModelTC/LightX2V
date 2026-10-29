# Qwen-Image 2.1 LoRA 重建

[English](RECONSTRUCTION.md)

从 base 和全参数 target 模型提取 LoRA，使 **base + LoRA 尽量接近 target**。重建时冻结 base，仅利用 target 的采样轨迹优化 LoRA 参数。示例配置选择 232 组 Linear，使用 rank 64、alpha 8，不导出归一化权重差分。

## 配置选择

| 配置 | 时间条件重建 | 输入混合 |
|---|---|---|
| [qdiff](configs/qwen_image_21_reconstruct_qdiff.json) | 依次重建时间通路的四个 Linear | 关闭 |
| [tfmq](configs/qwen_image_21_reconstruct_tfmq.json) | 联合匹配 modulation 和最终 scale 输出 | 关闭 |
| [tfmq_qdrop](configs/qwen_image_21_reconstruct_tfmq_qdrop.json) | 同 `tfmq` | 50% student / 50% teacher |

这些流程借鉴 [Q-Diffusion](https://github.com/Xiuyu-Li/q-diffusion)、[TFMQ-DM](https://github.com/ModelTC/TFMQ-DM) 和 [QDrop](https://github.com/wimh966/QDrop) 的思想，用于 LoRA 重建，不做量化。重建顺序均为：

```text
时间条件模块 → 输入投影 → block_00 … block_31 → 输出投影
```

时间通路重建后冻结。Teacher 和 student 分别维护自己的特征缓存，逐块向后传递，梯度不跨越重建单元。

## 启动

在仓库根目录执行命令。Python 环境需要包含 `diffusers.models.transformers.transformer_qwenimage21`；采集轨迹还需要 LightX2V 推理环境。先修改所选配置中的模型和数据路径，相对路径均相对于启动目录解析。

**复用已有校准轨迹，在 8 卡上重建：**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 tools/extract/lora_reconstructor.py \
  --config tools/extract/configs/qwen_image_21_reconstruct_tfmq_qdrop.json
```

需要采集时，在上面的命令后追加相应选项：

| 选项 | 行为 |
|---|---|
| `--collect-calibration` | 仅采集 target 轨迹 |
| `--collect-and-reconstruct` | 重新采集一份数据，完成后开始重建 |
| `--collect-calibration --dry-run` | 仅检查样本选择和参考图路径，不执行推理 |

单卡时，将 `torchrun --standalone --nproc_per_node=8` 换成 `python`，并设置 `CUDA_VISIBLE_DEVICES=0`。切换算法只需更换配置文件路径。

采集时每卡独立进行 BF16 推理，使用 block CPU offload；`compute_dtype` 仅控制重建精度。重放时间步先按轨迹记录的采集精度转换并除以 1000，再转为重建精度；未记录采集精度的旧轨迹按 BF16 处理。重建时通过 DDP 同步当前单元的 LoRA 梯度，依次加载 teacher/student 单元，无需 FSDP。各进程需要共享模型、数据和输出目录；单卡仍需容纳当前单元及其激活。

## 校准数据与参数

示例配置采集 **512 条 T2I + 256 条单参考图 + 256 条多参考图请求**。每条请求取 4 个去噪时间步，共 **4096 条重建样本**。每个 `.pt` 轨迹保存带噪 latent、条件和时间步；各块的监督目标在重建时另外计算。

| 字段 | 含义 / 示例配置值 |
|---|---|
| `source_model`、`target_model` | Base 和完整 target 权重，也支持已导出的学生模型权重 |
| `output`、`layers`、`rank`、`alpha` | LoRA 输出位置、选择的层、rank 64、alpha 8 |
| `calibration` | 已有轨迹文件或目录，也是采集结果的保存位置 |
| `calibration_manifest` | 请求清单，默认 `lightx2v_train/data/opengpt4o_qwen_image21/train.jsonl` |
| `calibration_counts` | 各任务请求数，尚未乘以时间步数 |
| `calibration_inference_config` | 原生推理配置，示例使用 4 步 |
| `iterations`、`temporal_iterations` | 普通单元 / 时间单元各优化 1000 iter |
| `batch_size` | 全局 batch 为 8，8 卡时每卡处理 1 条样本 |
| `learning_rate`、`gradient_clip` | `5e-5`、`1.0` |
| `compute_dtype` | BF16 计算；LoRA 主参数、优化器状态和 MSE 运算保留 FP32 |
| `dtype` | 导出的 LoRA 精度，默认 FP32 |
| `log_every` | 每 25 iter 更新当前训练 batch 的 MSE 日志 |
| `seed` | 请求选择和重建采样的随机种子 |
| `work_dir`、`keep_cache` | 重建临时目录；默认清理中间缓存 |

少于 8 卡时，梯度累积次数为 `batch_size / world_size`，必须整除。训练从缓存中随机抽样，按固定 iter 优化，不按完整 epoch 遍历。时间条件重建使用采集到的时间步，加上条件分支的 `t=0`。

采集要求 `target_model` 是完整推理 pipeline。如果 target 只有导出的 transformer 权重，需要另行采集并提供轨迹。采集的推理时间步应匹配 target 模型。`calibration_selection.json` 记录选中的文件及完成状态；采集未完成时，重建会报错。

采集工具会自动读取关闭动态 shifting 的 target 模型中 `model_index.json` 的 `sample_sigmas`，作为固定 `sigmas` 传给公共 scheduler，并关闭再次 shift；显式配置的 `sigmas` 优先。每个采集输出目录会保留实际使用的 `inference_config.json`，原配置文件不变。

## QDrop 输入混合

两种算法都支持以下配置，只有 `tfmq_qdrop` 示例默认启用：

```json
{
  "qdrop_input_mix": true,
  "qdrop_student_prob": 0.5
}
```

训练 Transformer block 和输出层时，每个特征元素以 `qdrop_student_prob` 的概率选择 student 前缀输出，否则选择对应的 teacher 前缀输出。每次微步重新生成随机掩码。设为 `0` 表示纯 teacher 特征，设为 `1` 表示纯 student 特征。

监督目标始终是 teacher 输出；时间条件、输入层和独立时间单元的重建不变。**全量评估及传给下一块的缓存始终使用纯 student 输入。** 启用混合需要额外缓存 teacher 输入，会增加临时磁盘占用。

## MSE、输出和缓存清理

每个单元在训练前后，都对**该单元的全部校准样本**评估一次。这里使用的是训练校准集，没有单独划分验证集。训练当前单元期间，teacher 目标和 student 前缀输入固定。若最终 MSE 变差，则恢复初始 LoRA 参数。`MSE 前 → 后` 表示最终保留参数的效果；进度条上的 MSE 是当前训练 batch 的误差。生成质量需要另用留出请求评估。

`tfmq_qdrop` 的默认保存位置：

```text
outputs/qwen_image_21_calibration/                         # 保留的原始轨迹
    calibration_selection.json
    collection_<id>/rank_<rank>/*.pt                       # 多卡采集结果
outputs/qwen_image_21_tfmq_qdrop_r64_a8.svd.safetensors     # 保留的 SVD 初始化 LoRA
outputs/qwen_image_21_tfmq_qdrop_r64_a8.safetensors         # 最终 LoRA
outputs/reconstruction_tfmq_qdrop/<id>/
    config.json
    calibration.json
    metrics.jsonl
    rank_<rank>/                                          # 临时特征缓存
```

使用 `initialization: "svd"` 时，训练前会保存初始化 LoRA，位置可通过 `svd_output` 覆盖。`keep_cache: false` 会随重建推进删除上一阶段的特征缓存，成功结束后清理剩余缓存及各卡的初始化分片。**原始轨迹、完整 SVD LoRA、最终 LoRA 和运行指标都会保留。**

中断后需要重新开始重建，目前不支持恢复优化器或已完成阶段。可设置 `initialization: "lora"`，并通过 `initial_lora` 加载层名、rank 和 alpha 匹配的导出 LoRA，复用初始化。不同实验应使用独立输出路径。

# VAE 分组剪枝：每组保留一个残差主分支

这三组实验使用独立的新配置、输出目录和启动脚本，旧全局剪枝配置与结果保持不变。搜索阶段固定每组保留数量，但通过门控学习具体保留组内哪一层；不是预先指定每组第一层或最后一层。

| 组件 | 分组 | 每组保留 | 总保留 |
| --- | --- | --- | --- |
| H3 encoder | 6 个 stage，每组 2 个残差块 | 1 | 6 / 12 |
| Wan encoder | 4 个 stage + middle，每组 2 个残差块 | 1 | 5 / 10 |
| Wan decoder | middle 组含 2 个块，4 个上采样 stage 各含 3 个块 | 1 | 5 / 14 |

搜索配置使用 `grouping: stage`、`keep_per_group: 1`，不再使用旧全局 `keep_residuals` 预算。只剪残差主分支，必要的形状变换旁路、上/下采样和注意力仍保留。搜索保留原有 rank16/alpha32 临时适配器；导出时按门控 EMA 选择结构，恢复从该结构对应的原始教师权重开始，不合并搜索适配器。

分组顺序沿网络前向：H3 encoder 为六个编码阶段，Wan encoder 为四个编码阶段后接 middle，Wan decoder 为 middle 后接四个解码阶段。middle 单独成组，即使它与相邻阶段分辨率相同也不合并。

`keep_per_group` 也支持按上述顺序填写整数列表，例如 Wan decoder 的 `[1, 1, 1, 2, 1]` 共保留六个分支。每组预算不能超过该组分支数。未指定 `grouping` 的旧配置仍使用全局搜索；变更分组或预算后应使用新输出目录重新搜索，不能续训旧结构。

数据、裁剪、损失、学习率和调度沿用各自原配置：搜索 1000 步、恢复 3000 步，梯度累积 4 步；两阶段每 100 步保存 checkpoint 并生成重建预览。分组约束本身不保证恢复质量，仍需训练及独立评估。

## 启动

本次仅准备配置与脚本，没有启动训练。新脚本默认 `GPU_LIST=0,1`，不会沿用旧脚本的 6、7 号卡默认值或继承 `CUDA_VISIBLE_DEVICES`；6、7 号卡当前已有任务。启动前仍需确认所选卡空闲，不要在相同 GPU 上并发启动下面的实验。每个组件先完成 search，再运行 recover。

```bash
cd /data/nvme6/gushiqiao/codes/latest/vae/LightX2V/lightx2v_train

GPU_LIST=0,1 bash scripts/run_minimax_h3_encoder_prune_grouped.sh search
GPU_LIST=0,1 bash scripts/run_minimax_h3_encoder_prune_grouped.sh recover

GPU_LIST=0,1 bash scripts/run_wan21_vae_prune_grouped.sh encoder search
GPU_LIST=0,1 bash scripts/run_wan21_vae_prune_grouped.sh encoder recover

GPU_LIST=0,1 bash scripts/run_wan21_vae_prune_grouped.sh decoder search
GPU_LIST=0,1 bash scripts/run_wan21_vae_prune_grouped.sh decoder recover
```

两个启动脚本的 DDP 进程数均按 `GPU_LIST` 长度确定，默认两卡。脚本使用现有 Python 环境，不安装依赖、不下载模型。

## 配置与输出隔离

| 组件 | 搜索配置 | 恢复配置 |
| --- | --- | --- |
| H3 encoder | [grouped keep6 search](../configs/train/vae/minimax_h3_encoder_prune_search_grouped_keep6_2gpu_ddp.yaml) | [grouped keep6 recover](../configs/train/vae/minimax_h3_encoder_prune_recover_grouped_keep6_2gpu_ddp.yaml) |
| Wan encoder | [grouped keep5 search](../configs/train/vae/wan21_encoder_prune_search_grouped_keep5_ddp.yaml) | [grouped keep5 recover](../configs/train/vae/wan21_encoder_prune_recover_grouped_keep5_ddp.yaml) |
| Wan decoder | [grouped keep5 search](../configs/train/vae/wan21_decoder_prune_search_grouped_keep5_ddp.yaml) | [grouped keep5 recover](../configs/train/vae/wan21_decoder_prune_recover_grouped_keep5_ddp.yaml) |

每个 YAML 默认输出到 `output_train/<该 YAML 不含扩展名的文件名>`。自动续训只查各自新目录；recover 默认读取对应新 search 目录的 `export/kept_layers.json`，不回退到旧全局剪枝导出。恢复时，同目录的 `minimax_h3_pruned_encoder.safetensors`、`wan21_pruned_encoder.safetensors` 或 `wan21_pruned_decoder.safetensors` 也必须存在。

自定义路径使用新的环境变量前缀：

- H3 encoder：`H3_VAE_ENCODER_GROUPED_SEARCH_OUTPUT`、`H3_VAE_ENCODER_GROUPED_RECOVER_OUTPUT`、`H3_VAE_ENCODER_GROUPED_SELECTION`。
- Wan encoder：`WAN_VAE_ENCODER_GROUPED_SEARCH_OUTPUT`、`WAN_VAE_ENCODER_GROUPED_RECOVER_OUTPUT`、`WAN_VAE_ENCODER_GROUPED_SELECTION`。
- Wan decoder：`WAN_VAE_DECODER_GROUPED_SEARCH_OUTPUT`、`WAN_VAE_DECODER_GROUPED_RECOVER_OUTPUT`、`WAN_VAE_DECODER_GROUPED_SELECTION`。
- 解释器：`H3_VAE_GROUPED_PYTHON` / `WAN_VAE_GROUPED_PYTHON`；Wan 原始权重：`WAN_VAE_GROUPED_PATH`。

旧的非 `GROUPED` 输出、selection、Python 和 Wan 权重环境变量不会被新脚本读取或覆盖。需要全新一轮实验时，给新命名空间设置新的输出目录；不要将不同组件或 search/recover 指向同一目录，也不要将 grouped selection 指向旧全局剪枝结果。

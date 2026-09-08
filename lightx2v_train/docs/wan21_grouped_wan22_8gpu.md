# Wan22 视频数据：Wan2.1 VAE 分组剪枝八卡启动

使用新机器现有的训练环境与原版 `Wan2.1_VAE.pth`。Wan22 是数据目录名，不代表更换 VAE 架构；encoder、decoder 仍各分 5 组，每组学习保留 1 个残差主分支。

将生成的 `/data/nvme6/gushiqiao/metadata_vae.jsonl` 复制到新机器：

```text
/mnt/lm_data_afs/wuzhuguanyu/LightX2V_train/lightx2v_train/data/wan22/metadata_vae.jsonl
```

清单使用该机器 `data/wan22/videos/` 下的绝对视频路径，不依赖 prompt 文件。新配置设置 `skip_missing: false`，不通过跳过缺失视频凑出可训练清单；远端视频尚未在本机验证。脚本不自动生成或改写清单。

## 启动

权重默认使用 `/mnt/devsft_afs_1/gushiqiao/Wan2.1_VAE.pth`。下面各组件必须先完成 search，再运行 recover；不要在相同八张卡上并发启动 encoder 与 decoder。

```bash
cd /mnt/lm_data_afs/wuzhuguanyu/LightX2V_train/lightx2v_train
export WAN_VAE_GROUPED_PATH=/mnt/devsft_afs_1/gushiqiao/Wan2.1_VAE.pth
# 非默认解释器时：export WAN_VAE_GROUPED_PYTHON=/实际环境/bin/python

bash scripts/run_wan21_vae_prune_grouped_wan22_8gpu.sh encoder search
bash scripts/run_wan21_vae_prune_grouped_wan22_8gpu.sh encoder recover

bash scripts/run_wan21_vae_prune_grouped_wan22_8gpu.sh decoder search
bash scripts/run_wan21_vae_prune_grouped_wan22_8gpu.sh decoder recover
```

默认 `GPU_LIST=0,1,2,3,4,5,6,7`，DDP 进程数按列表长度确定；解释器默认 `python`。清单可通过 `WAN_VAE_METADATA` 指定，权重可通过 `WAN_VAE_GROUPED_PATH` 覆盖。脚本先检查清单和权重，recover 还要求 selection 同目录具有对应完整导出权重。

新四份配置位于 `configs/train/vae/wan21_{encoder,decoder}_prune_{search,recover}_grouped_keep5_wan22_8gpu_ddp.yaml`。输出默认位于训练根的 `output_train/<配置文件名去掉 .yaml>`，与旧实验隔离。自定义输出和结构文件使用 `WAN_VAE_ENCODER_WAN22_{SEARCH_OUTPUT,RECOVER_OUTPUT,SELECTION}` 或 `WAN_VAE_DECODER_WAN22_{SEARCH_OUTPUT,RECOVER_OUTPUT,SELECTION}`；恢复不回退到旧 grouped/global 导出。重复启动会续训该新目录，需要新实验时另设输出目录。

需要同步到新机器的新增文件：

- [Decoder 搜索配置](../configs/train/vae/wan21_decoder_prune_search_grouped_keep5_wan22_8gpu_ddp.yaml)、[Decoder 恢复配置](../configs/train/vae/wan21_decoder_prune_recover_grouped_keep5_wan22_8gpu_ddp.yaml)。
- [Encoder 搜索配置](../configs/train/vae/wan21_encoder_prune_search_grouped_keep5_wan22_8gpu_ddp.yaml)、[Encoder 恢复配置](../configs/train/vae/wan21_encoder_prune_recover_grouped_keep5_wan22_8gpu_ddp.yaml)。
- [八卡启动脚本](../scripts/run_wan21_vae_prune_grouped_wan22_8gpu.sh) 和上述 `metadata_vae.jsonl`。

前提是新机器已同步上一轮 Wan 分组剪枝与蒸馏代码。如果只有旧训练代码，仅复制这些 YAML 和 Bash 不够。

如需在新机器从原清单重新生成，可同步 [清单转换脚本](../scripts/prepare_wan_vae_metadata.py)，使用未存在的新输出文件，并加 `--check-files` 检查视频是否齐全：

```bash
python scripts/prepare_wan_vae_metadata.py \
  --input /实际位置/metadata.jsonl \
  --output data/wan22/metadata_vae_checked.jsonl \
  --video-root /mnt/lm_data_afs/wuzhuguanyu/LightX2V_train/lightx2v_train/data/wan22/videos \
  --check-files
export WAN_VAE_METADATA="$PWD/data/wan22/metadata_vae_checked.jsonl"
```

## 当前训练设置：81 帧、480×832

四份 Wan22 八卡配置的 train/val 均设为 `height: 480`、`width: 832`、`num_frames: 81`、`geometry_from_metadata: false`。已检查的 `000414.mp4` 实际为 832×480、81 帧、16fps；该尺寸输入不会在读取阶段缩放或裁掉左右画面。清单中的 2048×2048、201 帧不是可靠的实测值，不据此决定训练尺寸，也没有将整份清单的几何字段统一改写。

`frame_rate: 16`、`fix_frame_rate: false` 保留视频实际帧序列；预览按 16fps 保存。81 符合 Wan 的 `4n+1` 协议，完整 81 帧对应 21 个潜变量时间位置。`min_source_frames: 81` 要求读取到至少 81 个源帧，不会把短视频重复补帧凑到 81；其余视频是否满足仍需在新机器确认。

| 阶段 | 每次读取 | 实际重建监督 | LPIPS 抽帧上限 | GAN 抽帧上限 |
| --- | --- | --- | --- | --- |
| Search，1000 iter | 81×480×832 | 随机 33×256×256 | 32 | 关闭 |
| Recover，0–599 iter | 81×480×832 | 随机 33×256×256 | 64，实际最多 33 | 关闭 |
| Recover，600–2999 iter | 81×480×832 | 81×384×384 空间裁剪 | 64 | 32，保留原渐入调度 |
| 重建预览 | 81×480×832 | 完整读取区域重建 | 不适用 | 不适用 |

恢复后段由 65 帧监督改为 81 帧；读取 81 帧不意味着搜索阶段也每步监督全部 81 帧。分组预算、rank16/alpha32、loss 种类与权重、学习率、搜索 1000 步/恢复 3000 步均不变，每 100 步保存和预览。

每卡 batch=1、梯度累积 4 步，八卡有效 batch=32。读取内存、恢复后段激活和全幅预览开销会增加；未实测新机器峰值显存。train/val 使用同一清单，预览不代表独立验证集质量。若已经用旧尺寸/帧数跑过该输出目录，建议另设新输出目录区分实验，不删除旧 checkpoint。

这次仅修改上述四份 Wan22 八卡配置与说明；旧数据集的 global/grouped 配置保留，没有启动训练。

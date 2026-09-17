# MiniMax-H3 CPU block offload 使用指南

本目录提供 DiT、文本主干和视频 VAE decoder 的 CPU block offload，以及 host／NUMA CPU 权重共享。固定卡数入口与按可见 GPU 数启动的 `_auto.sh` 入口均在下文说明。所有命令为单机推理，使用当前 Python 环境。

## 准备环境、权重和 AdaLN cache

激活已安装项目依赖的 Python 环境。默认使用原始权重、BF16、SageAttention2、SGL 和 Triton 算子。

```bash
cd /path/to/lightx2v_offload_opt
export MINIMAX_H3_MODEL_PATH="$PWD/models/MiniMax-H3"
```

模型根目录需要 `text_encoder/`、`tokenizer/`、`processor/`、`vae/`、`audio_vae/`；基础任务使用 `transformer/`，Ref2AV 使用 `transformer_ref/`。

**CPU offload 必须启用 `use_adaln_cache=true`，并提前准备匹配的 AdaLN cache。** 默认配置为 29 步，video/audio flow shift 为 12／3；没有匹配缓存时执行：

```bash
# T2AV／I2AV／L2AV／FL2AV 共用基础任务缓存
CUDA_VISIBLE_DEVICES=0 MINIMAX_H3_CACHE_TASK=fl2av \
MINIMAX_H3_CONFIG="$PWD/configs/minimax_h3/offload/minimax_h3_t2av_block_shared_offload_host_auto.json" \
bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh

# Ref2AV 需要单独的缓存
CUDA_VISIBLE_DEVICES=0 MINIMAX_H3_CACHE_TASK=ref2av \
MINIMAX_H3_CONFIG="$PWD/configs/minimax_h3/offload/minimax_h3_t2av_block_shared_offload_host_auto.json" \
bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh
```

缓存生成工具只用一张 GPU，模板中的 SP8 参数不会让离线工具启动八个进程。默认缓存根目录为 `~/.cache/lightx2v/adaln`，由 JSON 的 `adaln_cache_dir` 指定。已有同一模型、步数和 flow shift 的缓存时跳过生成，工具不会覆盖已有目标目录。改变权重或调度参数后需重新生成匹配缓存，必要时换一个缓存根目录；只改变卡数或共享范围不需要重建。

## 固定卡数入口

| 场景 | T2AV 启动脚本 |
| --- | --- |
| 单卡，私有 CPU 权重 | `run_minimax_h3_t2av_block_offload.sh` |
| 8 卡，每进程私有 CPU 权重 | `run_minimax_h3_t2av_block_offload_sp8.sh` |
| 8 卡，host 共享 | `run_minimax_h3_t2av_block_shared_offload_host_sp8.sh` |
| 8 卡，NUMA 共享 | `run_minimax_h3_t2av_block_shared_offload_numa_sp8.sh` |

```bash
# 单卡，私有 CPU 权重
CUDA_VISIBLE_DEVICES=0 \
bash scripts/minimax_h3/offload/run_minimax_h3_t2av_block_offload.sh

# 固定 8 卡 Ref2AV，host 共享
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/minimax_h3/offload/run_minimax_h3_ref2av_block_shared_offload_host_sp8.sh
```

五种任务均有 `run_minimax_h3_<task>_block_shared_offload_host_sp8.sh` 和 `run_minimax_h3_<task>_block_shared_offload_numa_sp8.sh`。`_sp8.sh` 固定启动 8 个进程；仅修改可见 GPU、`NPROC_PER_NODE` 或 `MINIMAX_H3_CONFIG` 不会改变进程数。需要按可见卡数启动时使用下一节的 `_auto.sh`。

## 按可见 GPU 数启动（`_auto.sh`）

每种共享方式都有独立入口。设置 `CUDA_VISIBLE_DEVICES` 后，脚本读取对应 `_auto.json` 模板，按卡数生成临时运行配置，并使用当前环境的 `python -m torch.distributed.run` 启动。

```bash
# 4 卡 Ref2AV，host 共享
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/minimax_h3/offload/run_minimax_h3_ref2av_block_shared_offload_host_auto.sh

# 2 卡 Ref2AV，NUMA 共享，可指定不连续的显卡
CUDA_VISIBLE_DEVICES=2,5 \
bash scripts/minimax_h3/offload/run_minimax_h3_ref2av_block_shared_offload_numa_auto.sh

# 单卡 T2AV
CUDA_VISIBLE_DEVICES=0 \
bash scripts/minimax_h3/offload/run_minimax_h3_t2av_block_shared_offload_host_auto.sh
```

支持 **1、2、4、7、8、14、28、56 卡**，全部使用纯 SP。模型有 56 个注意力头，Ulysses SP 大小须整除 56。3、6、16 等卡数会在加载模型前报错，脚本不会自动少用显卡；16 卡机器可显式选择其中 8 或 14 张。

未设置 `CUDA_VISIBLE_DEVICES` 时使用 GPU 0。无需设置 `NPROC_PER_NODE` 或手工生成临时配置。GPU 必须实际存在，并满足工作显存要求；CPU 权重共享不共享各 GPU 的激活和工作缓冲。

## 任务与输入

每个任务的两个可变卡数入口如下，固定 8 卡入口将 `_auto.sh` 换为 `_sp8.sh`：

```text
run_minimax_h3_<task>_block_shared_offload_host_auto.sh
run_minimax_h3_<task>_block_shared_offload_numa_auto.sh
```

| `<task>` | 输入环境变量 | AdaLN 缓存组 |
| --- | --- | --- |
| `t2av` | 无参考输入 | `fl2av` |
| `i2av` | `MINIMAX_H3_IMAGE_PATH`：首帧 | `fl2av` |
| `l2av` | `MINIMAX_H3_LAST_FRAME_PATH`：尾帧 | `fl2av` |
| `fl2av` | 上述首帧和尾帧变量 | `fl2av` |
| `ref2av` | `MINIMAX_H3_IMAGE_PATH`／`MINIMAX_H3_VIDEO_PATH`，可附加 `MINIMAX_H3_AUDIO_PATH` | `ref2av` |

Ref2AV 的同类参考素材用逗号分隔，音频须搭配图片或视频；不设置参考变量时使用仓库示例图片。FL2AV 自定义输入示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MINIMAX_H3_IMAGE_PATH="$PWD/assets/inputs/imgs/flf2v_input_first_frame-fs8.png" \
MINIMAX_H3_LAST_FRAME_PATH="$PWD/assets/inputs/imgs/flf2v_input_last_frame-fs8.png" \
MINIMAX_H3_PROMPT="Create a coherent transition with natural synchronized sound." \
MINIMAX_H3_SAVE_RESULT_PATH="$PWD/save_results/minimax_fl2av.mp4" \
bash scripts/minimax_h3/offload/run_minimax_h3_fl2av_block_shared_offload_host_auto.sh
```

全部入口支持 `MINIMAX_H3_MODEL_PATH`、`MINIMAX_H3_CONFIG`、`MINIMAX_H3_SAVE_RESULT_PATH`；I2AV／L2AV／FL2AV／Ref2AV 还支持 `MINIMAX_H3_PROMPT`。T2AV 提示词直接修改脚本的 `--prompt`。H3 不使用 CFG，也不传 `--negative_prompt`。

## 配置与共享方式

`_auto.sh` 的配置位于 `configs/minimax_h3/offload/`，按 scope 使用两份模板：

- host：`minimax_h3_t2av_block_shared_offload_host_auto.json`
- NUMA：`minimax_h3_t2av_block_shared_offload_numa_auto.json`

模板保留 8 卡基准参数；`_auto.sh` 按可见卡数在临时副本中设置并行参数，再将副本传给推理。每次启动的临时文件独立，正常结束或报错退出后自动删除，模板本身不被修改。

五种任务复用这些启动配置，实际任务由脚本中的 `--task` 指定；文件名中的 `t2av` 不会覆盖 Ref2AV 等任务。默认 124 帧、544×960、seed 42；使用默认模板时，8 卡保留模板的 VAE tile 并行设置，其余卡数关闭 VAE tile 并行。若手工开启 VAE 并行，需满足 tile 数不少于进程数。

使用 `_auto.sh` 时，需要改变步数、分辨率等参数可修改对应 `_auto.json` 模板；也可用 `MINIMAX_H3_CONFIG` 指向完整配置，但其 TP×SP×CFG 必须等于可见卡数，scope 必须匹配脚本。显式指定完整配置时，脚本不调整其中的并行参数和 VAE 设置，也不会重写原文件。

host 在同机同一 IPC 域共享一份 CPU 权重；NUMA 每个参与的 GPU NUMA 域一份。共享配置同时开启 `shared_cpu_weights`、`text_encoder_shared_cpu_weights`、`video_vae_shared_cpu_weights`，分别共享 DiT、文本主干、视频 VAE decoder 的 CPU 权重。依赖 Linux SysV、CUDA pinned memory；严格 NUMA 绑定不可用时可选择 host。当前接入不支持 TP、LoRA、量化、compile 或 lazy loading。

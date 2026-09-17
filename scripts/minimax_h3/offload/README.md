# MiniMax-H3 CPU block offload 使用指南

共享 offload 使用统一入口：`run_minimax_h3_block_shared_offload.sh`，使用 `configs/minimax_h3/offload/minimax_h3_block_shared_offload.json`。默认 **TASK=t2av、host 共享、8 卡、Ulysses SP8**，通过 `TASK` 选择任务，通过环境变量切换 NUMA 或指定显卡。使用当前环境的 `python -m torch.distributed.run`，以下命令均为单机推理。

## 准备环境、权重和 AdaLN cache

激活已安装项目依赖的 Python 环境，在仓库根目录执行：

```bash
cd /path/to/lightx2v_offload_opt
export MINIMAX_H3_MODEL_PATH="$PWD/models/MiniMax-H3"
```

基础任务的模型目录需要 `transformer/`、`text_encoder/`、`tokenizer/`、`processor/`、`vae/` 和 `audio_vae/`。Ref2AV 还需要 `transformer_ref/`。默认使用原始 BF16 权重、SageAttention2、SGL 和 Triton 算子。

**H3 CPU offload 需要匹配的 AdaLN cache。** 默认 29 步，video/audio flow shift 为 12／3；T2AV 使用基础任务的 `fl2av` 缓存：

```bash
CUDA_VISIBLE_DEVICES=0 MINIMAX_H3_CACHE_TASK=fl2av \
MINIMAX_H3_CONFIG="$PWD/configs/minimax_h3/offload/minimax_h3_block_shared_offload.json" \
bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh
```

缓存工具只用一张 GPU，JSON 中的 SP8 不会让它启动八个进程。默认缓存目录为 `~/.cache/lightx2v/adaln`，由 `adaln_cache_dir` 指定。已有匹配缓存可跳过；改变权重、步数或 flow shift 后需准备新缓存，只改变卡数或共享范围不需要重建。

Ref2AV 使用独立的 AdaLN cache，首次运行前另行生成：

```bash
CUDA_VISIBLE_DEVICES=0 MINIMAX_H3_CACHE_TASK=ref2av \
MINIMAX_H3_CONFIG="$PWD/configs/minimax_h3/offload/minimax_h3_block_shared_offload.json" \
bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh
```

## 启动命令

```bash
# 默认 host + 8 卡：未设置 CUDA_VISIBLE_DEVICES 时使用 0,1,2,3,4,5,6,7
bash scripts/minimax_h3/offload/run_minimax_h3_block_shared_offload.sh

# NUMA + 8 卡
SHARED_CPU_WEIGHT_SCOPE=numa \
bash scripts/minimax_h3/offload/run_minimax_h3_block_shared_offload.sh

# host + 4 卡
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/minimax_h3/offload/run_minimax_h3_block_shared_offload.sh

# NUMA + 2 卡，可使用不连续的显卡编号
CUDA_VISIBLE_DEVICES=2,5 SHARED_CPU_WEIGHT_SCOPE=numa \
bash scripts/minimax_h3/offload/run_minimax_h3_block_shared_offload.sh

# 单卡
CUDA_VISIBLE_DEVICES=0 \
bash scripts/minimax_h3/offload/run_minimax_h3_block_shared_offload.sh
```

支持 **1、2、4、7、8、14、28、56 卡**，使用全部指定显卡。模型有 56 个注意力头，Ulysses SP 大小必须整除 56；不支持的卡数会直接报错。默认模板按可见卡数调整 SP，8 卡保留 VAE tile 并行，其余卡数关闭它。无需设置 `NPROC_PER_NODE`。

## 选择任务与输入

同一个脚本通过 `TASK` 选择任务，省略时运行 `t2av`。五个任务共用同一份启动配置。

| `TASK` | 输入环境变量 | AdaLN 缓存组 |
| --- | --- | --- |
| `t2av` | 无参考输入 | `fl2av` |
| `i2av` | `MINIMAX_H3_IMAGE_PATH`：首帧 | `fl2av` |
| `l2av` | `MINIMAX_H3_LAST_FRAME_PATH`：尾帧 | `fl2av` |
| `fl2av` | 上述首帧和尾帧变量 | `fl2av` |
| `ref2av` | `MINIMAX_H3_IMAGE_PATH`／`MINIMAX_H3_VIDEO_PATH`，可附加 `MINIMAX_H3_AUDIO_PATH` | `ref2av` |

```bash
TASK=fl2av CUDA_VISIBLE_DEVICES=0,1 SHARED_CPU_WEIGHT_SCOPE=numa \
MINIMAX_H3_IMAGE_PATH=/path/to/first.png \
MINIMAX_H3_LAST_FRAME_PATH=/path/to/last.png \
bash scripts/minimax_h3/offload/run_minimax_h3_block_shared_offload.sh

TASK=ref2av MINIMAX_H3_IMAGE_PATH=/path/to/reference.png \
bash scripts/minimax_h3/offload/run_minimax_h3_block_shared_offload.sh
```

需要图像的任务未指定输入时使用仓库示例。Ref2AV 同类参考素材可用逗号分隔，音频必须搭配图片或视频。脚本根据任务选择默认提示词和输出文件名；`MINIMAX_H3_PROMPT` 可覆盖提示词。

## 参数与配置

| 环境变量 | 默认值／用途 |
| --- | --- |
| `TASK` | 默认 `t2av`；支持上述五种任务 |
| `CUDA_VISIBLE_DEVICES` | 未设置时为 `0,1,2,3,4,5,6,7`；已设置时使用指定显卡 |
| `SHARED_CPU_WEIGHT_SCOPE` | 未设置时读取 JSON 的 `shared_cpu_weight_scope`，默认 `host`；可设 `host` 或 `numa` |
| `MINIMAX_H3_MODEL_PATH` | 仓库内 `models/MiniMax-H3` |
| `MINIMAX_H3_PROMPT` | 默认提示词见脚本，可直接覆盖 |
| `MINIMAX_H3_SAVE_RESULT_PATH` | `save_results/minimax_h3_<TASK>_block_shared_offload.mp4` |
| `SEED` | `42` |
| `MINIMAX_H3_CONFIG` | 可选完整 JSON 配置，未设置时使用上述默认配置 |

```bash
CUDA_VISIBLE_DEVICES=0,1 MINIMAX_H3_PROMPT="A fox walks through snow with natural ambient sound." \
MINIMAX_H3_SAVE_RESULT_PATH="$PWD/save_results/h3_custom.mp4" SEED=123 \
bash scripts/minimax_h3/offload/run_minimax_h3_block_shared_offload.sh
```

默认配置为 124 帧、544×960、24 fps。步数、分辨率、帧数及缓存目录修改 JSON 中的 `infer_steps`、`target_height`、`target_width`、`target_video_length` 和 `adaln_cache_dir`。

脚本从配置生成独立临时 JSON，退出时删除，不改写源文件。显式传入 `MINIMAX_H3_CONFIG` 时保留其中的并行参数和 VAE 设置，TP×SP×CFG 必须等于可见卡数；`SHARED_CPU_WEIGHT_SCOPE` 若设置则覆盖该配置的 scope，否则保留配置中的值。模型、配置和输出的相对路径以调用脚本时的目录为基准。

host 在同机同一 IPC 域共享一份 CPU 权重；NUMA 按参与 GPU 的 NUMA 域建立副本，严格绑定失败时会报错。配置分别开启 DiT、文本主干和视频 VAE 的权重共享，以及对应的 block offload。依赖 Linux SysV 和 CUDA pinned memory；GPU 激活和工作缓冲仍各自占用显存。当前共享路径不支持 TP、量化、LoRA、compile 或 lazy loading。

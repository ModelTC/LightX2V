# MiniMax-H3 CPU block offload 使用指南

[English](README.md) | 简体中文

启动脚本：`run_minimax_h3_block_shared_offload.sh`。
配置文件：`configs/minimax_h3/offload/minimax_h3_block_shared_offload.json`。
默认运行 **T2AV、host 共享、8 卡、Ulysses SP8**。

直接修改脚本中的路径、显卡和推理参数；共享范围和并行设置修改 JSON。脚本固定设置参数，不自动推算卡数，也不转发 `bash 脚本.sh` 后追加的参数。`TASK`、`SHARED_CPU_WEIGHT_SCOPE`、`MINIMAX_H3_*` 环境变量不能覆盖此推理脚本的设置。若要在不修改脚本的情况下调整命令行参数，可直接运行完整的 `python -m torch.distributed.run ... -m lightx2v.infer ...` 命令。

## 准备与启动

1. 激活已安装项目依赖的 Python 环境。
2. 修改启动脚本顶部的 `lightx2v_path` 和 `model_path`，分别填写项目和权重目录的实际路径。
3. 按下节生成匹配的 AdaLN cache；已有匹配缓存可跳过。
4. 从项目根目录运行：

```bash
cd /path/to/LightX2V
bash scripts/minimax_h3/offload/run_minimax_h3_block_shared_offload.sh
```

基础任务需要模型目录中的 `transformer/`、`text_encoder/`、`tokenizer/`、`processor/`、`vae/` 和 `audio_vae/`；专用 Ref2AV 权重还需要 `transformer_ref/`。默认使用原始 BF16 权重、SageAttention2、SGL 和 Triton 算子。

## 准备 AdaLN cache

当前 H3 共享 block offload 接入需要匹配的 AdaLN cache。以下命令在项目根目录执行，将模型路径改成与推理脚本一致的实际路径：

```bash
CUDA_VISIBLE_DEVICES=0 \
MINIMAX_H3_MODEL_PATH=/path/to/MiniMax-H3 \
MINIMAX_H3_CACHE_TASK=fl2av \
MINIMAX_H3_CONFIG="$PWD/configs/minimax_h3/offload/minimax_h3_block_shared_offload.json" \
bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh
```

缓存工具通过上述环境变量传参，使用单张 GPU。缓存应匹配 `--model-variant`：`fl2av` 权重使用 `fl2av` 缓存，`ref2av` 权重使用 `ref2av` 缓存。生成后者时将 `MINIMAX_H3_CACHE_TASK` 改为 `ref2av`。

默认 29 步，video/audio flow shift 为 12／3，缓存目录为 `~/.cache/lightx2v/adaln`。生成缓存和推理必须使用匹配的权重、`infer_steps`、flow shift 和 `adaln_cache_dir`；只改变卡数或 host／NUMA 不需要重建缓存。

## 修改显卡数量

修改脚本中的 `CUDA_VISIBLE_DEVICES`、`--nproc_per_node`，同时修改 JSON 中的 `parallel.seq_p_size`。三者的卡数必须一致。

| 卡数 | 脚本中的 `CUDA_VISIBLE_DEVICES` | `--nproc_per_node` | JSON 的 `parallel.seq_p_size` | JSON 的 `vae_decode_parallel` 示例值 |
| --- | --- | --- | --- | --- |
| 8（默认） | `0,1,2,3,4,5,6,7` | `8` | `8` | `true` |
| 4 | `0,1,2,3` | `4` | `4` | `false` |
| 2 | `2,5` | `2` | `2` | `false` |
| 1 | `0` | `1` | `1` | `false` |

例如用第 2、5 号显卡运行：将脚本中的显卡列表改成 `export CUDA_VISIBLE_DEVICES=2,5`，启动参数改成 `--nproc_per_node=2`，JSON 对应字段改为：

```json
{
  "parallel": {
    "seq_p_size": 2,
    "seq_p_attn_type": "ulysses"
  },
  "vae_decode_parallel": false
}
```

以上是需要修改的 JSON 字段片段，其余配置保留。修改后仍运行同一个 `bash` 命令。脚本会设置自己的显卡列表，因此请直接修改脚本，不在命令前另设 `CUDA_VISIBLE_DEVICES`。

H3 有 56 个注意力头，Ulysses SP 大小需要整除 56，对应 **1、2、4、7、8、14、28、56 卡**。这是头数整除条件，不代表任意布局都已验证；此脚本使用单机启动，实际还需有足够的可见 GPU、内存和显存。

表中非 8 卡示例关闭 VAE 并行，便于先运行串行解码。当前 VAE 按时空 tile 分配工作，并不限于 8 卡；多卡时可将 `vae_decode_parallel` 保持为 `true`，单卡时 runner 会自动关闭它。

## 切换 host／NUMA

对应 JSON 配置默认使用 `"shared_cpu_weight_scope": "host"`。切换 NUMA 时，在同一份 JSON 中修改此字段：

```json
{
  "shared_cpu_weight_scope": "numa"
}
```

其余 JSON 字段保留，修改后仍运行同一个启动脚本。共享范围属于启动配置，不作为命令行或单次请求参数。

host 在同机同一 IPC 域共享一份兼容的 CPU 权重；NUMA 按参与 GPU 所属的 NUMA 域建立副本。切换后使用同一个启动脚本。

## 修改任务与输入

直接修改脚本中 Python 命令的 `--task` 和 `--model-variant`，并添加该任务需要的输入参数：

| `--task` | `--model-variant` | 输入参数 |
| --- | --- | --- |
| `t2av`（默认） | `fl2av` | 无 |
| `i2av` | `fl2av` | `--image_path /path/to/first.png` |
| `l2av` | `fl2av` | `--last_frame_path /path/to/last.png` |
| `fl2av` | `fl2av` | 同时提供 `--image_path` 和 `--last_frame_path` |
| `ref2av` | `ref2av` | `--image_path` 或 `--video_path`，可附加 `--audio_path` |

表中 Ref2AV 示例使用专用 `transformer_ref/` 权重。程序也允许 `--model-variant fl2av --task ref2av` 使用基础 transformer；此时应使用 `fl2av` 缓存，不能仅按任务名选择缓存。

例如运行 Ref2AV，将原启动命令中对应的参数替换为以下内容，再保留其他启动参数：

```bash
  --model-variant ref2av \
  --task ref2av \
  --image_path "${lightx2v_path}/assets/inputs/imgs/img_0.jpg" \
```

Ref2AV 同类参考素材可用逗号分隔，音频必须搭配图片或视频。脚本不再自动填充参考图；切换任务时自行修改 `--prompt` 和 `--save_result_path`。

## 其他参数

| 修改位置 | 参数 | 默认值／用途 |
| --- | --- | --- |
| Shell 脚本 | `lightx2v_path`、`model_path` | 项目和权重目录 |
| 推理命令 | `--prompt`、`--seed`、`--save_result_path` | 提示词、随机种子（默认 42）、输出文件 |
| JSON | `shared_cpu_weight_scope` | `host`（默认）或 `numa` |
| 推理命令 | `--config_json` | 使用其他配置文件时修改此路径 |
| JSON | `infer_steps` | `29` |
| JSON | `size` | `[544, 960]`，顺序为高、宽 |
| JSON | `num_frames`、`fps` | `124` 帧、`24` fps |
| JSON | `adaln_cache_dir` | `~/.cache/lightx2v/adaln` |

共享配置覆盖 DiT、文本主干和视频 VAE，并开启对应的 block offload。依赖 Linux SysV 共享内存和 CUDA pinned memory；GPU 激活和工作缓冲仍各自占用显存。当前共享路径不支持 TP、量化、LoRA、compile 或 lazy loading。

共享视频 VAE 使用 `vae_encoder_conv_mode="torch"`。channels-last／FP8 Conv3D 布局尚未接入共享 manifest，不能用于这条共享加载路径。

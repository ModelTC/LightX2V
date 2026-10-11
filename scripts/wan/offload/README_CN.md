# Wan 2.1 CPU block offload 使用指南

[English](README.md) | 简体中文

启动脚本：`run_wan_block_shared_offload.sh`。
配置文件：`configs/offload/block/wan_block_shared.json`。
默认运行 **I2V、host 共享、8 卡、Ulysses SP8**。

直接修改脚本中的路径、显卡和推理参数，共享范围和并行设置修改 JSON。脚本固定设置参数，不自动推算卡数，也不转发 `bash 脚本.sh` 后追加的参数。`TASK`、`CONFIG_JSON`、`SHARED_CPU_WEIGHT_SCOPE`、`WAN_*` 环境变量不能覆盖此脚本的设置。若要在不修改脚本的情况下调整命令行参数，可直接运行完整的 `python -m torch.distributed.run ... -m lightx2v.infer ...` 命令。

## 准备与启动

1. 激活已安装项目依赖的 Python 环境。
2. 修改脚本顶部的 `lightx2v_path` 和 `model_path`，填写项目和权重目录。
3. 修改 JSON 中的 `dit_quantized_ckpt`、`t5_quantized_ckpt`、`clip_quantized_ckpt` 和 `vae_path`，使其指向实际权重。仅修改脚本的 `model_path` 不会改写这些路径。
4. 从项目根目录运行：

```bash
cd /path/to/LightX2V
bash scripts/wan/offload/run_wan_block_shared_offload.sh
```

默认配置需要模型配置、FP8-vLLM DiT `block_*.safetensors`、FP8 T5／CLIP 和 `Wan2.1_VAE.pth`。沿用 `scripts/base/base.sh` 的默认 BF16 推理，注意力后端为 FlashAttention 3。JSON 中的相对权重路径以运行时的工作目录为基准，因此使用上述仓库根目录启动方式；也可填写绝对路径。

## 修改显卡数量

同时修改脚本的 `CUDA_VISIBLE_DEVICES`、`--nproc_per_node` 和 JSON 的 `parallel.seq_p_size`，三者卡数一致：

| 卡数 | `CUDA_VISIBLE_DEVICES` | `--nproc_per_node` | `parallel.seq_p_size` | `parallel.vae_parallel` 示例值 |
| --- | --- | --- | --- | --- |
| 8（默认） | `0,1,2,3,4,5,6,7` | `8` | `8` | `true` |
| 4 | `0,1,2,3` | `4` | `4` | `false` |
| 2 | `2,5` | `2` | `2` | `false` |
| 1 | `0` | `1` | `1` | `false` |

例如使用第 2、5 号显卡：脚本改为 `export CUDA_VISIBLE_DEVICES=2,5` 和 `--nproc_per_node=2`，JSON 的 `parallel` 改为：

```json
{
  "seq_p_size": 2,
  "seq_p_attn_type": "ulysses",
  "cfg_p_size": 1,
  "vae_parallel": false
}
```

以上是 `parallel` 对象的内容，其余配置保留。修改后运行同一个 `bash` 命令。显卡列表直接在脚本里设置，不在命令前通过环境变量覆盖。

默认 14B 模型有 40 个注意力头，Ulysses SP 大小必须整除 40，对应 **1、2、4、5、8、10、20、40 卡**。这是头数整除条件，不代表任意布局都已验证；此脚本使用单机启动，实际还需有足够的可见 GPU、内存和显存。本配置不使用 TP 或 CFG 并行，CFG 引导仍启用。

表中非 8 卡示例将 `parallel.vae_parallel` 设为 `false`，便于先运行串行 VAE。这是示例设置，并非 VAE 只支持 8 卡；启用 VAE 并行还需考虑图像和 latent 的空间切分条件。

## 切换 host／NUMA

对应 JSON 配置默认使用 `"shared_cpu_weight_scope": "host"`。切换 NUMA 时，在同一份 JSON 中修改此字段：

```json
{
  "shared_cpu_weight_scope": "numa"
}
```

其余 JSON 字段保留，修改后仍运行同一个启动脚本。共享范围属于启动配置，不作为命令行或单次请求参数。

host 在同机同一 IPC 域共享一份兼容的 DiT block CPU 权重；NUMA 按参与 GPU 所属的 NUMA 域建立副本。

## 修改任务与输入

默认 `--task i2v`，参考图直接填写在 `--image_path`。修改提示词、负面提示词和输出位置时，直接修改 `--prompt`、`--negative_prompt`、`--save_result_path`。

改为 T2V 时：

1. 将 `--task i2v` 改为 `--task t2v`，删除 `--image_path` 那一行。
2. 将 `model_path` 改为对应的 T2V 模型目录。
3. 准备匹配的 T2V 共享 offload JSON，并修改 `--config_json`。可复制默认 JSON，再更换 DiT、T5、VAE 路径，移除 I2V 专用的 CLIP 配置。
4. 同步修改提示词和输出文件名。

共享适配器要求 FP8-vLLM block 权重，不能仅修改任务名后继续使用 I2V checkpoint。上述卡数列表针对 14B 模型；T2V 的完整生成效果需用实际权重验证。

## 其他参数

| 修改位置 | 参数 | 默认值／用途 |
| --- | --- | --- |
| Shell 脚本 | `lightx2v_path`、`model_path` | 项目和模型目录 |
| 推理命令 | `--image_path`、`--prompt`、`--negative_prompt` | 首帧和提示词 |
| 推理命令 | `--save_result_path`、`--seed` | 输出文件和随机种子（默认 42） |
| JSON | `shared_cpu_weight_scope` | `host`（默认）或 `numa` |
| 推理命令 | `--config_json` | 配置文件路径 |
| JSON | `infer_steps`、`num_frames`、`size` | 40 步、81 帧、`[480, 832]`（高、宽） |

共享依赖 Linux SysV 和 CUDA pinned memory，严格 NUMA 绑定失败时会报错。当前共享路径不支持 TP、LoRA 或 lazy loading；T5、CLIP 和 VAE 未开启 CPU 权重共享，各 GPU 的激活和工作缓冲仍独立占用显存。

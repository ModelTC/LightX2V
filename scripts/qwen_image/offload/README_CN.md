# Qwen-Image-2512 CPU block offload 使用指南

[English](README.md) | 简体中文

启动脚本：`qwen_image_2512_block_shared_offload.sh`。
配置文件：`configs/qwen_image/offload/qwen_image_2512_block_shared.json`。
默认运行 **T2I、host 共享、8 卡、Ulysses SP8**。

直接修改脚本中的路径、显卡和推理参数，并行设置修改 JSON。脚本固定设置参数，不自动推算卡数，也不转发 `bash 脚本.sh` 后追加的参数。`TASK`、`CONFIG_JSON`、`SHARED_CPU_WEIGHT_SCOPE`、`QWEN_*` 环境变量不能覆盖此脚本的设置。若不修改文件，需要直接运行完整的 `python -m torch.distributed.run ... -m lightx2v.infer ...` 命令。

## 准备与启动

1. 激活已安装项目依赖的 Python 环境，默认注意力后端为 FlashAttention 3。
2. 修改脚本顶部的 `lightx2v_path` 和 `model_path`，填写项目和模型目录。模型使用原始 BF16 Diffusers 权重，保留 transformer、文本编码器、tokenizer 和 VAE 等组件。
3. 从项目根目录运行：

```bash
cd /path/to/LightX2V
bash scripts/qwen_image/offload/qwen_image_2512_block_shared_offload.sh
```

## 修改显卡数量

同时修改脚本的 `CUDA_VISIBLE_DEVICES`、`--nproc_per_node` 和 JSON 的 `parallel.seq_p_size`，三者卡数一致：

| 卡数 | `CUDA_VISIBLE_DEVICES` | `--nproc_per_node` | `parallel.seq_p_size` |
| --- | --- | --- | --- |
| 8（默认） | `0,1,2,3,4,5,6,7` | `8` | `8` |
| 4 | `0,1,2,3` | `4` | `4` |
| 2 | `2,5` | `2` | `2` |
| 1 | `0` | `1` | `1` |

例如使用第 2、5 号显卡：脚本改为 `export CUDA_VISIBLE_DEVICES=2,5` 和 `--nproc_per_node=2`，JSON 的 `parallel` 改为：

```json
{
  "seq_p_size": 2,
  "seq_p_attn_type": "ulysses",
  "cfg_p_size": 1
}
```

以上是 `parallel` 对象的内容，其余配置保留。修改后运行同一个 `bash` 命令。显卡列表直接在脚本里设置，不在命令前通过环境变量覆盖。

模型有 24 个注意力头，Ulysses SP 大小必须整除 24，对应 **1、2、3、4、6、8、12、24 卡**。这是头数整除条件，不代表任意布局都已验证；此脚本使用单机启动，实际还需有足够的可见 GPU、内存和显存。本配置不使用 TP 或 CFG 并行，CFG 引导仍启用。

## 切换 host／NUMA

脚本中的 Python 命令默认使用 `--shared_cpu_weight_scope host`。切换 NUMA 时将该参数改为下面的片段；这不是独立命令：

```bash
  --shared_cpu_weight_scope numa \
```

对应 JSON 无需填写此字段；命令行参数优先于旧 JSON 中的同名设置。

host 在同机同一 IPC 域共享一份兼容的 CPU block 权重；NUMA 按参与 GPU 所属的 NUMA 域建立副本。

## 任务与其他参数

当前 BF16 共享权重适配器仅支持 `--task t2i`；不能通过改任务名直接启用 `i2i` 或 layered，图像编辑需要单独适配。

| 修改位置 | 参数 | 默认值／用途 |
| --- | --- | --- |
| Shell 脚本 | `lightx2v_path`、`model_path` | 项目和模型目录 |
| 推理命令 | `--prompt`、`--negative_prompt` | 提示词和负面提示词 |
| 推理命令 | `--save_result_path`、`--seed` | 输出文件和随机种子（默认 42） |
| 推理命令 | `--shared_cpu_weight_scope` | `host`（默认）或 `numa` |
| 推理命令 | `--config_json` | 配置文件路径 |
| JSON | `infer_steps` | `50` |
| JSON | `aspect_ratio` | `"16:9"` |
| JSON | `sample_guide_scale` | `4.0` |

共享依赖 Linux SysV 和 CUDA pinned memory，严格 NUMA 绑定失败时会报错。当前共享入口支持原始 BF16、block 粒度及 `NoCaching`，不支持 TP、量化、LoRA、layered 或 lazy loading。文本编码器和 VAE 仍独立加载，各 GPU 的激活和工作缓冲仍独立占用显存。

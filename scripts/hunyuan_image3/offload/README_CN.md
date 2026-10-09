# Hunyuan Image 3.0 CPU block offload 使用指南

[English](README.md) | 简体中文

启动脚本：`run_hunyuan_image3_block_shared_offload.sh`。
配置文件：`configs/hunyuan_image3/offload/hunyuan_image3_block_shared.json`。
默认运行 **T2I、host 共享、8 卡，TP2 × SP2 × CFG2**。

直接修改脚本中的路径、显卡和推理参数，共享范围和并行设置修改 JSON。脚本固定设置参数，不自动推算卡数，也不转发 `bash 脚本.sh` 后追加的参数。`TASK`、`CONFIG_JSON`、`SHARED_CPU_WEIGHT_SCOPE`、`HUNYUAN_IMAGE3_MODEL_PATH` 等环境变量不能覆盖此脚本的设置；上游代码路径由脚本中的 `HUNYUAN_IMAGE3_REPO_PATH` 指定。若要在不修改脚本的情况下调整命令行参数，可直接运行完整的 `python -m torch.distributed.run ... -m lightx2v.infer ...` 命令。

## 准备与启动

1. 激活已安装项目依赖的 Python 环境。
2. 修改脚本顶部三个路径：
   - `lightx2v_path`：LightX2V 项目目录。
   - `model_path`：带 `model.safetensors.index.json` 的 HunyuanImage-3-Instruct 原始 BF16 权重目录。
   - `HUNYUAN_IMAGE3_REPO_PATH`：HunyuanImage-3.0 上游代码目录，提供 tokenizer、图像处理、VAE／视觉模块。
3. 从项目根目录运行：

```bash
cd /path/to/LightX2V
bash scripts/hunyuan_image3/offload/run_hunyuan_image3_block_shared_offload.sh
```

默认启用 FlashInfer MoE、KV cache、50 步推理和 `think_recaption`。环境需要 FlashInfer 的 `cutlass_fused_moe`、`ActivationType` 和支持缓存参数的 autotune 接口。

## 修改显卡数量

同时修改脚本的 `CUDA_VISIBLE_DEVICES`、`--nproc_per_node`，以及 JSON 的 `parallel`。进程数应等于可见卡数，也等于 TP×SP×CFG。

| 卡数／进程数 | 显卡列表示例 | `tensor_p_size` | `seq_p_size` | `cfg_p_size` | `cfg_mode` |
| --- | --- | --- | --- | --- | --- |
| 1 | `0` | `1` | `1` | `1` | `"serial"` |
| 2 | `2,5` | `2` | `1` | `1` | `"serial"` |
| 4 | `0,1,2,3` | `2` | `2` | `1` | `"serial"` |
| 8（默认） | `0,1,2,3,4,5,6,7` | `2` | `2` | `2` | `"parallel"` |
| 16 | `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15` | `2` | `4` | `2` | `"parallel"` |

例如使用第 2、5 号显卡：脚本改为 `export CUDA_VISIBLE_DEVICES=2,5` 和 `--nproc_per_node=2`，JSON 的 `parallel` 改为：

```json
{
  "pipeline_parallel": false,
  "tensor_p_size": 2,
  "seq_p_size": 1,
  "cfg_p_size": 1,
  "seq_p_attn_type": "ulysses",
  "cfg_mode": "serial"
}
```

以上是 `parallel` 对象的内容，其余配置保留，`enable_cfg` 仍为 `true`。CFG=1 时引导的两个分支串行执行。修改后运行同一个 `bash` 命令；显卡列表直接在脚本里设置，不在命令前通过环境变量覆盖。

同时将 JSON 的 `flashinfer_autotune_cache` 改为对应任务和并行布局的独立路径，例如两卡 T2I 使用 `save_results/hunyuan_image3_flashinfer_autotune_t2i_tp2_sp1_cfg1.json`。首次使用新缓存可能需要调优。

上述 **1、2、4、8、16 卡**是满足头数整除条件的布局示例，不代表每种布局都已验证。模型有 32 个 Q 头、8 个 KV 头，Ulysses 要求 TP×SP 同时整除它们。此脚本使用单机启动，需有对应数量的可见 GPU，以及足够显存容纳 block 缓冲、MoE 工作区、KV cache 和其他模型组件。

## 切换 host／NUMA

对应 JSON 配置默认使用 `"shared_cpu_weight_scope": "host"`。切换 NUMA 时，在同一份 JSON 中修改此字段：

```json
{
  "shared_cpu_weight_scope": "numa"
}
```

其余 JSON 字段保留，修改后仍运行同一个启动脚本。共享范围属于启动配置，不作为命令行或单次请求参数。

host 在同机同一 IPC 域按兼容的 TP 分片共享 transformer block CPU 权重；NUMA 按参与 GPU 所属的 NUMA 域建立相应副本。

## 切换 TI2I 参考图编辑

1. 将命令中的 `--task t2i` 改为 `--task ti2i`。
2. 在启动命令中添加参考图参数：

```bash
  --image_path "${HUNYUAN_IMAGE3_REPO_PATH}/assets/demo_instruct_imgs/input_0_0.png" \
```

3. 修改 `--prompt` 为编辑指令，并修改 `--save_result_path`。

不改 JSON、也不传 `--size` 时，TI2I 仍输出 **1024×1024**，不会自动使用参考图的尺寸。若需按参考图对齐输出尺寸，可删除 JSON 中的 `size`，添加 `"image_size": "auto"` 和 `"align_image_size": true`。

建议为 TI2I 使用独立的 `flashinfer_autotune_cache` 路径，例如默认八卡使用 `save_results/hunyuan_image3_flashinfer_autotune_ti2i_tp2_sp2_cfg2.json`。这用于区分任务的调优缓存，任务选择本身由 `--task` 决定。

切回默认 T2I 时，将 `--task` 改回 `t2i`，删除 `--image_path`。若调整过 JSON，删除 `image_size`、`align_image_size`，恢复 `"size": [1024, 1024]`，并使用对应的 T2I 缓存路径。

## 其他参数

| 修改位置 | 参数 | 默认值／用途 |
| --- | --- | --- |
| Shell 脚本 | 三个路径变量 | 项目、权重和上游代码目录 |
| 推理命令 | `--prompt`、`--save_result_path`、`--seed` | 提示词、输出文件、随机种子（默认 42） |
| JSON | `shared_cpu_weight_scope` | `host`（默认）或 `numa` |
| 推理命令 | `--config_json` | 使用其他配置时修改此路径 |
| JSON | `infer_steps`、`size` | 50 步、T2I 默认 `[1024, 1024]`（高、宽） |
| JSON | `bot_task`、`max_new_tokens` | `"think_recaption"`、`2048`；较长的思考文本会增加耗时 |
| JSON | `flashinfer_autotune_cache` | 与任务及 TP／SP／CFG 匹配的缓存文件 |

如需减少思考文本长度，将 JSON 中的 `max_new_tokens` 改成 `512` 等合适值，或在脚本的 Python 命令中添加 `--max_new_tokens 512`。请从项目根目录启动；`flashinfer_autotune_cache` 的相对路径由程序按 LightX2V 项目根目录解析。

共享依赖 Linux SysV 和 CUDA pinned memory，严格 NUMA 绑定失败时会报错。pre/post、VAE、视觉编码器及 GPU 的 KV cache／激活仍独立占用内存。当前共享路径不支持量化、LoRA、lazy loading、compile、CUDA Graph 或跨 GPU pipeline。

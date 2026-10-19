# Wan2.2

[English](README.md) | [简体中文](README_zh.md)

设置所选脚本中的 `lightx2v_path`、`model_path` 和 `CUDA_VISIBLE_DEVICES`。以下命令使用仓库相对路径，也可使用脚本绝对路径。

## 环境与权重

先完成[环境安装](../../docs/ZH_CN/source/getting_started/quickstart.md)，按所选任务下载需要的基础模型和额外权重：

```bash
# 从huggingface下载
hf download Wan-AI/Wan2.2-T2V-A14B --local-dir Wan-AI/Wan2.2-T2V-A14B
hf download Wan-AI/Wan2.2-I2V-A14B --local-dir Wan-AI/Wan2.2-I2V-A14B

# 下载蒸馏模型
hf download lightx2v/Wan2.2-Distill-Models --local-dir lightx2v/Wan2.2-Distill-Models
hf download lightx2v/Wan2.2-Distill-Loras --local-dir lightx2v/Wan2.2-Distill-Loras

# 下载 UMT5-XXL 文本编码器 FP8 量化权重
hf download lightx2v/Encoders --include "models_t5_umt5-xxl-enc-fp8.pth" --local-dir encoders/t5
```

`model_path` 指向完整基础模型目录。所选 JSON 中的 checkpoint 和 LoRA 路径填写下载后的本地文件路径。

## 基础任务

| 任务 | 脚本 | 模型权重 |
| --- | --- | --- |
| MoE 文生视频 | `run_wan22_moe_t2v.sh` | Wan2.2-T2V-A14B |
| MoE 图生视频 | `run_wan22_moe_i2v.sh` | Wan2.2-I2V-A14B |
| MoE 首尾帧生视频 | `run_wan22_moe_flf2v.sh` | Wan2.2-I2V-A14B |
| TI2V 文生视频 | `run_wan22_t2v.sh` | Wan2.2-TI2V-5B |
| TI2V 图生视频 | `run_wan22_i2v.sh` | Wan2.2-TI2V-5B |
| 图片与音频驱动视频 | `run_wan22_s2v.sh` | Wan2.2-S2V-14B |
| 角色动画 | `run_wan22_animate.sh` | Wan2.2-Animate-14B |
| 角色替换 | `run_wan22_animate_replace.sh` | Wan2.2-Animate-14B 及配套量化权重 |
| Animate2 | `run_wan22_animate2_distill.sh` | Wan2.2-Animate-2-14B 蒸馏权重 |

```bash
bash scripts/wan22/run_wan22_moe_t2v.sh
```

输入、提示词、seed 和输出路径在脚本中修改；推理设置在 `--config_json` 指定的 JSON 中修改。`size` 为 `[高, 宽]`。

切换到 `enable_cfg: false` 的配置时，移除脚本中的 `--negative_prompt`；服务请求也不传 `negative_prompt`，包括空字符串。

MoE 的 T2V、I2V 使用不同权重。TI2V 两种任务共用 [wan_ti2v.json](../../configs/wan22/wan_ti2v.json)，或使用开启 CPU offload、采用 FlashAttention2 的 [wan_ti2v_4090.json](../../configs/wan22/wan_ti2v_4090.json)。

Animate 脚本先预处理驱动视频和参考图片，再执行推理。替换脚本默认使用 [wan_animate_replace_4090.json](../../configs/wan22/wan_animate_replace_4090.json)，需准备量化 DiT/T5/CLIP 权重；原始权重使用 [wan_animate_replace.json](../../configs/wan22/wan_animate_replace.json)。Animate2 直接传参考图片与驱动视频，并在 JSON 中设置组件路径。

## 服务

选择一个 MoE T2V 服务入口，默认端口为 8000，单卡与 TP4 共用 `server/post_t2v.py`：

| 卡数 | 启动脚本 |
| --- | --- |
| 单卡 | `server/start_server_t2v.sh` |
| TP4 | `server/start_server_t2v_tp.sh` |

```bash
# 终端 1：启动服务
bash scripts/wan22/server/start_server_t2v.sh
```

```bash
# 终端 2：模型加载完成后提交请求
python scripts/wan22/server/post_t2v.py
```

I2V、FLF2V 将单卡服务的 `model_path`、`--task`、`--config_json` 改为对应离线脚本的设置，分别使用 `scripts/wan/server/post_i2v.py`、`post_flf2v.py`。任务在服务启动时确定。TI2V 还需设置 `--model_cls wan2.2`，使用对应权重和共用 JSON。

客户端修改 `url` 和 `message`。图片可传 Base64 或服务端可访问的绝对路径，输出位置通过 `save_result_path` 设置。

### Reuse

MoE T2V 在单卡服务中选择 [wan_moe_t2v_reuse.json](../../configs/wan22/wan_moe_t2v_reuse.json)，在 `server/post_t2v.py` 的 `message` 中添加 `reuse`。MoE I2V 在基础 JSON 中添加 `"enable_reuse": true`、`"reuse_cache_path": "save_results/reuse_cache/wan22_moe_i2v"`，并在普通客户端的 `message` 中添加 `reuse`。不同模型或服务使用独立缓存目录。

首次请求 `reuse=False`，复用输入编码时 `reuse=True`。提示词、帧数、尺寸及 I2V 图片路径和内容保持一致，可修改 seed；更换输入先用 `False`。

## 多卡

四卡 TP 示例使用 [wan_moe_t2v_tp.json](../../configs/wan22/wan_moe_t2v_tp.json)：

```bash
bash scripts/wan22/run_wan22_moe_t2v_tp.sh
```

`CUDA_VISIBLE_DEVICES`、`torchrun --nproc_per_node` 与所选 JSON 保持一致。TP／CFG／SP 组合的进程数为 `tensor_p_size × cfg_p_size × seq_p_size`，未设置的维度按 1 计算；Ulysses 卡数须整除模型注意力头数。

Animate 可在 `run_wan22_animate.sh` 中选择 [wan_animate_8gpu.json](../../configs/wan22/wan_animate_8gpu.json)，设置八张 GPU，将推理命令改为 `torchrun --nproc_per_node=8 -m lightx2v.infer`，预处理仍使用 `python`。两卡 LoRA 使用同一脚本，选择 `wan_animate_lora.json` 并添加 `"parallel": {"seq_p_size": 2, "seq_p_attn_type": "ulysses"}`，设置两张 GPU、启动两个推理进程。

## 进阶功能

进阶脚本位于 `inference/`，通过 `--config_json` 选择其他配置。蒸馏、LoRA 和量化配置需填写匹配的本地权重路径。

### Compile

在 `run_wan22_moe_t2v.sh` 中选择 [wan_moe_t2v_compile.json](../../configs/wan22/wan_moe_t2v_compile.json)；MoE I2V 在基础 JSON 中添加 `"use_compile": true`。添加 `"warmup": true` 可在启动时预热；修改服务启动配置后需重启。

### 蒸馏

单卡入口位于 `inference/distill/`：T2V 使用 `run_wan22_moe_t2v_distill.sh`（默认 LoRA），I2V 使用 `run_wan22_moe_i2v_distill.sh`（默认完整蒸馏权重）。修改脚本的 `--config_json` 选择配置：

| 权重形式 | T2V 配置 | I2V 配置 |
| --- | --- | --- |
| 完整蒸馏权重 | [wan_moe_t2v_distill.json](../../configs/wan22/wan_moe_t2v_distill.json) | [wan_moe_i2v_distill_model.json](../../configs/distill/wan22/wan_moe_i2v_distill_model.json) |
| LoRA | [wan_moe_t2v_distill_lora.json](../../configs/distill/wan22/wan_moe_t2v_distill_lora.json) | [wan_moe_i2v_distill_with_lora.json](../../configs/distill/wan22/wan_moe_i2v_distill_with_lora.json) |
| FP8 | — | [wan_moe_i2v_distill_quant.json](../../configs/distill/wan22/wan_moe_i2v_distill_quant.json) |
| NVFP4 + 稀疏注意力 | [wan_moe_t2v_distill_nvfp4_sparse_attn.json](../../configs/wan22/extreme/wan_moe_t2v_distill_nvfp4_sparse_attn.json) | [wan_moe_i2v_distill_nvfp4_sparse_attn.json](../../configs/wan22/extreme/wan_moe_i2v_distill_nvfp4_sparse_attn.json) |

其他蒸馏入口也位于 `inference/distill/`：

| 任务／权重形式 | 脚本 |
| --- | --- |
| FLF2V 完整蒸馏权重 | `run_wan22_distill_moe_flf2v.sh` |
| T2V LoRA，CFG2 × SP4 | `run_wan22_moe_t2v_distill_lora_4step_cfg_ulysses.sh` |

FLF2V 配置位于 [configs/wan22](../../configs/wan22)，另有 FP8、INT8、LoRA 配置，保留各自的采样设置和权重路径。

I2V 完整权重、LoRA、FP8 使用 CFG2 × SP4 时，在对应单卡 JSON 中修改或添加：

```json
{
  "enable_cfg": true,
  "parallel": {
    "seq_p_size": 4,
    "seq_p_attn_type": "ulysses",
    "cfg_p_size": 2
  }
}
```

设置八张 GPU，将 `python -m lightx2v.infer` 改为 `torchrun --nproc_per_node=8 -m lightx2v.infer`。这些设置会同时开启 CFG 和并行推理，其他权重、采样设置沿用所选模型。

### Animate LoRA

在基础脚本中用 `--config_json` 选择对应的完整 JSON，并填写 `lora_configs` 中的本地权重路径：

| 脚本 | 配置 |
| --- | --- |
| `run_wan22_animate.sh` | [wan_animate_lora.json](../../configs/wan22/wan_animate_lora.json) |
| `run_wan22_animate_replace.sh` | [wan_animate_replace_lora.json](../../configs/wan22/wan_animate_replace_lora.json) |

### 其他示例

下表路径相对 `inference/`：

| 功能 | 入口 |
| --- | --- |
| S2V 姿态视频输入 | `run_wan22_s2v_pose_audio.sh` |
| Animate 预处理 | `run_wan22_animate_preprocess.sh` |
| Animate2 八卡 | `run_wan22_animate2_distill_8gpu.sh` |
| CUDA RainFusion 四卡 | `distill/run_wan22_moe_i2v_distill_4step_ulysses_rainfusion.sh` |
| 昇腾 INT8 + RainFusion 两卡 | `distill/run_wan22_moe_i2v_distill_int8_4step_ulysses_npu.sh` |
| NVFP4 + 稀疏注意力，八卡 | `extreme/`：T2V／I2V 示例 |

RainFusion、NPU、Extreme 示例保留各自的算子和通信设置，需使用匹配的硬件与依赖；NPU 脚本通过 `ASCEND_RT_VISIBLE_DEVICES` 选择设备。

## Python API 示例

生成器创建、配置优先级、卸载和多次生成见 [Python API 指南](../../docs/ZH_CN/source/getting_started/python_api.md)。
安装后，修改所选文件中的路径，从仓库根目录运行。各示例保留各自的模型、采样和注意力设置；
改用 Shell 入口的 JSON 属于显式切换配置。

| 用法 | Python 入口 |
| --- | --- |
| MoE T2V | [wan_moe_t2v.py](inference/wan_moe_t2v.py) |
| MoE I2V + offload | [wan_i2v.py](inference/wan_i2v.py) |
| MoE I2V distilled | [wan_i2v_distilled.py](inference/distill/wan_i2v_distilled.py) |
| MoE I2V distilled LoRA | [wan_i2v_with_distill_loras.py](inference/distill/wan_i2v_with_distill_loras.py) |
| TI2V 5B | [wan_ti2v_5b.py](inference/wan_ti2v_5b.py) |
| Animate | [wan_animate.py](inference/wan_animate.py) |

MoE 量化推理需要匹配的高噪声、低噪声 DiT 权重；量化 T5 使用 `t5_quantized_ckpt`。
在所选 JSON 中填写本地文件路径。需要 CPU offload 时设置 `cpu_offload` 和支持的
`offload_granularity`，见[参数卸载](../../docs/ZH_CN/source/deploy_guides/for_low_resource.md)。

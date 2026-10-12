# Wan2.1 与 VACE

[English](README.md) | [简体中文](README_zh.md)

本目录提供 Wan2.1 与 VACE 的推理、服务和蒸馏示例。

设置脚本中的 `lightx2v_path`、`model_path` 和 `CUDA_VISIBLE_DEVICES`。以下命令使用仓库相对路径，也可改用脚本绝对路径。

## 基础任务

| 任务 | 脚本 | 示例权重 |
| --- | --- | --- |
| 文生视频 | `run_wan_t2v.sh` | Wan2.1-T2V-1.3B |
| 图生视频 | `run_wan_i2v.sh` | Wan2.1-I2V-14B-480P |
| 首尾帧生视频 | `run_wan_flf2v.sh` | Wan2.1-FLF2V-14B-720P |
| 参考图／视频条件生成 | `run_wan_vace.sh` | Wan2.1-VACE-14B |

```bash
bash scripts/wan/run_wan_t2v.sh
```

输入、提示词、seed 和输出路径在 sh 中修改；采样参数在 `--config_json` 指定的 JSON 中修改。`num_frames`、`size` 是未传入时的默认值，`size` 为 `[高, 宽]`。

## Server 服务

入口位于 `server/`，权重选择同上，默认监听 `0.0.0.0:8000`。

| 任务 | 启动服务 | 提交请求 |
| --- | --- | --- |
| T2V | `start_server_t2v.sh` | `post_t2v.py` |
| I2V | `start_server_i2v.sh` | `post_i2v.py` |
| FLF2V | `start_server_flf2v.sh` | `post_flf2v.py` |
| VACE | `start_server_vace.sh` | `post_vace.py` |
| T2V Reuse | `start_server_t2v_reuse.sh` | `post_t2v_reuse.py` |

以 T2V 为例：

```bash
# 终端 1：启动服务
bash scripts/wan/server/start_server_t2v.sh
```

```bash
# 终端 2：模型加载完成后提交请求
python scripts/wan/server/post_t2v.py
```

客户端修改 `url` 和 `message`。图片可传 Base64 或绝对路径，VACE 视频通过 `video_path` 传绝对路径；绝对路径须服务端可访问。输出位置通过 `save_result_path` 设置。

### Reuse：复用输入编码

T2V 使用上表 Reuse 入口。I2V 在基础 JSON 中添加 `"enable_reuse": true`、`"reuse_cache_path": "save_results/reuse_cache/wan_i2v"`，使用普通服务，在客户端 `message` 中添加 `reuse`。不同模型或服务使用独立缓存目录。

首次请求 `reuse=False`，复用输入编码时 `reuse=True`。提示词、负面提示词、帧数、尺寸及 I2V 图片路径和内容保持一致，可修改 seed；更换输入先用 `False`。

## 多卡

两卡 Ulysses 示例沿用基础 T2V 的权重和输入，配置为 [wan_t2v_parallel.json](../../configs/wan/wan_t2v_parallel.json)：

```bash
bash scripts/wan/run_wan_t2v_parallel.sh
```

其他任务使用对应权重和基础 JSON，参照示例添加 `parallel` 并改用 `torchrun`。显卡数、`--nproc_per_node`、`parallel.seq_p_size` 保持一致，Ulysses 卡数须整除模型注意力头数。

## 进阶功能

以下入口位于 `inference/`。蒸馏和 Lazy loading 的额外权重路径需在对应 JSON 中设置。

| 功能 | 入口 |
| --- | --- |
| Compile | `run_wan_t2v_compile.sh` |
| Lazy loading | `run_wan_i2v_lazy_load.sh` |
| Sol-Attn | `run_wan_i2v_sol_attn.sh` |
| 共享 CPU offload | [使用说明](inference/shared_offload/README_CN.md) |
| 连续内存布局与 baseline 对比 | [使用说明](inference/offload_layout/README_CN.md) |

T2V／I2V 在基础 JSON 中添加 `"use_compile": true` 开启编译、`"warmup": true` 开启预热。I2V 使用普通入口；服务配置修改后需重启。

### 蒸馏

`inference/distill/` 提供单卡 4 步示例，按权重形式选择；model、LoRA、FP8 的多卡设置参照上一节。

| 权重形式 | T2V | I2V |
| --- | --- | --- |
| 完整蒸馏 DiT | `run_wan_t2v_distill_model.sh` | `run_wan_i2v_distill_model.sh` |
| 原始模型 + 蒸馏 LoRA | `run_wan_t2v_distill_lora.sh` | `run_wan_i2v_distill_lora.sh` |
| FP8 蒸馏 DiT | `run_wan_t2v_distill_fp8.sh` | `run_wan_i2v_distill_fp8.sh` |
| INT8 蒸馏 DiT | — | `run_wan_i2v_distill_int8.sh` |

- `run_wan_i2v_distill_single_file.sh`：蒸馏 DiT 单文件与配置、编码器、VAE 位于同一模型目录。
- `run_wan_t2v_distill_dynamic_cfg.sh`：dynamic CFG 模型，JSON 中的 `cfg_scale` 控制引导强度。
- `run_wan_t2v_distill_model_ulysses.sh`：完整蒸馏 T2V 模型的 8 卡 Ulysses 示例。

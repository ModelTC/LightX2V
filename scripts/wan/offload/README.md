# Wan 2.1 CPU block offload 使用指南

共享 offload 使用统一入口：`run_wan_block_shared_offload.sh`，使用 `configs/offload/block/wan_block_shared.json`。默认 **TASK=i2v、host 共享、8 卡、Ulysses SP8**，通过环境变量切换 NUMA 或指定显卡。使用当前环境的 `python -m torch.distributed.run`，以下命令均为单机推理。

## 准备环境和权重

激活已安装项目依赖的 Python 环境，在仓库根目录执行：

```bash
cd /path/to/lightx2v_offload_opt
export WAN_MODEL_PATH="$PWD/models/Wan2.1-I2V-14B-720P-Lightx2v"
```

默认模型目录需要模型配置、FP8 DiT block、FP8 T5／CLIP 和 `Wan2.1_VAE.pth`。默认 FP16 推理、FP8-vLLM 算子和 FlashAttention 3。

权重放在其他目录时，除设置 `WAN_MODEL_PATH` 外，还需修改 JSON 中的 `dit_quantized_ckpt`、`t5_quantized_ckpt`、`clip_quantized_ckpt` 和 `vae_path`。JSON 内相对路径按仓库根目录解析。

## 启动命令

```bash
# 默认 host + 8 卡：未设置 CUDA_VISIBLE_DEVICES 时使用 0,1,2,3,4,5,6,7
bash scripts/wan/offload/run_wan_block_shared_offload.sh

# NUMA + 8 卡
SHARED_CPU_WEIGHT_SCOPE=numa \
bash scripts/wan/offload/run_wan_block_shared_offload.sh

# host + 4 卡
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/wan/offload/run_wan_block_shared_offload.sh

# NUMA + 2 卡
CUDA_VISIBLE_DEVICES=2,5 SHARED_CPU_WEIGHT_SCOPE=numa \
bash scripts/wan/offload/run_wan_block_shared_offload.sh

# 单卡
CUDA_VISIBLE_DEVICES=0 \
bash scripts/wan/offload/run_wan_block_shared_offload.sh
```

支持 **1、2、4、5、8、10、20、40 卡**。默认模型有 40 个注意力头，Ulysses SP 大小必须整除 40。默认模板设置 SP 等于可见卡数，使用全部指定显卡，不支持的卡数会直接报错。8 卡保留 VAE 并行，其余卡数关闭它。本入口不使用 TP 或 CFG 并行，无需设置 `NPROC_PER_NODE`。

## 选择任务与输入

`TASK=i2v`（默认）传入 `WAN_IMAGE_PATH` 作为首帧。`TASK=t2v` 不传图片，需要显式提供对应的 `WAN_MODEL_PATH` 和 `CONFIG_JSON`，不能继续使用默认的 I2V checkpoint：

```bash
TASK=t2v WAN_MODEL_PATH=/path/to/Wan2.1-T2V-14B \
CONFIG_JSON=/path/to/wan_t2v_shared.json \
bash scripts/wan/offload/run_wan_block_shared_offload.sh
```

T2V 配置可从默认 JSON 复制后修改：指定与 T2V 模型匹配的 FP8-vLLM DiT block、T5、VAE 路径，去掉 I2V 专用的 CLIP 路径设置，并使并行规模与可见卡数一致。此入口的卡数列表面向 40 个注意力头的 14B 模型。任务参数已接入启动链；对应 T2V 权重的完整推理效果需要用实际资产验证。

## 参数与配置

| 环境变量 | 默认值／用途 |
| --- | --- |
| `TASK` | 默认 `i2v`；可设 `t2v`，需匹配的模型和配置 |
| `CUDA_VISIBLE_DEVICES` | 未设置时为 `0,1,2,3,4,5,6,7`；支持不连续编号 |
| `SHARED_CPU_WEIGHT_SCOPE` | 未设置时读取 JSON 的 scope，默认 `host`；可设 `host` 或 `numa` |
| `WAN_MODEL_PATH` | 仓库内 `models/Wan2.1-I2V-14B-720P-Lightx2v` |
| `WAN_IMAGE_PATH` | 仓库内 `assets/inputs/imgs/img_0.jpg` |
| `WAN_PROMPT` | 默认提示词见脚本，可直接覆盖 |
| `WAN_SAVE_RESULT_PATH` | `save_results/output_lightx2v_wan_<TASK>_block_shared_offload.mp4` |
| `SEED` | `42` |
| `CONFIG_JSON` | 可选完整 JSON 配置，未设置时使用上述默认配置 |

```bash
CUDA_VISIBLE_DEVICES=0,1 WAN_IMAGE_PATH="$PWD/assets/inputs/imgs/img_0.jpg" \
WAN_PROMPT="A cat enjoys the sea breeze on a sunny beach." \
WAN_SAVE_RESULT_PATH="$PWD/save_results/wan_custom.mp4" SEED=123 \
bash scripts/wan/offload/run_wan_block_shared_offload.sh
```

默认 40 步、81 帧、480×832，修改 JSON 中的 `infer_steps`、`num_frames` 和 `size`（高、宽）。负面提示词位于脚本的 `--negative_prompt`。

脚本从配置生成独立临时 JSON，退出时删除，不改写源文件。显式传入 `CONFIG_JSON` 时保留其中的并行和 VAE 设置，TP×SP×CFG 必须等于可见卡数；`SHARED_CPU_WEIGHT_SCOPE` 若设置则覆盖该配置的 scope，否则保留配置中的值。环境变量中的模型、输入、配置和输出相对路径以调用脚本时的目录为基准。

host 在同机同一 IPC 域共享一份 DiT block CPU 权重；NUMA 按参与 GPU 的 NUMA 域建立副本。依赖 Linux SysV 和 CUDA pinned memory，严格 NUMA 绑定不可用时会报错。当前共享路径要求 FP8-vLLM block 权重，不支持 TP、LoRA 或 lazy loading；T5、CLIP 和 VAE 未因此开启共享。各 GPU 的激活和工作缓冲仍需要足够显存。

# Qwen-Image-2512 CPU block offload 使用指南

共享 offload 使用不含任务名的统一入口：`qwen_image_2512_block_shared_offload.sh`，使用 `configs/qwen_image/offload/qwen_image_2512_block_shared.json`。默认 **TASK=t2i、host 共享、8 卡、Ulysses SP8**，通过环境变量切换 NUMA 或指定显卡。使用当前环境的 `python -m torch.distributed.run`，以下命令均为单机推理。

## 准备环境和权重

激活已安装项目依赖的 Python 环境，默认注意力后端为 FlashAttention 3。在仓库根目录执行：

```bash
cd /path/to/lightx2v_offload_opt
export QWEN_MODEL_PATH="$PWD/models/Qwen-Image-2512"
```

模型目录使用原始 BF16 Diffusers 权重，保留 transformer、文本编码器、tokenizer 和 VAE 等组件。

## 启动命令

```bash
# 默认 host + 8 卡：未设置 CUDA_VISIBLE_DEVICES 时使用 0,1,2,3,4,5,6,7
bash scripts/qwen_image/offload/qwen_image_2512_block_shared_offload.sh

# NUMA + 8 卡
SHARED_CPU_WEIGHT_SCOPE=numa \
bash scripts/qwen_image/offload/qwen_image_2512_block_shared_offload.sh

# host + 4 卡
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/qwen_image/offload/qwen_image_2512_block_shared_offload.sh

# NUMA + 2 卡
CUDA_VISIBLE_DEVICES=2,5 SHARED_CPU_WEIGHT_SCOPE=numa \
bash scripts/qwen_image/offload/qwen_image_2512_block_shared_offload.sh

# 单卡
CUDA_VISIBLE_DEVICES=0 \
bash scripts/qwen_image/offload/qwen_image_2512_block_shared_offload.sh
```

支持 **1、2、3、4、6、8、12、24 卡**。模型有 24 个注意力头，Ulysses SP 大小必须整除 24。默认模板设置 SP 等于可见卡数，使用全部指定显卡，不支持的卡数会直接报错。本入口不使用 TP 或 CFG 并行，CFG 引导计算仍启用。无需设置 `NPROC_PER_NODE`。

## 选择任务

任务通过 `TASK` 传给推理入口。当前 Qwen BF16 共享权重适配器仅支持 `t2i`，可显式设置 `TASK=t2i`；`i2i` 等任务会在启动前报错。文件名通用化不会扩展适配器的任务能力，图像编辑需要另行接入。

## 参数与配置

| 环境变量 | 默认值／用途 |
| --- | --- |
| `TASK` | 默认 `t2i`，当前共享适配器仅支持此任务 |
| `CUDA_VISIBLE_DEVICES` | 未设置时为 `0,1,2,3,4,5,6,7`；支持不连续编号 |
| `SHARED_CPU_WEIGHT_SCOPE` | 未设置时读取 JSON 的 scope，默认 `host`；可设 `host` 或 `numa` |
| `QWEN_MODEL_PATH` | 仓库内 `models/Qwen-Image-2512` |
| `QWEN_PROMPT` | 默认提示词见脚本，可直接覆盖 |
| `QWEN_SAVE_RESULT_PATH` | `save_results/qwen_image_t2i_2512_block_shared_offload.png` |
| `SEED` | `42` |
| `CONFIG_JSON` | 可选完整 JSON 配置，未设置时使用上述默认配置 |

```bash
CUDA_VISIBLE_DEVICES=0,1 QWEN_PROMPT="A small coffee shop beside a lake." \
QWEN_SAVE_RESULT_PATH="$PWD/save_results/qwen_custom.png" SEED=123 \
bash scripts/qwen_image/offload/qwen_image_2512_block_shared_offload.sh
```

默认 50 步、16:9、CFG scale 4；分别修改 JSON 的 `infer_steps`、`aspect_ratio`、`sample_guide_scale`。负面提示词位于脚本的 `--negative_prompt`。

脚本从配置生成独立临时 JSON，退出时删除，不改写源文件。显式传入 `CONFIG_JSON` 时保留其中的并行设置，TP×SP×CFG 必须等于可见卡数；`SHARED_CPU_WEIGHT_SCOPE` 若设置则覆盖该配置的 scope，否则保留配置中的值。模型、配置和输出的相对路径以调用脚本时的目录为基准。

host 在同机同一 IPC 域共享一份 CPU block 权重；NUMA 按参与 GPU 的 NUMA 域建立副本。依赖 Linux SysV 和 CUDA pinned memory，严格 NUMA 绑定不可用时会报错。当前共享入口支持原始 BF16、T2I、block 粒度及 `NoCaching`；不支持 TP、量化、LoRA、layered 或 lazy loading。文本编码器和 VAE 仍独立加载，各 GPU 仍需足够的工作显存。

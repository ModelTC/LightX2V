# Hunyuan Image 3.0 CPU block offload 使用指南

共享 offload 使用统一入口：`run_hunyuan_image3_block_shared_offload.sh`，使用 `configs/hunyuan_image3/offload/hunyuan_image3_block_shared.json`。默认 **TASK=t2i、host 共享、8 卡，TP2 × SP2 × CFG2**，通过环境变量切换 NUMA 或指定显卡。使用当前环境的 `python -m torch.distributed.run`，以下命令均为单机推理。

## 准备环境和模型

激活已安装项目依赖的 Python 环境，在仓库根目录执行：

```bash
cd /path/to/lightx2v_offload_opt
export HUNYUAN_IMAGE3_MODEL_PATH="$PWD/../HunyuanImage-3-Instruct"
export HUNYUAN_IMAGE3_REPO_PATH="$PWD/../HunyuanImage-3.0"
```

两个路径分别为带 `model.safetensors.index.json` 的原始 BF16 权重目录，以及提供 tokenizer、图像处理、VAE／视觉模块的上游代码目录；默认值也是仓库相邻的这两个目录。

默认启用 FlashInfer MoE、KV cache、50 步推理和 `think_recaption`。环境需要 FlashInfer 的 `cutlass_fused_moe`、`ActivationType` 和支持缓存参数的 autotune 接口。

## 启动命令

```bash
# 默认 host + 8 卡：未设置 CUDA_VISIBLE_DEVICES 时使用 0,1,2,3,4,5,6,7
bash scripts/hunyuan_image3/offload/run_hunyuan_image3_block_shared_offload.sh

# NUMA + 8 卡
SHARED_CPU_WEIGHT_SCOPE=numa \
bash scripts/hunyuan_image3/offload/run_hunyuan_image3_block_shared_offload.sh

# host + 4 卡
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/hunyuan_image3/offload/run_hunyuan_image3_block_shared_offload.sh

# NUMA + 2 卡
CUDA_VISIBLE_DEVICES=2,5 SHARED_CPU_WEIGHT_SCOPE=numa \
bash scripts/hunyuan_image3/offload/run_hunyuan_image3_block_shared_offload.sh

# 单卡
CUDA_VISIBLE_DEVICES=0 \
bash scripts/hunyuan_image3/offload/run_hunyuan_image3_block_shared_offload.sh
```

默认模板按可见显卡数量选择以下并行配置：

| GPU 数 | TP | SP | CFG 并行大小 | CFG 方式 |
| --- | ---: | ---: | ---: | --- |
| 1 | 1 | 1 | 1 | serial |
| 2 | 2 | 1 | 1 | serial |
| 4 | 2 | 2 | 1 | serial |
| 8 | 2 | 2 | 2 | parallel |
| 16 | 2 | 4 | 2 | parallel |

支持 **1、2、4、8、16 卡**，使用全部指定显卡，进程数等于 TP×SP×CFG。模型有 32 个 Q 头、8 个 KV 头，Ulysses 要求 TP×SP 同时整除它们；不支持的卡数会直接报错。CFG=1 仍计算引导，两个分支串行执行。无需设置 `NPROC_PER_NODE` 或 TP／SP／CFG 环境变量。

单卡也需要容纳 block 缓冲、MoE 工作区、KV cache 及其他模型组件；并行切分条件满足不代表显存一定够。

## 选择任务与输入

`TASK=t2i` 为文生图，`TASK=ti2i` 为参考图编辑，二者使用同一个脚本和配置模板：

```bash
TASK=ti2i CUDA_VISIBLE_DEVICES=0,1,2,3 \
HUNYUAN_IMAGE3_IMAGE_PATH=/path/to/reference.png \
HUNYUAN_IMAGE3_PROMPT="将参考图改成新年主题的宠物海报。" \
bash scripts/hunyuan_image3/offload/run_hunyuan_image3_block_shared_offload.sh
```

TI2I 未指定图片时使用上游示例 `assets/demo_instruct_imgs/input_0_0.png`。使用默认模板时，T2I 默认 1024×1024；TI2I 设置 `image_size=auto` 和 `align_image_size=true`，按参考图对齐输出尺寸。FlashInfer autotune 缓存按任务和 TP／SP／CFG 分开。

## 参数与配置

| 环境变量 | 默认值／用途 |
| --- | --- |
| `TASK` | 默认 `t2i`；可设为 `ti2i` |
| `CUDA_VISIBLE_DEVICES` | 未设置时为 `0,1,2,3,4,5,6,7`；支持不连续编号 |
| `SHARED_CPU_WEIGHT_SCOPE` | 未设置时读取 JSON 的 scope，默认 `host`；可设 `host` 或 `numa` |
| `HUNYUAN_IMAGE3_MODEL_PATH` | 仓库相邻的 `HunyuanImage-3-Instruct` |
| `HUNYUAN_IMAGE3_REPO_PATH` | 仓库相邻的 `HunyuanImage-3.0` |
| `HUNYUAN_IMAGE3_IMAGE_PATH` | TI2I 参考图，默认使用上游示例 |
| `HUNYUAN_IMAGE3_PROMPT` | 默认提示词见脚本，可直接覆盖 |
| `HUNYUAN_IMAGE3_SAVE_RESULT_PATH` | `save_results/hunyuan_image3_<TASK>_block_shared_offload.png` |
| `SEED` | `42` |
| `CONFIG_JSON` | 可选完整 JSON 配置，未设置时使用上述默认配置 |

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
HUNYUAN_IMAGE3_PROMPT="生成图片：雪山脚下的一座木屋，清晨阳光。" \
HUNYUAN_IMAGE3_SAVE_RESULT_PATH="$PWD/save_results/hunyuan_custom.png" SEED=123 \
bash scripts/hunyuan_image3/offload/run_hunyuan_image3_block_shared_offload.sh \
  --max_new_tokens 512
```

脚本支持附加 `lightx2v.infer` 的请求参数，例如 `--max_new_tokens`；任务通过 `TASK` 选择，启动配置通过 `CONFIG_JSON` 指定。默认 `max_new_tokens=2048`，较长的思考／重描述文本会增加推理耗时。默认 50 步，可修改 JSON 的 `infer_steps`；T2I 尺寸由 `size`（高、宽） 指定，TI2I 默认使用参考图尺寸策略。

脚本从配置生成独立临时 JSON，退出时删除，不改写源文件。使用默认模板时，FlashInfer autotune 缓存路径随任务和 TP／SP／CFG 改变，首次使用可能需要调优。显式传入 `CONFIG_JSON` 时须选择与 `TASK` 匹配的尺寸及缓存配置，脚本保留其中的拓扑、尺寸设置和缓存路径，TP×SP×CFG 必须等于可见卡数；`SHARED_CPU_WEIGHT_SCOPE` 若设置则覆盖该配置的 scope，否则保留配置中的值。模型、上游代码、配置和输出的相对路径以调用脚本时的目录为基准。

host 在同机同一 IPC 域共享一份 transformer block CPU 权重；NUMA 按参与 GPU 的 NUMA 域建立副本。共享依赖 Linux SysV、CUDA pinned memory，严格 NUMA 绑定不可用时会报错。pre/post、VAE、视觉编码器及 GPU 的 KV cache／激活仍独立占用内存。当前共享路径不支持量化、LoRA、lazy loading、compile、CUDA Graph 或跨 GPU pipeline。

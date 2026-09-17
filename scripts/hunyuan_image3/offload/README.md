# Hunyuan Image 3.0 CPU block offload 使用指南

本目录支持 T2I／TI2I、私有 CPU 权重、host 共享和 NUMA 共享。可使用默认 8 卡入口，或通过 `_auto.sh` 按可见 GPU 数匹配 TP／SP／CFG。以下均为单机命令，使用当前 Python 环境。

## 准备环境和模型

激活已安装项目依赖的 Python 环境，在仓库根目录执行：

```bash
cd /path/to/lightx2v_offload_opt
export HUNYUAN_IMAGE3_MODEL_PATH="$PWD/../HunyuanImage-3-Instruct"
export HUNYUAN_IMAGE3_REPO_PATH="$PWD/../HunyuanImage-3.0"
```

两个路径分别为带 `model.safetensors.index.json` 的原始 BF16 权重目录，以及提供 tokenizer、图像处理、VAE／视觉模块的上游代码目录。按实际位置修改，默认值也是仓库相邻的这两个目录。

默认启用 FlashInfer MoE、KV cache、50 步推理和 `think_recaption`。环境需提供 FlashInfer 的 `cutlass_fused_moe`、`ActivationType` 及支持缓存参数的 autotune 接口。

## 默认 8 卡入口

以下入口默认使用 **TP2 × SP2 × CFG2 = 8 个进程**。`<task>` 为 `t2i` 或 `ti2i`，JSON 位于 `configs/hunyuan_image3/offload/`：

| 模式 | 启动脚本 | 默认 JSON |
| --- | --- | --- |
| 私有 | `run_hunyuan_image3_<task>_block_offload.sh` | `hunyuan_image3_<task>_block_tp2_sp2_cfg2.json` |
| host | `run_hunyuan_image3_<task>_block_shared_offload_host.sh` | `hunyuan_image3_<task>_block_shared_host_tp2_sp2_cfg2.json` |
| NUMA | `run_hunyuan_image3_<task>_block_shared_offload_numa.sh` | `hunyuan_image3_<task>_block_shared_numa_tp2_sp2_cfg2.json` |

```bash
# 默认 8 卡 TI2I，host 共享
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/hunyuan_image3/offload/run_hunyuan_image3_ti2i_block_shared_offload_host.sh

# 默认 8 卡 T2I，私有 CPU 权重
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/hunyuan_image3/offload/run_hunyuan_image3_t2i_block_offload.sh
```

这些脚本通过 `run_hunyuan_image3_offload.sh` 读取配置并计算进程数，也支持用 `CONFIG_JSON` 选择与任务和共享模式匹配的完整配置。可见 GPU 数需与配置中的 TP×SP×CFG 一致；`NPROC_PER_NODE` 只作一致性检查。需要只指定可见 GPU、自动匹配拓扑时，使用下一节的 `_auto.sh`。

## 按可见 GPU 数启动（`_auto.sh`）

每种共享方式都有独立入口。设置 `CUDA_VISIBLE_DEVICES` 后，脚本读取对应 `_auto.json` 模板，按卡数生成临时运行配置，并使用当前环境的 `python -m torch.distributed.run` 启动。

```bash
# 4 卡 TI2I，host 共享
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/hunyuan_image3/offload/run_hunyuan_image3_ti2i_block_shared_offload_host_auto.sh

# 2 卡 TI2I，NUMA 共享
CUDA_VISIBLE_DEVICES=2,5 \
bash scripts/hunyuan_image3/offload/run_hunyuan_image3_ti2i_block_shared_offload_numa_auto.sh

# 8 卡 T2I，host 共享
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/hunyuan_image3/offload/run_hunyuan_image3_t2i_block_shared_offload_host_auto.sh
```

T2I 的 NUMA 入口为 `run_hunyuan_image3_t2i_block_shared_offload_numa_auto.sh`。单卡只需 `CUDA_VISIBLE_DEVICES=0`，16 卡设置完整的 16 个 GPU 编号。

| 可见 GPU 数 | TP | SP | CFG 并行大小 | CFG 方式 |
| --- | ---: | ---: | ---: | --- |
| 1 | 1 | 1 | 1 | serial |
| 2 | 2 | 1 | 1 | serial |
| 4 | 2 | 2 | 1 | serial |
| 8 | 2 | 2 | 2 | parallel |
| 16 | 2 | 4 | 2 | parallel |

脚本使用全部指定显卡，进程数等于 TP×SP×CFG。CFG=1 仍计算引导，只是串行运行两个分支。模型有 32 个 Q 头、8 个 KV 头，Ulysses 要求 TP×SP 同时整除它们；3、6 等卡数会直接报错，不会自动减少用卡。6 卡机器可以显式选择 4 张。

未设置 `CUDA_VISIBLE_DEVICES` 时使用 GPU 0。无需设置 `NPROC_PER_NODE`、TP／SP／CFG 环境变量或手工生成 JSON。拓扑满足切分条件不代表显存一定够，尤其单卡仍需容纳 block 缓冲、MoE 工作区、KV cache、视觉编码器和 VAE。

## 配置与输入

`_auto.sh` 的配置位于 `configs/hunyuan_image3/offload/`，按任务和 scope 使用四份模板：

```text
hunyuan_image3_t2i_block_shared_host_auto.json
hunyuan_image3_t2i_block_shared_numa_auto.json
hunyuan_image3_ti2i_block_shared_host_auto.json
hunyuan_image3_ti2i_block_shared_numa_auto.json
```

模板保留 8 卡基准参数；`_auto.sh` 按可见卡数在临时副本中设置并行参数，再将副本传给推理。每次启动的临时文件独立，正常结束或报错退出后自动删除，模板本身不被修改。

不同任务／拓扑使用各自的 FlashInfer autotune 缓存，首次运行可能需要生成缓存。常用环境变量：

| 变量 | 用途 |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | 本次推理使用的 GPU 列表 |
| `HUNYUAN_IMAGE3_MODEL_PATH`／`HUNYUAN_IMAGE3_REPO_PATH` | 权重／上游代码目录 |
| `HUNYUAN_IMAGE3_IMAGE_PATH` | TI2I 参考图，默认上游示例图片 |
| `HUNYUAN_IMAGE3_PROMPT` | 提示词 |
| `HUNYUAN_IMAGE3_SAVE_RESULT_PATH` | 输出图片路径 |
| `SEED` | 随机种子，默认 42 |
| `CONFIG_JSON` | 可选完整配置覆盖；显式指定时保留其中的拓扑和缓存路径，scope 和进程数须匹配入口 |
| `LIGHTX2V_DIST_TIMEOUT_SECONDS` | 初始化通信超时，默认 3600 秒 |

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
HUNYUAN_IMAGE3_IMAGE_PATH="$HUNYUAN_IMAGE3_REPO_PATH/assets/demo_instruct_imgs/input_0_0.png" \
HUNYUAN_IMAGE3_PROMPT="将参考图改成新年主题的宠物海报。" \
HUNYUAN_IMAGE3_SAVE_RESULT_PATH="$PWD/save_results/hunyuan_custom.png" \
bash scripts/hunyuan_image3/offload/run_hunyuan_image3_ti2i_block_shared_offload_host_auto.sh
```

脚本也支持附加 `--max_new_tokens` 等请求参数。默认 `max_new_tokens=2048`，较长的思考／重描述文本会增加 offload 推理时间，可按任务需要调整。步数、图像尺寸等启动配置修改对应 `_auto.json` 模板；显式传入 `CONFIG_JSON` 时按完整配置使用，不改其并行参数、缓存路径或原文件。

host 在同机同一 IPC 域共享一份 transformer block CPU 权重；NUMA 每个参与的 GPU NUMA 域一份。共享需要 Linux SysV、CUDA pinned memory，NUMA 还需要可用的拓扑及内存绑定。pre/post、VAE、视觉编码器及 GPU 的 KV cache／激活仍独立占用内存。当前接入不支持量化、LoRA、lazy loading、compile、CUDA Graph 或跨 GPU pipeline。

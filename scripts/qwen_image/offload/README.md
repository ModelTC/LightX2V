# Qwen-Image-2512 CPU block offload 使用指南

本目录用于 Qwen-Image-2512 文生图，提供 DiT block offload、host 共享和 NUMA 共享。可使用固定单卡／8 卡入口，或通过 `_auto.sh` 按可见 GPU 数启动。以下命令均在仓库根目录使用当前 Python 环境执行。

## 准备环境和权重

先激活已安装项目依赖的 Python 环境，默认注意力后端为 FlashAttention 3。

```bash
cd /path/to/lightx2v_offload_opt
export QWEN_MODEL_PATH="$PWD/models/Qwen-Image-2512"
```

将路径改为实际 BF16 Diffusers 模型根目录，保留 transformer、文本编码器、tokenizer 和 VAE 等组件，不需要重新切分权重。

## 固定卡数入口

| 场景 | 启动脚本 |
| --- | --- |
| 单卡，私有 CPU 权重 | `qwen_image_t2i_2512_block_offload.sh` |
| 8 卡，每进程私有 CPU 权重 | `qwen_image_t2i_2512_block_offload_sp8.sh` |
| 8 卡，host 共享 | `qwen_image_t2i_2512_block_shared_offload_host_sp8.sh` |
| 8 卡，NUMA 共享 | `qwen_image_t2i_2512_block_shared_offload_numa_sp8.sh` |

```bash
# 单卡，私有 CPU 权重
CUDA_VISIBLE_DEVICES=0 \
bash scripts/qwen_image/offload/qwen_image_t2i_2512_block_offload.sh

# 固定 8 卡，host 共享
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/qwen_image/offload/qwen_image_t2i_2512_block_shared_offload_host_sp8.sh
```

这些入口支持 `QWEN_MODEL_PATH`、`QWEN_SAVE_RESULT_PATH` 和 `CUDA_VISIBLE_DEVICES`，配置路径写在脚本中，不读取 `CONFIG_JSON`。`_sp8.sh` 固定启动 8 个进程；仅修改可见 GPU 或 `NPROC_PER_NODE` 不会改变进程数，其他卡数使用下一节的 `_auto.sh`。

## 按可见 GPU 数启动（`_auto.sh`）

每种共享方式都有独立入口。设置 `CUDA_VISIBLE_DEVICES` 后，脚本读取对应 `_auto.json` 模板，按卡数生成临时运行配置，并使用当前环境的 `python -m torch.distributed.run` 启动。

```bash
# 4 卡，host 共享
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/qwen_image/offload/qwen_image_t2i_2512_block_shared_offload_host_auto.sh

# 2 卡，NUMA 共享
CUDA_VISIBLE_DEVICES=2,5 \
bash scripts/qwen_image/offload/qwen_image_t2i_2512_block_shared_offload_numa_auto.sh

# 单卡
CUDA_VISIBLE_DEVICES=0 \
bash scripts/qwen_image/offload/qwen_image_t2i_2512_block_shared_offload_host_auto.sh
```

支持 **1、2、3、4、6、8、12、24 卡**，使用 Ulysses SP，SP 大小等于卡数。模型有 24 个注意力头，SP 必须整除 24；本入口不提供 TP 或 CFG 并行。CFG 引导计算仍启用。16 卡机器可显式选择其中 8 或 12 张，不支持的卡数会直接报错，不会自动减少使用的 GPU。

未设置 `CUDA_VISIBLE_DEVICES` 时使用 GPU 0。不需要设置 `NPROC_PER_NODE`，也不需要手工生成 JSON。可见 GPU 必须真实存在，单卡工作显存仍需足够。

## `_auto.sh` 配置和常用参数

配置位于 `configs/qwen_image/offload/`：

| 模式 | 配置模板 |
| --- | --- |
| host | `qwen_image_t2i_2512_block_shared_host_auto.json` |
| NUMA | `qwen_image_t2i_2512_block_shared_numa_auto.json` |

模板保留 8 卡基准参数；`_auto.sh` 按可见卡数在临时副本中设置并行参数，再将副本传给推理。每次启动的临时文件独立，正常结束或报错退出后自动删除，模板本身不被修改。

默认 50 步、16:9、CFG scale 4、seed 42。脚本使用当前环境的 `python -m torch.distributed.run`。

| 环境变量 | 用途 |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | 参与本次推理的 GPU 列表，可不连续 |
| `QWEN_MODEL_PATH` | 模型根目录 |
| `QWEN_SAVE_RESULT_PATH` | 输出图片路径 |
| `CONFIG_JSON` | 可选完整配置覆盖；显式指定时不改写并行设置，进程数和 scope 须匹配入口 |

修改提示词时编辑对应脚本中的 `--prompt`；步数、长宽比、引导强度分别修改对应 `_auto.json` 模板的 `infer_steps`、`aspect_ratio`、`sample_guide_scale`。例如仅改变输出位置：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
QWEN_SAVE_RESULT_PATH="$PWD/save_results/qwen_sp2.png" \
bash scripts/qwen_image/offload/qwen_image_t2i_2512_block_shared_offload_host_auto.sh
```

host 在同机同一 IPC 域共享一份 CPU block 权重；NUMA 每个参与的 GPU NUMA 域一份。共享需要 Linux SysV 和 CUDA pinned memory，严格 NUMA 绑定不可用时可选择 host。当前共享入口支持原始 BF16、`t2i`、block 粒度及 `NoCaching`，不支持 TP、量化、LoRA、layered 或 lazy loading。文本编码器和 VAE 仍独立加载。

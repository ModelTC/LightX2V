# Wan 2.1 I2V CPU block offload 使用指南

本目录提供 Wan 2.1 I2V 的 DiT block offload 和 CPU 权重共享。可使用固定单卡／8 卡入口，或通过 `_auto.sh` 按可见 GPU 数启动。以下均为单机命令，使用当前 Python 环境。

## 准备环境和权重

先激活已安装项目依赖的 Python 环境。以下命令从仓库根目录执行：

```bash
cd /path/to/lightx2v_offload_opt
```

默认权重目录为 `models/Wan2.1-I2V-14B-720P-Lightx2v/`，需要模型配置、FP8 DiT block、FP8 T5／CLIP 和 `Wan2.1_VAE.pth`。配置使用 FP16 推理、FP8-vLLM 算子和 FlashAttention 3，需相应依赖与硬件支持。

## 固定卡数入口

| 场景 | 启动脚本 | 配置文件，位于 `configs/offload/block/` |
| --- | --- | --- |
| 单卡，私有 CPU 权重 | `run_wan_i2v_block_offload.sh` | `wan_i2v_block.json` |
| 8 卡，每进程私有 CPU 权重 | `run_wan_i2v_block_offload_sp8.sh` | `wan_i2v_block_sp8.json` |
| 8 卡，host 共享 | `run_wan_i2v_block_shared_offload_sp8.sh` | `wan_i2v_block_shared_sp8.json` |

```bash
# 单卡，私有 CPU 权重
CUDA_VISIBLE_DEVICES=0 \
bash scripts/wan/offload/run_wan_i2v_block_offload.sh

# 固定 8 卡，host 共享
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/wan/offload/run_wan_i2v_block_shared_offload_sp8.sh
```

这些入口支持 `CUDA_VISIBLE_DEVICES` 和 `WAN_SAVE_RESULT_PATH`，模型与配置路径写在脚本中，不读取 `WAN_MODEL_PATH` 或 `CONFIG_JSON`。`_sp8.sh` 固定启动 8 个进程；仅修改可见 GPU 或 `NPROC_PER_NODE` 不会改变进程数，其他卡数使用下一节的 `_auto.sh`。

## 按可见 GPU 数启动（`_auto.sh`）

每种共享方式都有独立入口。设置 `CUDA_VISIBLE_DEVICES` 后，脚本读取对应 `_auto.json` 模板，按卡数生成临时运行配置，并使用当前环境的 `python -m torch.distributed.run` 启动。

```bash
# 4 卡，host 共享
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/wan/offload/run_wan_i2v_block_shared_offload_host_auto.sh

# 2 卡，NUMA 共享；也可以指定不连续的 GPU 编号
CUDA_VISIBLE_DEVICES=2,5 \
bash scripts/wan/offload/run_wan_i2v_block_shared_offload_numa_auto.sh

# 单卡
CUDA_VISIBLE_DEVICES=0 \
bash scripts/wan/offload/run_wan_i2v_block_shared_offload_host_auto.sh
```

支持 **1、2、4、5、8、10、20、40 卡**，使用 Ulysses SP，SP 大小等于卡数。默认模型有 40 个注意力头，SP 必须整除 40；本入口不提供 TP 或 CFG 并行。3、6、16 等卡数会直接报错，脚本不会自动少用显卡。16 卡机器可显式选择其中 8 或 10 张。未设置 `CUDA_VISIBLE_DEVICES` 时使用 GPU 0。

实际可用的 GPU 数及显存必须满足所选配置；列出的卡数表示并行切分条件，不表示所有机器都能运行。无需设置 `NPROC_PER_NODE` 或手动生成 JSON。

## `_auto.sh` 配置和常用参数

配置位于 `configs/offload/block/`：

| 模式 | 配置模板 |
| --- | --- |
| host | `wan_i2v_block_shared_host_auto.json` |
| NUMA | `wan_i2v_block_shared_numa_auto.json` |

模板保留 8 卡基准参数；`_auto.sh` 按可见卡数在临时副本中设置并行参数，再将副本传给推理。每次启动的临时文件独立，正常结束或报错退出后自动删除，模板本身不被修改。

默认 40 步、81 帧、480×832、seed 42，输入为 `assets/inputs/imgs/img_0.jpg`。使用默认模板时，8 卡保留模板的 VAE 并行设置，其余卡数关闭 VAE 并行。

| 环境变量 | 用途 |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | 要使用的 GPU 列表，全部参与同一次推理 |
| `WAN_MODEL_PATH` | 模型根目录 |
| `WAN_SAVE_RESULT_PATH` | 输出视频路径 |
| `CONFIG_JSON` | 可选完整配置覆盖；显式指定时不改写并行设置，进程数和 scope 须匹配入口 |

权重放在其他目录时，除 `WAN_MODEL_PATH` 外，还需修改所用 JSON 的 `dit_quantized_ckpt`、`t5_quantized_ckpt`、`clip_quantized_ckpt`、`vae_path`。JSON 相对路径按仓库根目录解析。修改输入、提示词或 CLI 中的帧数等参数，直接编辑对应脚本；步数等配置项修改对应 `_auto.json` 模板。

host 在同机同一 IPC 域共享一份 CPU block 权重；NUMA 每个参与的 GPU NUMA 域一份。共享需要 Linux SysV 和 CUDA pinned memory；严格 NUMA 绑定不可用时可选择 host。各 GPU 的工作显存仍独立，当前共享路径要求 FP8-vLLM block 权重，不支持 TP、LoRA 或 lazy loading。

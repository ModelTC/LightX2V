# OpenPI π0.5-LIBERO

该目录提供 π0.5-LIBERO 的权重转换、运行环境准备、本地 rollout、定量评测和
ROS 交互。批量评测从 LightX2V 公共入口启动：

```text
shell -> python -m lightx2v.infer -> OpenPIRunner
      -> 本地 PyTorch policy -> LIBERO/MuJoCo -> 结果文件
```

这条路径不启动 policy server。ROS 路径由 `openpi_node` 和 `libero_node` 通过
topic 直连，同样不启动 OpenPI server。两条路径都使用当前 base Python；仅任务
特异的 Transformers 代码放在私有 overlay 中。

模型实现改编自 Physical Intelligence OpenPI（Apache-2.0）；随仓库提供的
Transformers replacement 文件保留原始版权和许可证声明。

以下命令默认在项目根目录执行：

```bash
cd /data/liuhongda/lightx2v_openpi
conda activate base
```

启动脚本直接调用当前环境中的 `python`，因此运行前应确认
`command -v python` 指向 base 环境。

## 默认路径

| 内容 | 路径 |
| --- | --- |
| Python | 当前已激活的 base 环境 |
| JAX checkpoint | `/data/liuhongda/openpi_data/openpi-assets/checkpoints/pi05_libero` |
| FP32 PyTorch checkpoint | `/data/liuhongda/openpi_data/openpi-assets/checkpoints/pi05_libero_pytorch_fp32` |
| OpenPI 源码 | `/data/liuhongda/openpi` |
| LIBERO | `/data/liuhongda/openpi/third_party/libero` |
| Transformers overlay | `/data/liuhongda/openpi_data/python_deps/openpi_official_pytorch_runtime` |

路径都可以通过下文列出的环境变量覆盖。

### 从零构造 `openpi_data`

`/data/liuhongda/openpi_data` 不是 OpenPI 仓库自带的目录，也不是官方固定路径。
OpenPI 官方下载器默认使用 `~/.cache/openpi`，并通过 `OPENPI_DATA_HOME` 修改缓存
根目录；LightX2V 使用 `OPENPI_DATA_ROOT` 查找同一批资源。本指南将两个变量指向
同一个目录：

```bash
export OPENPI_DATA_HOME=/data/liuhongda/openpi_data
export OPENPI_DATA_ROOT=/data/liuhongda/openpi_data
```

两个变量的职责不同：`OPENPI_DATA_HOME` 只影响 OpenPI 官方下载器，
`OPENPI_DATA_ROOT` 只影响本目录下的 LightX2V 脚本。

如果已经有转换完成的 PyTorch checkpoint，只做推理或评测所需的最小结构是：

```text
openpi_data/
├── openpi-assets/checkpoints/pi05_libero_pytorch_fp32/
│   ├── model.safetensors
│   ├── config.json
│   └── assets/
│       ├── paligemma_tokenizer.model
│       └── physical-intelligence/libero/norm_stats.json
└── python_deps/openpi_official_pytorch_runtime/  # 由环境准备脚本生成
```

如果要从官方 JAX checkpoint 开始转换，转换前还需要：

```text
openpi_data/
├── openpi-assets/checkpoints/pi05_libero/
│   ├── params/                                   # 完整 Orbax checkpoint
│   └── assets/physical-intelligence/libero/norm_stats.json
└── big_vision/paligemma_tokenizer.model
```

在已经执行过 `uv sync` 的 OpenPI 仓库中，可以用官方 `maybe_download()` 下载这两项：

```bash
cd /data/liuhongda/openpi

export OPENPI_DATA_HOME=/data/liuhongda/openpi_data
export OPENPI_DATA_ROOT=/data/liuhongda/openpi_data

uv run --no-sync python -c \
  'from openpi.shared import download; print(download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero"))'

uv run --no-sync python -c \
  'from openpi.shared import download; print(download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}))'
```

下载器会自动创建 `openpi_data` 及其子目录，不需要手工逐级创建。

下载完成后回到 LightX2V，依次生成 Transformers overlay 和 FP32 PyTorch 权重：

```bash
cd /data/liuhongda/lightx2v_openpi
conda activate base

export OPENPI_DATA_ROOT=/data/liuhongda/openpi_data

bash scripts/openpi/1_setup_pytorch_runtime.sh
bash scripts/openpi/2_convert_pi05_libero_to_pytorch.sh
bash scripts/openpi/1_setup_pytorch_runtime.sh check
```

对应的目录生成关系是：

| 操作 | 生成内容 |
| --- | --- |
| OpenPI `maybe_download()` | `openpi-assets/checkpoints/pi05_libero` 和 `big_vision/paligemma_tokenizer.model` |
| `1_setup_pytorch_runtime.sh` | `python_deps/openpi_official_pytorch_runtime` |
| `2_convert_pi05_libero_to_pytorch.sh` | `openpi-assets/checkpoints/pi05_libero_pytorch_fp32` |

LIBERO 仿真环境仍位于 `/data/liuhongda/openpi/third_party/libero`，不放在
`openpi_data` 中。`lerobot/physical-intelligence/libero` 只在微调时需要；
`raw/`、`results/`、`manifests/`、`runtime_configs/` 和各种 cache 目录也都不是
推理或评测的最小依赖。

## 1. 准备运行环境

setup 只安装或修复 OpenPI 所需的小包：base 环境中的 `mujoco==3.2.3`，以及
私有 overlay 中的 `transformers==4.53.2` 和官方 replacement 文件。它不会修改
Python、PyTorch 或 CUDA。

```bash
bash scripts/openpi/1_setup_pytorch_runtime.sh
```

训练或评测前可做只读检查：

```bash
bash scripts/openpi/1_setup_pytorch_runtime.sh check
```

检查覆盖 checkpoint、Transformers replacement、MuJoCo 来源、LIBERO 来源和
CUDA。启动脚本不会在每次运行前重复执行这项检查。

## 2. 转换官方权重

默认将官方 JAX checkpoint 转为 FP32 PyTorch checkpoint：

```bash
bash scripts/openpi/2_convert_pi05_libero_to_pytorch.sh
```

选择输出精度或路径：

```bash
bash scripts/openpi/2_convert_pi05_libero_to_pytorch.sh --precision bfloat16

OPENPI_JAX_CHECKPOINT=/path/to/pi05_libero \
OPENPI_PYTORCH_CHECKPOINT=/path/to/pi05_libero_pytorch_fp32 \
bash scripts/openpi/2_convert_pi05_libero_to_pytorch.sh --precision float32
```

转换器使用 OpenPI 自身的环境，默认是
`/data/liuhongda/openpi/.venv/bin/python`；本地推理仍使用 base Python。输出目录
必须不存在或为空，脚本不会覆盖已有权重。

## 3. 单 episode rollout

默认在 4 号卡运行 `libero_spatial/task_00/init_00`：

```bash
bash scripts/openpi/run_libero_task_i2va.sh
```

指定 suite、task id 和 init-state id：

```bash
CUDA_VISIBLE_DEVICES=6 \
bash scripts/openpi/run_libero_task_i2va.sh libero_goal 3 0
```

默认输出结构：

```text
save_results/openpi_libero_tasks/<suite>/task_XX/init_XX/
├── rollout.mp4
├── actions.npy
├── metrics.json
└── runtime/
```

使用 `OPENPI_LIBERO_RESULT_ROOT=/path/to/results` 修改根目录。直接执行
`run_libero_i2va.sh` 也会运行默认 episode，但 `run_libero_task_i2va.sh` 提供更清晰
的任务目录组织。

## 4. 单卡定量评测

默认在一张卡上顺序评测四个 suite：

```bash
CUDA_VISIBLE_DEVICES=6 \
bash scripts/openpi/run_libero_evaluate_i2va.sh
```

只测一个 suite：

```bash
CUDA_VISIBLE_DEVICES=6 \
OPENPI_EVAL_BENCHMARKS=libero_goal \
bash scripts/openpi/run_libero_evaluate_i2va.sh
```

最小 smoke test 应使用独立输出目录：

```bash
CUDA_VISIBLE_DEVICES=4 \
OPENPI_EVAL_BENCHMARKS=libero_goal \
OPENPI_EVAL_TASK_IDS=3 \
OPENPI_EVAL_NUM_TRIALS_PER_TASK=1 \
OPENPI_EVAL_MAX_STEPS=2 \
OPENPI_EVAL_RESUME=0 \
OPENPI_EVAL_OUTPUT_DIR=/tmp/openpi-libero-smoke \
bash scripts/openpi/run_libero_evaluate_i2va.sh
```

## 5. 按 suite 多卡评测

默认使用 4、5、6、7 号卡，每张卡运行一个 suite：

```bash
bash scripts/openpi/run_libero_evaluate_parallel_i2va.sh
```

两卡运行完整 LIBERO-40：

```bash
CUDA_VISIBLE_DEVICES=4,5 \
bash scripts/openpi/run_libero_evaluate_parallel_i2va.sh
```

并行脚本固定评测四个官方 suite，支持 1、2 或 4 张卡。两卡运行时每张卡顺序执行
两个 suite；每个 suite 都是独立的
`python -m lightx2v.infer` 进程，并使用独立 cache 和 LIBERO config。

默认输出结构：

```text
save_results/pi05_libero_pytorch_fp32_parallel_evaluation/
├── logs/<suite>.log
├── runtime/<suite>/
├── <suite>/
│   ├── episodes.jsonl
│   ├── episodes/<suite>/task_XX/init_XX/metrics.json
│   └── summary.json
└── parallel_summary.json
```

`support/libero_summary.py` 只读取各 suite 的 JSON 结果并生成最终汇总，不是推理入口。
输出锁会阻止两个并行任务同时写入同一目录。

## 6. ROS 单 episode 交互

### 6.1 调用链与职责边界

```text
libero_node
  ├─ 发布 256×256 RGB、8-D state、语言指令和 observation identity
  ▼
openpi_node
  ├─ 同步 agentview / wrist / state / instruction
  ├─ action queue 为空时调用一次 OpenPIPolicy.predict_action_chunk()
  ├─ 得到 10×7 action，保存前 5 个
  └─ 每个新 observation 发布一个 7-D action
  ▼
libero_node
  ├─ env.step(action)
  ├─ 发布下一 observation
  └─ 成功或达到 suite step cap 时结束 episode
```

| 组件 | 职责 |
| --- | --- |
| `OpenPIPolicy` | 加载模型、维护采样 RNG、一次调用生成完整 10-action chunk |
| `openpi_node` | 对齐跨 topic observation，维护 5-action 执行队列 |
| `libero_node` | 构造/重置环境、执行 warmup 和 action、判断成功和步数上限 |
| `SimulatorNode` | 提供 LightX2V ROS 通用状态机和 control/status topic |
| `libero_evaluate` | 按官方顺序调度 task/init state 并持久化指标 |

主要实现文件：

| 职责 | 文件 |
| --- | --- |
| ROS topic 契约 | `lightx2v_ros/src/common/common/contract.py` |
| OpenPI ROS 推理 node | `lightx2v_ros/src/inference/inference/openpi_node/main.py` |
| LIBERO 环境协议 | `lightx2v_ros/src/simulator/simulator/libero_node/env.py` |
| LIBERO observation 适配 | `lightx2v_ros/src/simulator/simulator/libero_node/observer.py` |
| ROS suite 调度与汇总 | `lightx2v_ros/src/simulator/simulator/libero_node/evaluate.py` |
| 通用 simulator 状态机 | `lightx2v_ros/src/simulator/simulator/sim/node.py` |
| 纯模型边界 | `lightx2v/models/runners/openpi/openpi_runner.py` |
| 三种进程的统一启动脚本 | `scripts/openpi/run_libero_ros_i2va.sh` |

ROS 层不复制 PyTorch 模型、输入变换或 quantile 归一化实现，也不启动 OpenPI
policy server。

### 6.2 构建 ROS overlay

首次使用或修改 ROS package 后构建 overlay：

```bash
source /data/liuhongda/ros2_jazzy/install/setup.bash
cd /data/liuhongda/lightx2v_openpi/lightx2v_ros
colcon build --symlink-install --packages-select common simulator inference
source install/local_setup.bash
```

### 6.3 启动单 episode

然后在两个 base 环境终端中启动同一个 episode。两个终端的 `ROS_DOMAIN_ID` 必须
相同；下面让 MuJoCo 使用 4 号卡、模型使用 7 号卡。

Simulator 和 policy 也可以设置相同的 `CUDA_VISIBLE_DEVICES`，在显存允许时共享
一张物理 GPU；这不会改变两个进程的 ROS 解耦关系，只可能带来少量资源竞争。例如
两边都设置 `CUDA_VISIBLE_DEVICES=7` 时，进程内的 7 号卡会映射为逻辑设备 0。
不要手动设置 `MUJOCO_EGL_DEVICE_ID=7`，LIBERO adapter 会自动选择逻辑 EGL 设备 0。

终端 1（LIBERO 仿真）：

```bash
cd /data/liuhongda/lightx2v_openpi
conda activate base
ROS_DOMAIN_ID=77 CUDA_VISIBLE_DEVICES=4 \
bash scripts/openpi/run_libero_ros_i2va.sh simulator libero_goal 3 0
```

终端 2（OpenPI 推理）：

```bash
cd /data/liuhongda/lightx2v_openpi
conda activate base
ROS_DOMAIN_ID=77 CUDA_VISIBLE_DEVICES=7 \
bash scripts/openpi/run_libero_ros_i2va.sh policy
```

可在第三个已 source ROS overlay 的终端查看当前状态和 episode 结果：

```bash
ros2 topic echo /libero/status
```

### 6.4 ROS 数据契约

OpenPI 使用 `/libero` namespace：

| topic | 类型 | 内容 |
| --- | --- | --- |
| `/libero/agentview/image_raw` | `sensor_msgs/Image` | 第三人称 RGB |
| `/libero/wrist/image_raw` | `sensor_msgs/Image` | 腕部 RGB |
| `/libero/state` | `std_msgs/Float64MultiArray` | 8-D 末端/夹爪 state |
| `/libero/observation_context` | `std_msgs/String` | episode、observation、plan epoch 和语言指令 |
| `/libero/action` | `std_msgs/Float64MultiArray` | 带 observation identity 的 7-D action |
| `/libero/success` | `std_msgs/Bool` | 当前任务是否成功 |
| `/libero/control` | `std_msgs/String` | start/set_task 等 JSON 控制命令 |
| `/libero/status` | `std_msgs/String` | 状态机、步数和历史结果 JSON |

相机消息、state 和 context 都携带 `(episode, observation)` 标识。推理 node 仅在
两路图像、state 和 context 完全匹配时处理一次，避免 ROS topic 异步到达造成跨帧
拼接。episode 切换时会丢弃未执行 action，但不会重置 policy RNG，从而保持官方
suite 内连续 RNG。pause/resume 会递增 plan epoch；仿真端会拒绝旧 episode、旧
observation 或旧 plan epoch 的迟到 action。

脚本默认使用 FP32 转换权重和官方 LIBERO checkout；可分别通过
`OPENPI_MODEL_PATH`、`OPENPI_CONFIG`、`OPENPI_LIBERO_ROOT` 覆盖。

## 7. ROS 完整 suite 评测

完整 ROS 评测由 `libero_evaluate` ROS node 调度。它要求 simulator 以
`autostart:=false`、`loop:=false` 启动，并从未运行过 episode 的初始 `ready` 状态
接管，然后按 task id 0–9、init-state id 0–49 的官方顺序发送控制命令。单 suite
使用三个终端；前两个命令与上面相同，只需让 simulator 等待 evaluator 启动：

```bash
ROS_DOMAIN_ID=81 CUDA_VISIBLE_DEVICES=4 OPENPI_ROS_AUTOSTART=false \
bash scripts/openpi/run_libero_ros_i2va.sh simulator libero_spatial 0 0

ROS_DOMAIN_ID=81 CUDA_VISIBLE_DEVICES=7 \
bash scripts/openpi/run_libero_ros_i2va.sh policy

ROS_DOMAIN_ID=81 \
OPENPI_ROS_OUTPUT_DIR=/data/liuhongda/lightx2v_openpi/save_results/openpi_ros_evaluation \
bash scripts/openpi/run_libero_ros_i2va.sh evaluate libero_spatial
```

coordinator 固定以 task id 0–9 为外层、init-state id 0–49 为内层运行 500 个
episode。每个 episode 会立即追加并 `fsync` 到 `episodes.jsonl`，`summary.json`
采用原子更新。已有输出默认拒绝覆盖；确认重跑时设置
`OPENPI_ROS_OVERWRITE=true`。coordinator 只接受 episode 0、history 为空的初始
`ready` simulator。如果初始环境已经是 task 0/init 0，它会直接发送 `start`，避免
多做一次 reset。

四个 suite 并行时必须使用互不相同的 `ROS_DOMAIN_ID`，每个 domain 分别启动
simulator、policy 和 evaluator。只有 4 张卡时，MuJoCo 与 policy 可以共享同一张卡：

| suite | ROS domain | GPU |
| --- | ---: | ---: |
| `libero_spatial` | 81 | 4 |
| `libero_object` | 82 | 5 |
| `libero_goal` | 83 | 6 |
| `libero_10` | 84 | 7 |

最终输出结构：

```text
save_results/openpi_ros_evaluation/
├── libero_spatial/{episodes.jsonl,summary.json}
├── libero_object/{episodes.jsonl,summary.json}
├── libero_goal/{episodes.jsonl,summary.json}
└── libero_10/{episodes.jsonl,summary.json}
```

非 ROS 的完整定量回归入口仍保留：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
bash scripts/openpi/run_libero_evaluate_parallel_i2va.sh
```

该命令经过 `python -m lightx2v.infer`，不是 ROS 路径。ROS coordinator 只负责发送
`set_task/start`、读取 `/libero/status` 和保存汇总，不会把环境循环放回模型 node。

## 8. 评测协议与定量证据

`configs/openpi/pi05_libero_eval.json` 是默认协议来源：每个 suite 包含 10 个任务，
每个任务测试 50 个 init states，共 500 episodes；完整 LIBERO-40 共 2000 个。
环境/策略 seed 为 7/0，先执行 10 个 no-op，每次预测 10 个 action 并执行前 5 个。
四个 suite 的最大步数分别为 220、280、300、520。

| 项目 | 对齐设置 |
| --- | --- |
| 环境渲染 | 256×256 |
| 相机方向 | agentview 和 wrist 均旋转 180° |
| 模型图像输入 | PIL bilinear resize-with-pad 到 224×224，并保持官方 uint8 量化点 |
| state | 末端位置 3 + 四元数转 axis-angle 3 + gripper 2，共 8 维 |
| 数值链路 | ROS state/action 为 float64；quantile JSON 统计量保留 float64 |
| 环境 / policy seed | 7 / 0，policy RNG 在 suite 内连续 |
| episode warmup | 10 次 `[0, 0, 0, 0, 0, 0, -1]`，不送入模型 |
| action horizon / replan | 生成 10 个，执行前 5 个后重新规划 |
| 最大 policy steps | spatial 220、object 280、goal 300、LIBERO-10 520 |

成功条件直接采用 LIBERO `env.step()` 返回的 `done`；warmup 不计入 policy step
上限。模型反归一化后的 7-D action 不做额外夹爪二值化，也不强制降为 float32。

直接评测默认支持断点恢复。`protocol_id`、输入文件 manifest 和保存的 policy RNG
state 用于避免混用不同协议，并保证前缀恢复后的随机数流与一次性运行一致。ROS
coordinator 当前按完整 suite 写新目录，不提供中途恢复。

已有的完整 2000-episode 数据来自直接 LightX2V 评测链路：

| 实现 | spatial | object | goal | LIBERO-10 | 总计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 官方本地 JAX/OpenPI | 493/500 | 492/500 | 486/500 | 472/500 | 1943/2000（97.15%） |
| LightX2V PyTorch 直接评测 | 497/500 | 492/500 | 489/500 | 466/500 | 1944/2000（97.20%） |

二者相差 `+0.05` 个百分点，说明 PyTorch 权重、模型数值路径和直接 LIBERO 协议
处于同一水平。证据位置：

- 官方日志：`/data/liuhongda/openpi/data/libero/official_pi05_libero/logs/`
- LightX2V 汇总：
  `/data/liuhongda/lightx2v_openpi/save_results/pi05_libero_pytorch_fp32_parallel_evaluation_lhdtest/parallel_summary.json`

已实测 `libero_spatial/task_00/init_00`：ROS 和直接评测均在第 79 个 policy step
成功，且都是 16 次模型前向。coordinator 随后正确切换到 `init_01`，并将第一局结果
写入 JSONL/summary；完整 2000 局仍需实际跑完后再作为 ROS 定量结果引用。

## 9. 常用覆盖项

| 环境变量 | 作用 |
| --- | --- |
| `OPENPI_DATA_HOME` | OpenPI 官方下载器的缓存根目录，仅下载阶段使用 |
| `OPENPI_DATA_ROOT` | checkpoint、tokenizer 和 Transformers overlay 的数据根目录 |
| `OPENPI_MODEL_PATH` | PyTorch checkpoint |
| `OPENPI_CONFIG` | 模型 JSON |
| `OPENPI_EVAL_CONFIG` | 评测协议 JSON |
| `OPENPI_LIBERO_ROOT` | LIBERO checkout |
| `OPENPI_TRANSFORMERS_RUNTIME_PATH` | Transformers overlay |
| `CUDA_VISIBLE_DEVICES` | 单卡 GPU，或并行脚本的 GPU 列表 |
| `OPENPI_EVAL_BENCHMARKS` | 单卡评测的 suite 列表 |
| `OPENPI_EVAL_TASK_IDS` | task id 列表 |
| `OPENPI_EVAL_NUM_TRIALS_PER_TASK` | 每任务 trial 数 |
| `OPENPI_EVAL_MAX_STEPS` | 统一覆盖最大步数 |
| `OPENPI_EVAL_VIDEO_POLICY` | `none`、`failures` 或 `all` |
| `OPENPI_EVAL_SAVE_ACTIONS` | 是否保存 action，使用 `0/1` |
| `OPENPI_EVAL_RESUME` | 是否恢复已有结果，使用 `0/1` |
| `OPENPI_EVAL_OUTPUT_DIR` | 单卡评测输出目录 |
| `OPENPI_PARALLEL_OUTPUT_ROOT` | 多卡评测输出目录 |
| `OPENPI_ROS_AUTOSTART` | ROS simulator 是否直接开始；suite 评测必须为 `false` |
| `OPENPI_ROS_OUTPUT_DIR` | ROS suite 结果根目录 |
| `OPENPI_ROS_OVERWRITE` | 是否覆盖已有 ROS suite 结果，默认 `false` |

## 10. 开发检查

```bash
bash -n scripts/openpi/*.sh
pre-commit run --all-files
```

数值路径的关键约束是官方图像 resize/uint8 量化、FP64 动作反归一化、连续 policy
RNG 和 5-action replan queue；清理启动脚本时不应改变这些逻辑。

修改 ROS 文件后需要重新构建 `common simulator inference`；LIBERO adapter 会把
可见的物理 GPU 映射为 EGL 逻辑设备 0。

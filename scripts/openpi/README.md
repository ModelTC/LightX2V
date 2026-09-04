# OpenPI π0.5-LIBERO

该目录提供 π0.5-LIBERO 的权重转换、运行环境准备、本地 rollout 和定量评测。
所有推理都从 LightX2V 公共入口启动：

```text
shell -> python -m lightx2v.infer -> OpenPIRunner
      -> 本地 PyTorch policy -> LIBERO/MuJoCo -> 结果文件
```

不启动 policy server，不使用 ROS，也不切换 Python 环境。OpenPI worker 与
`lightx2v.infer` 使用同一个 base Python；仅任务特异的 Transformers 代码放在私有
overlay 中。

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

## 1. 准备运行环境

setup 只安装或修复 OpenPI 所需的小包：base 环境中的 `mujoco==3.2.3`，以及
私有 overlay 中的 `transformers==4.53.2` 和官方 replacement 文件。它不会修改
Python、PyTorch 或 CUDA。

```bash
bash scripts/openpi/2_setup_pytorch_runtime.sh
```

训练或评测前可做只读检查：

```bash
bash scripts/openpi/2_setup_pytorch_runtime.sh check
```

检查覆盖 checkpoint、Transformers replacement、MuJoCo 来源、LIBERO 来源和
CUDA。启动脚本不会在每次运行前重复执行这项检查。

## 2. 转换官方权重

默认将官方 JAX checkpoint 转为 FP32 PyTorch checkpoint：

```bash
bash scripts/openpi/1_convert_pi05_libero_to_pytorch.sh
```

选择输出精度或路径：

```bash
bash scripts/openpi/1_convert_pi05_libero_to_pytorch.sh --precision bfloat16

OPENPI_JAX_CHECKPOINT=/path/to/pi05_libero \
OPENPI_PYTORCH_CHECKPOINT=/path/to/pi05_libero_pytorch_fp32 \
bash scripts/openpi/1_convert_pi05_libero_to_pytorch.sh --precision float32
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

`libero_summary.py` 只读取各 suite 的 JSON 结果并生成最终汇总，不是推理入口。
输出锁会阻止两个并行任务同时写入同一目录。

## 评测协议

`configs/openpi/pi05_libero_eval.json` 是默认协议来源：每个 suite 包含 10 个任务，
每个任务测试 50 个 init states，共 500 episodes；完整 LIBERO-40 共 2000 个。
环境/策略 seed 为 7/0，先执行 10 个 no-op，每次预测 10 个 action 并执行前 5 个。
四个 suite 的最大步数分别为 220、280、300、520。

评测默认支持断点恢复。`protocol_id`、输入文件 manifest 和保存的 policy RNG state
用于避免混用不同协议，并保证前缀恢复后的随机数流与一次性运行一致。

## 常用覆盖项

| 环境变量 | 作用 |
| --- | --- |
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

## 开发检查

```bash
bash -n scripts/openpi/run_libero_*.sh
python -m unittest discover -s scripts/openpi/tests -p 'test_*.py' -v
python scripts/openpi/tests/validate_pytorch_parity.py --self-check
pre-commit run --all-files
```

数值路径的关键约束是官方图像 resize/uint8 量化、FP64 动作反归一化、连续 policy
RNG 和 5-action replan queue；清理启动脚本时不应改变这些逻辑。

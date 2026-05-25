# PyTorch Trace Profiling

## 概述

LightX2V 提供 `lightx2v.utils.torch_trace_profiler` 模块，基于 PyTorch Profiler 采集 CPU / CUDA kernel 级 trace，并导出为：

- **TensorBoard** 格式（`.pt.trace.json`，在 TensorBoard 的 **PYTORCH PROFILER** 页查看）
- **Chrome Trace** 格式（`.json`，可用 Perfetto、Chrome Tracing 或命令行工具查看）

模块与具体模型无关：任意 Python 代码只需提供 `step_fn`，即可按 schedule 重复执行并导出 trace。

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LIGHTX2V_TORCH_PROFILE` | 关 | 设为 `1` 开启 trace 采集 |
| `LIGHTX2V_TORCH_PROFILE_FORMAT` | `tensorboard` | `tensorboard` / `chrome` / `both` |
| `LIGHTX2V_TORCH_PROFILE_TB_DIR` | `{cwd}/save_results/torch_profile` | TensorBoard logdir |
| `LIGHTX2V_TORCH_PROFILE_CHROME` | `{cwd}/save_results/trace.json` | Chrome trace 输出路径 |
| `LIGHTX2V_TORCH_PROFILE_WAIT` | `1` | schedule：等待步数（不采集） |
| `LIGHTX2V_TORCH_PROFILE_WARMUP` | `3` | schedule：预热步数（采集但不导出） |
| `LIGHTX2V_TORCH_PROFILE_ACTIVE` | `1` | schedule：有效采集步数（导出 trace） |
| `LIGHTX2V_TORCH_PROFILE_STEPS` | `wait+warmup+active` | 总 `prof.step()` 次数，不足时自动补齐 |
| `LIGHTX2V_TORCH_PROFILE_ONCE` | `1` | 整次进程内只 profile 一次 |
| `LIGHTX2V_TORCH_PROFILE_STACK` | `0` | 设为 `1` 采集 Python 调用栈（供 `stack` 子命令使用） |
| `TENSORBOARD_PORT` | `16006` | TensorBoard 端口（共用机器建议改端口） |

### 导出格式说明

| `FORMAT` | 产出 | 查看方式 |
|----------|------|----------|
| `tensorboard` | `{TB_DIR}/*.pt.trace.json` | TensorBoard → **PYTORCH PROFILER** |
| `chrome` | `{CHROME_PATH}` | 见下文「Chrome trace 三种打开方式」 |
| `both` | 上述两者 | TensorBoard + Chrome 工具 |

---

## 快速开始

### 1. 采集 trace

以 Qwen Image 为例，示例脚本 `scripts/torch_profiling/torch_profiling_qwen.sh` 已预置 profile 环境变量：

```bash
bash scripts/torch_profiling/torch_profiling_qwen.sh
```

脚本内主要变量（可按需修改）：

```bash
export LIGHTX2V_TORCH_PROFILE=1
export LIGHTX2V_TORCH_PROFILE_FORMAT=both
export LIGHTX2V_TORCH_PROFILE_TB_DIR=${lightx2v_path}/save_results/torch_profile
export LIGHTX2V_TORCH_PROFILE_CHROME=${lightx2v_path}/save_results/trace.json
```

推理结束后日志会打印 trace 路径及查看命令。

### 2. 查看 TensorBoard

先安装 PyTorch Profiler 插件（一次性）：

```bash
pip install tensorboard torch-tb-profiler
```

**注意：** 必须打开 **PYTORCH PROFILER** 标签页。默认 SCALARS 页没有 `events.out.tfevents.*`，会显示 “No dashboards are active”，属于正常现象。

#### 何时需要 IDE Ports 转发？

| 情况 | 是否需要 IDE Ports |
|------|-------------------|
| 浏览器与 TensorBoard **在同一台机器**上（本地桌面开发） | **不需要** |
| 通过 **Remote SSH** 连远程机器，浏览器在本地笔记本 | **需要**（无论是否 Docker） |

IDE Ports 转发的是**远程主机**上的端口，与是否 Docker 无关。

---

#### 场景 A：推理与 TensorBoard 在同一环境（无 Docker）

使用常规 TensorBoard 命令即可，**不需要**项目自制脚本：

```bash
tensorboard --logdir save_results/torch_profile --port 16006 --bind_all
```

浏览器打开：

```
http://127.0.0.1:16006/#pytorch_profiler
```

若通过 Remote SSH 开发：在 IDE **Ports** 面板转发远程的 `16006`，再在本地浏览器打开上述 URL。

---

#### 场景 B：推理在 Docker 内、TensorBoard 也需在容器内

容器内 TB 监听容器 IP，Remote SSH 下宿主机需 bridge 到容器：

**完整命令：**

```bash
TENSORBOARD_CONTAINER=your_container_name \
LIGHTX2V_TORCH_PROFILE_TB_DIR=/path/to/LightX2V/save_results/torch_profile \
TENSORBOARD_PORT=16006 \
bash scripts/run_tensorboard_docker_bridge.sh
```

**常用简化（先 export 容器名）：**

```bash
export TENSORBOARD_CONTAINER=your_container_name
bash scripts/run_tensorboard_docker_bridge.sh
```

脚本会：在容器内安装/启动 TensorBoard → 宿主机 TCP 代理到容器 → 输出 URL。

**Remote SSH：** 在 IDE **Ports** 转发宿主机 `16006`，浏览器打开 `http://127.0.0.1:16006/#pytorch_profiler`。

---

### 3. 查看Chrome trace

Chrome 格式 trace（`LIGHTX2V_TORCH_PROFILE_FORMAT=chrome` 或 `both` 时的 `trace.json`）的打开方式：

#### 方式 1：Perfetto UI（推荐）

1. 打开 [https://ui.perfetto.dev/](https://ui.perfetto.dev/)
2. **Open trace file**，选择 trace 文件

#### 方式 2：Chrome Tracing

1. 将 trace 下载到本机（Remote 环境需先下载）
2. 浏览器打开 `chrome://tracing`
3. **Load**，选择 JSON 文件

#### 方式 3：命令行 `tools/trace_kernel_inspector.py`

适合终端里快速统计、导出 CSV、查调用栈。完整帮助：

```bash
python tools/trace_kernel_inspector.py --help
python tools/trace_kernel_inspector.py stat --help
python tools/trace_kernel_inspector.py trace --help
python tools/trace_kernel_inspector.py stack --help
python tools/trace_kernel_inspector.py list --help
```

**子命令一览：**

| 子命令 | 作用 |
|--------|------|
| `list` | 列出 trace 中的 `ProfilerStep#N` 与 `gpu_user_annotation` 窗口名（供 `--window` 使用） |
| `stat` | 按 GPU activity 名称聚合 self-time 统计 |
| `trace` | 按时间顺序列出 GPU 事件，含 Correlation / External id |
| `stack` | 按 Correlation 或 External id 查询调用链（GPU kernel → CUDA → Python） |

**`stat` / `trace` 共有选项：**

| 选项 | 说明 |
|------|------|
| `--step ProfilerStep#N` | 指定 profiler step（默认自动选最后一个有效 step） |
| `--window NAME` | 只分析 `record_function(NAME)` 对应的 GPU 区间 |
| `--short-name` | 缩短 kernel 名称显示 |
| `--output out.csv` | 导出 CSV（`stat` / `trace`） |
| `--sort count\|total\|avg` | `stat` 排序方式（默认 `total`） |

**`stack` 选项：**

| 选项 | 说明 |
|------|------|
| `--id ID` | Correlation 或 External id（从 `trace` 输出中获取） |

**典型工作流：**

```bash
# 1. 先看有哪些 step / 窗口
python tools/trace_kernel_inspector.py list save_results/trace.json

# 2. 整体 kernel 耗时 Top 列表
python tools/trace_kernel_inspector.py stat save_results/trace.json --short-name --sort total

# 3. 导出完整时间线（便于 Excel / 二次分析）
python tools/trace_kernel_inspector.py trace save_results/trace.json --output events.csv

# 4. 只看某段 record_function 标记的区域
python tools/trace_kernel_inspector.py stat save_results/trace.json --window my_region --short-name

# 5. 从 trace 输出里取 id，查 Python 调用栈（采集时需 LIGHTX2V_TORCH_PROFILE_STACK=1）
python tools/trace_kernel_inspector.py trace save_results/trace.json
python tools/trace_kernel_inspector.py stack save_results/trace.json --id 386475
```

---

## 在代码中接入

```python
from lightx2v.utils.torch_trace_profiler import TorchTraceProfiler, log_profile_start

profiler = TorchTraceProfiler.from_env()
if profiler.should_run():
    log_profile_start(profiler.cfg)

    def step_fn():
        my_forward()

    profiler.run(step_fn)
```

要点：

- `should_run()`：检查是否开启，以及 `LIGHTX2V_TORCH_PROFILE_ONCE=1` 时是否已采过
- `run(step_fn)`：按 schedule 循环调用 `step_fn()`，每轮末尾 `prof.step()`
- 需要 Python 调用栈：`export LIGHTX2V_TORCH_PROFILE_STACK=1`
- 需要再次采集：`TorchTraceProfiler.reset_session()`
- 子区间分析：在代码里用 `torch.profiler.record_function("my_region")` 包裹，再用 `--window my_region`

Qwen Image 参考实现：`lightx2v/models/networks/qwen_image/infer/transformer_infer.py` 的 `infer_calculating`。

---

## Schedule 说明

默认 schedule 为 **wait=1 / warmup=3 / active=1**（共 5 步）：

```
step 1        : wait    — 不采集
step 2 ~ 4    : warmup  — 采集但不导出（GPU 预热、编译稳定）
step 5        : active  — 采集并导出 trace
```

若修改 `wait/warmup/active`，请确认 `LIGHTX2V_TORCH_PROFILE_STEPS` ≥ 三者之和。

---

## Docker bridge 脚本（仅场景 B）

`scripts/run_tensorboard_docker_bridge.sh` 仅用于 **推理在 Docker 内** 且需在容器内启动 TensorBoard 的情况。

| 变量 | 说明 |
|------|------|
| `TENSORBOARD_CONTAINER` | **必填**，容器名或 ID |
| `LIGHTX2V_TORCH_PROFILE_TB_DIR` | logdir（绝对路径，需为容器内可访问路径） |
| `TENSORBOARD_PORT` | 宿主机监听端口，默认 `16006` |

---

## 常见问题

### TensorBoard 显示 “No dashboards are active”

- 打开 **PYTORCH PROFILER** 页，不是 SCALARS
- 确认 logdir 下有 `.pt.trace.json`
- 确认已安装 `torch-tb-profiler`

### `trace.json` 未生成

- `LIGHTX2V_TORCH_PROFILE_FORMAT` 为 `chrome` 或 `both`
- schedule 已跑完 active 步
- 查看日志中 `[Profile] step=... chrome=...`

### Remote SSH 下 localhost 无响应

- **无 Docker**：远程机上用 `tensorboard --logdir ... --bind_all` 启动，IDE 转发该端口
- **有 Docker**：用 `run_tensorboard_docker_bridge.sh`，IDE 转发宿主机端口

### 共用机器端口冲突

为每位使用者指定不同 `--port` / `TENSORBOARD_PORT`（如 `16006`、`26006`）。

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `lightx2v/utils/torch_trace_profiler.py` | 核心模块 |
| `scripts/run_tensorboard_docker_bridge.sh` | Docker bridge（仅场景 B） |
| `tools/trace_kernel_inspector.py` | Chrome trace 命令行分析 |
| `scripts/torch_profiling/torch_profiling_qwen.sh` | 以Qwen Image为例的 profile 示例推理脚本 |

# PyTorch Trace Profiling

## 概述

LightX2V 提供 `lightx2v.utils.torch_trace_profiler` 模块，基于 PyTorch Profiler 采集 CPU / CUDA kernel 级 trace，并导出为：

- **TensorBoard** 格式（`.pt.trace.json`，在 TensorBoard 的 **PYTORCH PROFILER** 页查看）
- **Chrome Trace** 格式（`.json`，可用 Perfetto 或 Chrome Tracing 查看）

在目标函数的**调用处**使用 `TorchTraceProfileContext` 即可采集；未包裹的调用不受影响。每个进程全局只允许 profile **一个**调用点（首次执行者生效）。

---

## 构造参数

在调用处创建 `TorchTraceProfileContext(...)`，所有参数均有默认值：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `name` | `None` | 调用点标签，用于日志与全局唯一性识别；省略时使用被调用函数的 qualified name |
| `profile_format` | `tensorboard` | `tensorboard` 或 `chrome` |
| `tb_dir` | `{cwd}/save_results/torch_profile` | TensorBoard logdir |
| `chrome_path` | `{cwd}/save_results/trace.json` | Chrome trace 输出路径（`profile_format=chrome` 时生效） |
| `wait` | `1` | schedule：等待步数（不采集） |
| `warmup` | `3` | schedule：预热步数（采集但不导出） |
| `active` | `1` | schedule：有效采集步数（导出 trace） |
| `with_stack` | `False` | 是否采集 Python 调用栈 |
| `tensorboard_port` | `16006` | 日志中提示的 TensorBoard 端口 |

是否采集 trace 取决于代码里是否写了 `TorchTraceProfileContext`；导出路径与 schedule 通过构造参数配置。

### 导出格式说明

| `FORMAT` | 产出 | 查看方式 |
|----------|------|----------|
| `tensorboard` | `{TB_DIR}/*.pt.trace.json` | TensorBoard → **PYTORCH PROFILER** |
| `chrome` | `{CHROME_PATH}` | 见下文「查看 Chrome trace」 |

---

## 快速开始

### 1. 采集 trace

以 Qwen Image 为例，在 `qwen_image_runner.py` 顶部取消注释 import，并在 `run()` 的 `infer_main` 调用处取消注释 profile 代码（可按需修改参数），然后正常运行推理（如 `scripts/qwen_image/qwen_image_i2i_2511.sh`）：

```python
from lightx2v.utils.torch_trace_profiler import TorchTraceProfileContext


with TorchTraceProfileContext(
    "🚀 infer_main",
    tb_dir="save_results/torch_profile",
    with_stack=True,
) as profile:
    profile.run(self.model.infer, self.inputs)
```

推理结束后日志会打印 trace 路径及查看命令。

### 2. 查看 TensorBoard

先安装 PyTorch Profiler 插件（一次性）：

```bash
pip install tensorboard torch-tb-profiler
```

**注意：** 必须打开 **PYTORCH PROFILER** 标签页。默认 SCALARS 页没有 profiler trace 文件，会显示 “No dashboards are active”，属于正常现象。

在 **trace 文件所在的环境** 启动 TensorBoard（logdir 与构造参数 tb_dir 一致）：

```bash
tensorboard --logdir save_results/torch_profile --port 16006 --bind_all
```

浏览器打开：

```
http://127.0.0.1:16006/#pytorch_profiler
```

下面分两种常见部署情况说明，二者可叠加（例如 Remote SSH 连远程宿主机，而推理又在 Docker 里）。

#### Remote SSH

只要 TensorBoard 跑在远程机器上，而浏览器在本地，就需要把远程端口转到本地。

在 IDE **Ports** 面板转发远程的 `16006`（或你指定的 `TENSORBOARD_PORT`），再在本地浏览器打开 `http://127.0.0.1:16006/#pytorch_profiler`。

#### 推理在 Docker 内

trace 写在容器文件系统里，宿主机上直接 `tensorboard --logdir ...` 读不到容器内的 logdir；且容器内 TensorBoard 监听的是容器网络地址，宿主机 `127.0.0.1:16006` 默认也访问不到。

推荐使用 bridge 脚本（需显式指定容器名与 logdir，示例见脚本头部注释）：

```bash
TENSORBOARD_CONTAINER=wyr_lightx2v_h100_202605 \
TORCH_PROFILE_TB_DIR=/data/nvme0/wangyingrui/LightX2V/save_results/torch_profile \
bash /data/nvme0/wangyingrui/wyr_scripts/run_tensorboard_docker_bridge.sh
```

若代码里使用 `tb_dir="save_results/torch_profile"` 且在 LightX2V repo 根目录运行推理，则 `TORCH_PROFILE_TB_DIR` 通常为 `{LightX2V}/save_results/torch_profile` 的**容器内绝对路径**。

脚本会：

1. 在推理同一容器内启动 TensorBoard（`--logdir` 为 `TORCH_PROFILE_TB_DIR`）
2. 在宿主机启动 TCP 代理，把 `0.0.0.0:16006` 转发到容器内 TB 端口

若 logdir 下没有 `*.pt.trace.json`，脚本会打印 WARNING。

### 3. 查看 Chrome trace

Chrome 格式 trace（`profile_format="chrome"` 时的 `trace.json`）可用以下方式打开：

#### Perfetto UI（推荐）

1. 打开 [https://ui.perfetto.dev/](https://ui.perfetto.dev/)
2. **Open trace file**，选择 trace 文件

#### Chrome Tracing

1. 将 trace 下载到本机（Remote 环境需先下载）
2. 浏览器打开 `chrome://tracing`
3. **Load**，选择 JSON 文件

Docker 内采集时，同样需先把 `trace.json` 从容器拷到本机（或挂载目录可见），再在 Perfetto / Chrome 中打开。

---

## 在代码中接入

在**调用处**用 context 包裹，并通过 `.run(func, *args)` 发起 profile；未包裹的代码路径零开销：

```python
from lightx2v.utils.torch_trace_profiler import TorchTraceProfileContext

with TorchTraceProfileContext(
    "my_forward",
    profile_format="tensorboard",
    tb_dir="save_results/torch_profile",
    with_stack=True,
) as profile:
    profile.run(my_forward, arg1, arg2)
```

首次执行该调用点时，会按 schedule **重复调用**目标函数（默认 5 次：wait=1 / warmup=3 / active=1），每轮末尾 `prof.step()`，并在 active 阶段导出 trace。

要点：

- **开关在调用处**：想 profile 哪段逻辑，就在哪行调用外包一层；评测完删除或注释即可
- **全局唯一**：每个进程只允许 profile 一个调用点；若多处写了 context，仅**首次执行**的那个会采集，其余打 log 并正常执行
- 采集完成后，后续调用恢复为单次正常执行
- 需要再次采集：`TorchTraceProfiler.reset_session()`
- 需要 Python 调用栈：构造时设 `with_stack=True`
- 子区间分析：在代码里用 `torch.profiler.record_function("my_region")` 包裹目标代码

Qwen Image 参考实现：`lightx2v/models/runners/qwen_image/qwen_image_runner.py` 的 `run()` 中 `infer_main` 调用处（见注释示例）。

---

## Schedule 说明

默认 schedule 为 **wait=1 / warmup=3 / active=1**（共 5 步）：

```
step 1        : wait    — 不采集
step 2 ~ 4    : warmup  — 采集但不导出（GPU 预热、编译稳定）
step 5        : active  — 采集并导出 trace
```

总 `prof.step()` 次数固定为 `wait + warmup + active`，由三者自动决定。

---

## 常见问题

### TensorBoard 显示 “No dashboards are active”

- 打开 **PYTORCH PROFILER** 页，不是 SCALARS
- 确认 logdir 下有 `.pt.trace.json`
- 确认已安装 `torch-tb-profiler`

### `trace.json` 未生成

- `profile_format="chrome"`
- schedule 已跑完 active 步
- 查看日志中 `[Profile] step=... chrome=...`

### Remote SSH 下 localhost 无响应

TensorBoard 已在远程（或容器已桥接到宿主机）监听 `--bind_all` 后，在 IDE **Ports** 转发对应端口；与是否在 Docker 内无关。

### Docker 内 TB 打不开 / logdir 为空

- 确认 TB 与推理在**同一容器**，且 `--logdir` 为容器内路径
- 确认宿主机已 `-p` 映射或存在到容器 IP 的 TCP 代理

### Trace 里点 GPU kernel 只有 C++ 栈、没有 Python 栈

- 采集时需设 `with_stack=True` 并重新 profile
- GPU kernel 事件本身挂在 CUDA 层；Python 栈在 CPU / `python_function` 侧，可在 TensorBoard Operator 视图或 Perfetto 的 CPU track 中查看

### 共用机器端口冲突

为每位使用者指定不同 `tensorboard_port`（如 `16006`、`26006`）。

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `lightx2v/utils/torch_trace_profiler.py` | 核心模块 |
| `lightx2v/models/runners/qwen_image/qwen_image_runner.py` | Qwen Image 调用处注释示例 |

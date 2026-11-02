# PyTorch Trace Profiling

## Overview

The `lightx2v.utils.torch_trace_profiler` module uses PyTorch Profiler to collect CPU/CUDA traces and export them in two formats:

- **TensorBoard** (`.pt.trace.json`), viewed in the **PYTORCH PROFILER** tab.
- **Chrome Trace** (`.json`), viewed in Perfetto or Chrome Tracing.

Wrap the target call with `TorchTraceProfileContext`; other calls are unaffected. Each process captures only the **first invoked call site** in a profiling session; subsequent calls execute normally after capture. Call `TorchTraceProfiler.reset_session()` to capture again.

---

## Constructor parameters

Create `TorchTraceProfileContext(...)` at the call site. All parameters have defaults:

| Parameter | Default | Description |
| --- | --- | --- |
| `name` | `None` | Call-site label for logs; defaults to the callable's qualified name |
| `profile_format` | `tensorboard` | `tensorboard` or `chrome` |
| `tb_dir` | `{cwd}/save_results/torch_profile` | TensorBoard log directory |
| `chrome_path` | `{cwd}/save_results/trace.json` | Output path when `profile_format=chrome` |
| `wait` | `1` | Calls without collection |
| `warmup` | `3` | Warmup calls, collected but not exported |
| `active` | `1` | Calls included in the exported trace |
| `with_stack` | `False` | Record Python call stacks |
| `tensorboard_port` | `16006` | Port shown in log hints; does not start TensorBoard |

Profiling is enabled by the wrapper at the call site. Configure output paths and the schedule through these parameters.

### Export formats

| Format | Output | Viewer |
| --- | --- | --- |
| `tensorboard` | `{TB_DIR}/*.pt.trace.json` | TensorBoard → **PYTORCH PROFILER** |
| `chrome` | `{CHROME_PATH}` | See "View Chrome traces" below |

---

## Quick start

### 1. Capture a trace

For Qwen Image, import `TorchTraceProfileContext` at the top of `lightx2v/models/runners/qwen_image/qwen_image_runner.py`. In the `infer_main` region of `run()`, replace `self.model.infer(self.inputs)` with the following, keeping the scheduler and progress updates in place:

```python
from lightx2v.utils.torch_trace_profiler import TorchTraceProfileContext


with TorchTraceProfileContext(
    "🚀 infer_main",
    tb_dir="save_results/torch_profile",
    with_stack=True,
) as profile:
    profile.run(self.model.infer, self.inputs)
```

The first capture repeats `model.infer` within the same denoising step, **5 times by default**. Account for mutable state and side effects when choosing the call site. Later steps execute the target once, even when creating a new context.

Run the corresponding inference script, such as `scripts/qwen_image/qwen_image_i2i_2511.sh`. After collection, the logs print the trace location and viewing commands.

### 2. View TensorBoard traces

Install TensorBoard and the PyTorch Profiler plugin once:

```bash
pip install tensorboard torch-tb-profiler
```

**Select the PYTORCH PROFILER tab.** The default SCALARS tab does not display profiler traces and may show "No dashboards are active".

Start TensorBoard in an environment that can read the trace files. Match `--logdir` to the configured `tb_dir`:

```bash
tensorboard --logdir save_results/torch_profile --port 16006 --host 127.0.0.1
```

Open:

```text
http://127.0.0.1:16006/#pytorch_profiler
```

The following scenarios can apply together, for example when using Remote SSH to a host that runs inference in Docker.

#### Remote SSH

When TensorBoard runs on a remote machine and the browser is local, forward the remote port to your machine.

Use your IDE's **Ports** panel to forward `16006`, or the port selected when starting TensorBoard, then open `http://127.0.0.1:16006/#pytorch_profiler` locally.

#### Inference inside Docker

If traces are written to a host-mounted directory, read that directory on the host. Otherwise, copy them out before starting TensorBoard. Replace the container name and absolute container path below:

```bash
docker cp lightx2v-container:/workspace/LightX2V/save_results/torch_profile ./torch_profile
tensorboard --logdir ./torch_profile --port 16006 --host 127.0.0.1
```

With `tb_dir="save_results/torch_profile"` and inference launched from the repository root, the absolute container path is `{LightX2V}/save_results/torch_profile`.

Alternatively, run TensorBoard inside the container:

1. Publish the port when creating the container, for example `-p 127.0.0.1:16006:16006`.
2. Start TensorBoard inside the container using its log directory. Replace `--host 127.0.0.1` with `--bind_all`; these options cannot be used together.
3. If the host is remote, also forward the port as described under Remote SSH.

### 3. View Chrome traces

For `profile_format="chrome"`, open the exported JSON using either viewer below.

#### Perfetto UI (recommended)

1. Open [Perfetto](https://ui.perfetto.dev/).
2. Select **Open trace file** and choose the JSON file.

#### Chrome Tracing

1. Copy the trace to your local machine if it was collected remotely.
2. Open `chrome://tracing` in your browser.
3. Select **Load** and choose the JSON file.

For Docker captures, first copy the JSON out of the container or access it through a mounted directory.

---

## Integrate into your code

Wrap the target call with a context and invoke `.run(func, *args)`. Calls outside the wrapper are unaffected:

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

The first `.run()` in each profiling session repeats the target according to the schedule: 5 calls by default (`wait=1`, `warmup=3`, `active=1`). Each call ends with `prof.step()`, and the active phase is exported.

Key points:

- **Enable at the call site:** wrap the operation you want to profile; remove or comment out the wrapper when finished.
- **Process-wide session:** only the first invoked call site captures. Other call sites and subsequent calls execute the target once; creating a new context does not start another capture.
- To capture again, call `TorchTraceProfiler.reset_session()` after the previous capture finishes. The next invoked call site starts a new capture, including when reusing an existing context.
- Set `with_stack=True` to record Python call stacks.
- Use `torch.profiler.record_function("my_region")` to annotate smaller regions.

For Qwen Image, the integration point is the `infer_main` call in `run()` in `lightx2v/models/runners/qwen_image/qwen_image_runner.py`; see Quick start.

---

## Schedule

The default schedule is **wait=1 / warmup=3 / active=1**. Each step below is one repeated call to the target function:

```text
step 1        : wait    — no collection
steps 2–4     : warmup  — collect without exporting
step 5        : active  — collect and export
```

The number of `prof.step()` calls is `wait + warmup + active`.

---

## Troubleshooting

### TensorBoard shows "No dashboards are active"

- Select **PYTORCH PROFILER**, rather than SCALARS.
- Check that the log directory contains `.pt.trace.json` files.
- Check that `torch-tb-profiler` is installed.

### No `trace.json` was generated

- Set `profile_format="chrome"`.
- Confirm that the schedule completed the active phase.
- Look for `[Profile] step=... chrome=...` in the logs.

### Localhost does not respond over Remote SSH

Confirm that TensorBoard is listening on the remote host, then forward its port through your IDE's **Ports** panel. If TensorBoard runs in Docker, first confirm that the container port is published to the host.

### TensorBoard in Docker is unreachable or its log directory is empty

- When viewing on the host, copy the traces out or expose them through a mounted directory.
- When running TensorBoard inside the container, use the container's log directory and publish its port to the host.

### GPU kernel events have no Python stack

- Set `with_stack=True` and capture again.
- CUDA kernel events show CUDA-side stacks. Inspect related CPU/Python events in TensorBoard's Operator view or Perfetto's CPU tracks for Python stacks.

### Port conflicts on a shared machine

Use a different `--port` in each TensorBoard launch command, such as `16006` or `26006`, and update port mappings or forwarding accordingly. The constructor's `tensorboard_port` parameter only changes the log hints.

---

## Related files

| File | Purpose |
| --- | --- |
| `lightx2v/utils/torch_trace_profiler.py` | Profiling implementation |
| `lightx2v/models/runners/qwen_image/qwen_image_runner.py` | Qwen Image integration point |

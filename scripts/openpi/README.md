# OpenPI pi0.5-LIBERO

This integration runs the converted `pi05_libero` checkpoint locally with
PyTorch. The public entry remains `python -m lightx2v.infer`; a synchronous
child process isolates OpenPI's patched Transformers dependency.

## Paths

The launchers derive their defaults from the LightX2V checkout:

| Variable | Default |
| --- | --- |
| `OPENPI_DATA_ROOT` | `../openpi_data` |
| `OPENPI_PATH` | `../openpi` |
| `OPENPI_MODEL_PATH` | `$OPENPI_DATA_ROOT/openpi-assets/checkpoints/pi05_libero_pytorch` |
| `OPENPI_TRANSFORMERS_RUNTIME_PATH` | `$OPENPI_DATA_ROOT/python_deps/openpi_pytorch_runtime` |
| `OPENPI_LIBERO_ROOT` | `$OPENPI_PATH/third_party/libero` |
| `OPENPI_LIBERO_CONFIG_DIR` | `$OPENPI_DATA_ROOT/runtime_configs/lightx2v_openpi_libero` |
| `OPENPI_PYTHON` | `python` from the active environment |

Override the two roots for a relocated installation:

```bash
OPENPI_DATA_ROOT=/mnt/models/openpi_data \
OPENPI_PATH=/workspace/openpi \
bash scripts/openpi/run_libero_i2va.sh
```

More specific path variables take precedence. The JSON files contain model
semantics and evaluation settings, not machine-specific paths.

## 1. Convert the checkpoint

```bash
bash scripts/openpi/1_convert_pi05_libero_to_pytorch.sh
```

The converter uses OpenPI's official
`examples/convert_jax_model_to_pytorch.py` and writes:

```text
pi05_libero_pytorch/
├── model.safetensors
├── config.json
├── SHA256SUMS
├── .lightx2v_openpi_checkpoint
└── assets/
    ├── paligemma_tokenizer.model
    └── physical-intelligence/libero/norm_stats.json
```

The conversion interpreter defaults to `$OPENPI_PATH/.venv/bin/python`. Use
`OPENPI_CONVERT_PYTHON` to select another isolated environment containing
Transformers 4.53.2. The script refuses a base interpreter, overlapping source
and output checkpoints, broad targets, and unowned non-empty directories.
`OPENPI_FORCE_CONVERT=1` is accepted only for a checkpoint already marked by
the converter.

## 2. Prepare the private runtime

```bash
bash scripts/openpi/2_setup_pytorch_runtime.sh
```

This creates a private dependency directory containing Transformers 4.53.2,
Hugging Face Hub 0.32.3, Tokenizers 0.21.1, and the five OpenPI replacement
files. It does not modify the active environment. Set
`OPENPI_RUNTIME_PYTHON` if the active interpreter is not named `python`.

The shared LightX2V process keeps its normal dependencies. Only the local
OpenPI child receives the private directory at the front of `PYTHONPATH`.

## 3. Local LIBERO rollout

```bash
bash scripts/openpi/run_libero_i2va.sh
```

Call chain:

```text
run_libero_i2va.sh
  -> python -m lightx2v.infer
  -> RUNNER_REGISTER["openpi"]
  -> OpenPIRunner
  -> local libero_rollout process
  -> MP4 + actions + metrics
```

Default task:

```text
benchmark     libero_spatial
task_id      0
init_state   0
```

Default outputs:

```text
save_results/output_openpi_pi05_libero.mp4
save_results/output_openpi_pi05_libero.actions.npy
save_results/output_openpi_pi05_libero.metrics.json
```

Select a task and output files with:

```bash
LIBERO_BENCHMARK=libero_goal \
LIBERO_TASK_ID=3 \
LIBERO_INIT_STATE_ID=0 \
OPENPI_SAVE_VIDEO_PATH=/absolute/output/rollout.mp4 \
OPENPI_SAVE_ACTION_PATH=/absolute/output/rollout.actions.npy \
OPENPI_SAVE_METRICS_PATH=/absolute/output/rollout.metrics.json \
bash scripts/openpi/run_libero_i2va.sh
```

Useful rollout overrides are `OPENPI_SEED`, `OPENPI_ACTIONS_PER_PLAN`,
`OPENPI_NUM_STEPS_WAIT`, `OPENPI_MAX_STEPS`, `OPENPI_RENDER_SIZE`, and
`OPENPI_VIDEO_FPS`.

For task-ID based output organization, use:

```bash
bash scripts/openpi/run_libero_task_i2va.sh <benchmark> <task_id> [init_state_id]
```

For example:

```bash
bash scripts/openpi/run_libero_task_i2va.sh libero_goal 3 0
```

This writes:

```text
save_results/openpi_libero_tasks/libero_goal_task_3/
├── init_state_0.mp4
├── init_state_0.actions.npy
└── init_state_0.metrics.json
```

The worker validates benchmark names and task/init-state ranges against the
loaded LIBERO suite.

## 4. Quantitative evaluation

```bash
bash scripts/openpi/run_libero_evaluate_i2va.sh
```

The default protocol in `configs/openpi/pi05_libero_eval.json` evaluates:

| Suite | Tasks | Trials per task | Step cap |
| --- | ---: | ---: | ---: |
| `libero_spatial` | 10 | 50 | 220 |
| `libero_object` | 10 | 50 | 280 |
| `libero_goal` | 10 | 50 | 300 |
| `libero_10` | 10 | 50 | 520 |

The model is loaded once. Its random generator is reset at each suite boundary,
not at each episode. Results default to:

```text
save_results/openpi_pi05_libero_evaluation/
├── resolved_eval_config.json
├── episodes.jsonl
├── summary.json
└── episodes/<suite>/task_<id>/init_<id>/metrics.json
```

`summary.json` reports per-task, per-suite, and complete-run success rates.
Partial runs leave official success-rate fields null. Rollout exceptions count
as failures; CUDA out-of-memory and setup errors stop evaluation.

Resume is limited to complete suite boundaries. Model, task input, source, and
protocol fingerprints prevent records from different evaluations being mixed.
Use a new `OPENPI_EVAL_OUTPUT_DIR` after changing any of those inputs.

Reference results for the released checkpoint are:

| Suite | Success |
| --- | ---: |
| LIBERO Spatial | 98.8% |
| LIBERO Object | 98.2% |
| LIBERO Goal | 98.0% |
| LIBERO 10 | 92.4% |
| Average | 96.85% |

Evaluator overrides:

| Variable | Value |
| --- | --- |
| `OPENPI_EVAL_OUTPUT_DIR` | Result directory |
| `OPENPI_EVAL_BENCHMARKS` | Comma-separated suites |
| `OPENPI_EVAL_TASK_IDS` | IDs or ranges such as `0,2-4` |
| `OPENPI_EVAL_NUM_TRIALS_PER_TASK` | Trials per task |
| `OPENPI_EVAL_MAX_STEPS` | Common step cap |
| `OPENPI_EVAL_VIDEO_POLICY` | `none`, `failures`, or `all` |
| `OPENPI_EVAL_RESUME` | `1` or `0` |
| `OPENPI_EVAL_FAIL_FAST` | `1` or `0` |
| `OPENPI_EVAL_SAVE_ACTIONS` | `1` or `0` |

Use a separate directory for a short smoke run:

```bash
OPENPI_EVAL_OUTPUT_DIR=/absolute/output/openpi_eval_smoke \
OPENPI_EVAL_BENCHMARKS=libero_spatial \
OPENPI_EVAL_TASK_IDS=0 \
OPENPI_EVAL_NUM_TRIALS_PER_TASK=1 \
OPENPI_EVAL_MAX_STEPS=2 \
OPENPI_EVAL_VIDEO_POLICY=none \
OPENPI_EVAL_RESUME=0 \
bash scripts/openpi/run_libero_evaluate_i2va.sh
```

## 5. ROS rollout

```bash
LIBERO_BENCHMARK=libero_10 \
LIBERO_TASK_ID=5 \
LIBERO_INIT_STATE_ID=0 \
bash scripts/openpi/run_libero_ros_i2va.sh
```

The launcher sources `ROS_SETUP`, or falls back to
`~/ros2_lyrical/install/setup.sh` and `/opt/ros/jazzy/setup.bash`. It builds
the existing `common`, `simulator`, and `inference` packages, then starts
the shared LIBERO simulator and `openpi_node`. The private Transformers path
is applied only to the inference node.

The ROS bridge publishes actions online and does not save MP4, NumPy, or
metrics files. Stop it with `Ctrl+C`; the launcher also terminates the
simulator process.

## 6. Numerical parity

`validate_pytorch_parity.py` compares upstream OpenPI and LightX2V with the
same LIBERO observation and flow noise:

```bash
/path/to/openpi/.venv/bin/python scripts/openpi/validate_pytorch_parity.py
```

Use `--checkpoint`, `--config`, `--sample`, `--output`, and `--device`
for non-default locations. Exit code zero and
`allclose_atol_1e-6: true` indicate a passing comparison.

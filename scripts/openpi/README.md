# OpenPI pi0.5-LIBERO in LightX2V

This integration runs the released `pi05_libero` policy as a native local
PyTorch model. It does not start an OpenPI policy server, web page, or viewer.

## 1. Convert the released checkpoint

```bash
cd /data/liuhongda/LightX2V
bash scripts/openpi/convert_pi05_libero_to_pytorch.sh
```

The default output is:

```text
/data/liuhongda/openpi_data/openpi-assets/checkpoints/pi05_libero_pytorch/
├── model.safetensors
├── config.json
└── assets/
    ├── paligemma_tokenizer.model
    └── physical-intelligence/libero/norm_stats.json
```

## 2. Prepare the isolated PyTorch dependency layer

```bash
bash scripts/openpi/setup_pytorch_runtime.sh
```

This creates
`/data/liuhongda/openpi_data/python_deps/openpi_pytorch_runtime` with patched
Transformers 4.53.2. It does not change packages installed in the base
environment. The launchers set `USE_FLAX=0`, so Transformers does not
auto-import the base environment's JAX/Flax packages.

## 3. Local closed-loop LIBERO rollout video

OpenPI predicts actions rather than video pixels. The default launcher now
runs those actions in the local LIBERO MuJoCo simulator and records the real
agent-view rollout:

```bash
cd /data/liuhongda/LightX2V
bash scripts/openpi/run_libero_i2va.sh
```

Default outputs:

```text
save_results/output_openpi_pi05_libero.mp4
save_results/output_openpi_pi05_libero.actions.npy
save_results/output_openpi_pi05_libero.metrics.json
```

Select another LIBERO task or output path with environment variables:

```bash
LIBERO_BENCHMARK=libero_goal \
LIBERO_TASK_ID=3 \
LIBERO_INIT_STATE_ID=0 \
OPENPI_SAVE_VIDEO_PATH=/absolute/output/rollout.mp4 \
OPENPI_SAVE_ACTION_PATH=/absolute/output/rollout.actions.npy \
OPENPI_SAVE_METRICS_PATH=/absolute/output/rollout.metrics.json \
bash scripts/openpi/run_libero_i2va.sh
```

The rollout performs 10 dummy stabilization steps, replans every 5 policy
steps, records the correctly oriented `agentview_image` at 10 FPS, and stops
when LIBERO reports BDDL task success or the suite-specific step cap is hit.

## 4. Run a sample by LIBERO benchmark and task ID

Use the task launcher when selecting examples from the local LIBERO checkout:

```text
/data/liuhongda/openpi/third_party/libero/libero/libero/bddl_files
/data/liuhongda/openpi/third_party/libero/libero/libero/init_files
```

The first argument is the benchmark (task suite), the second is the zero-based
task ID, and the optional third argument is the zero-based initialization-state
ID (default: `0`). Every bundled `.pruned_init` contains 50 states, so valid
initialization-state IDs are `0-49`:

```bash
cd /data/liuhongda/LightX2V
bash scripts/openpi/run_libero_task_i2va.sh libero_goal 3 0
```

Supported benchmark and task-ID ranges:

| Benchmark | Task IDs |
| --- | ---: |
| `libero_spatial` | 0-9 |
| `libero_object` | 0-9 |
| `libero_goal` | 0-9 |
| `libero_10` | 0-9 |
| `libero_90` | 0-89 |

LIBERO's benchmark map (rather than alphabetical filename order) resolves the
selected pair to its exact `.bddl` and `.pruned_init` files. For the example
above, results are organized as:

```text
save_results/openpi_libero_tasks/
└── libero_goal_task_3/
    ├── init_state_0.mp4
    ├── init_state_0.actions.npy
    └── init_state_0.metrics.json
```

The metrics JSON records the task name and description, success result, exact
BDDL/init-state file paths, selected initialization-state ID, and rollout
statistics. Running a different initialization state of the same task places
another set of `init_state_N.*` files in the same task directory.

Change the result root or cap the rollout during a smoke test with:

```bash
OPENPI_LIBERO_RESULT_ROOT=/absolute/output/root \
OPENPI_MAX_STEPS=20 \
bash scripts/openpi/run_libero_task_i2va.sh libero_10 5 2
```

Use `bash scripts/openpi/run_libero_task_i2va.sh --help` to display the command
summary. Other model/runtime overrides accepted by `run_libero_i2va.sh` remain
available, including `OPENPI_SEED`, `OPENPI_RENDER_SIZE`, and
`OPENPI_VIDEO_FPS`.

## 5. Static image/state-to-action smoke inference

Place the two RGB frames and state in one input directory:

```text
INPUT_DIR/
├── agentview_image.png
├── wrist_image.png
└── state.npy                 # float32 shape (8,)
```

A reproducible sample can be extracted from the already-downloaded raw
LIBERO dataset with:

```bash
python scripts/openpi/prepare_libero_sample.py
```

Run:

```bash
OPENPI_RUN_MODE=single_observation \
OPENPI_IMAGE_PATH=/absolute/path/to/INPUT_DIR \
OPENPI_TASK_DESCRIPTION="pick up the black bowl and place it on the plate" \
OPENPI_SAVE_ACTION_PATH=/absolute/path/to/actions.npy \
bash scripts/openpi/run_libero_i2va.sh
```

The result is a float32 NumPy array with shape `(10, 7)` at the exact
`OPENPI_SAVE_ACTION_PATH`.

The default closed-loop rollout was exercised on an H200. The generated
artifacts are:

```text
/data/liuhongda/LightX2V/save_results/output_openpi_pi05_libero.mp4
/data/liuhongda/LightX2V/save_results/output_openpi_pi05_libero.actions.npy
/data/liuhongda/LightX2V/save_results/output_openpi_pi05_libero.metrics.json
/data/liuhongda/openpi_data/results/pi05_libero_pytorch_parity.json
```

The parity report uses identical LIBERO input and fixed flow noise for upstream
OpenPI PyTorch and the LightX2V-native network.

## 6. Closed-loop ROS LIBERO rollout

```bash
LIBERO_BENCHMARK=libero_10 \
LIBERO_TASK_ID=5 \
LIBERO_INIT_STATE_ID=0 \
bash scripts/openpi/run_libero_ros_i2va.sh
```

The ROS script builds the existing LightX2V workspace, starts the shared local
LIBERO simulator, and starts `openpi_node`. The simulator already performs the
official 180-degree image rotation; the runner and policy do not flip again.

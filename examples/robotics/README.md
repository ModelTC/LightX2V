# Robotics Evaluation

Run LIBERO, LIBERO-Plus, and RoboTwin 2.0 with LightX2V policies. The benchmark
scripts manage task discovery, GPU workers, episodes, and success-rate summaries.
They reuse the simulator adapters in `lightx2v_ros` without requiring ROS or colcon.
Run the commands below from the LightX2V repository root.

## Environment

Use Linux with an NVIDIA GPU and a CUDA-compatible driver. The shared environment
uses Python 3.10, PyTorch 2.7.1 / CUDA 12.8, MuJoCo 3.3.2, and SAPIEN 3.0.0b1.
It includes LightX2V and all three benchmarks' Python dependencies; cuRobo is
built separately for RoboTwin. LIBERO and LIBERO-Plus workers select their own
source trees and write separate runtime configurations.

On Ubuntu, install the system libraries first (or use a container with them):

```bash
sudo apt-get update
sudo apt-get install -y build-essential git git-lfs unzip ffmpeg \
  libegl1 libgl1 libglvnd0 libvulkan1 libglib2.0-0 \
  libexpat1 libfontconfig1 libmagickwand-dev

conda env create -f scripts/bench/robotics/environment.yml
conda activate lightx2v-robotics
```

The environment file installs this checkout as an editable package. It pins the key model
and simulator dependencies, rather than exporting machine-specific paths or a
complete package lock. In containers, expose NVIDIA graphics capabilities as
well as compute (for example, `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`).
EGL is used for LIBERO rendering; RoboTwin also requires a working Vulkan driver.

## Benchmark sources and assets

Initialize the repository-pinned benchmark versions:

```bash
git submodule update --init \
  lightx2v_ros/src/simulator/simulator/libero_node/LIBERO \
  lightx2v_ros/src/simulator/simulator/libero_node/LIBERO-plus \
  lightx2v_ros/src/simulator/simulator/robotwin_node/RoboTwin

export LIBERO_SOURCE_DIR="$PWD/lightx2v_ros/src/simulator/simulator/libero_node/LIBERO"
export LIBERO_PLUS_SOURCE_DIR="$PWD/lightx2v_ros/src/simulator/simulator/libero_node/LIBERO-plus"
export ROBOTWIN_ROOT="$PWD/lightx2v_ros/src/simulator/simulator/robotwin_node/RoboTwin"
```

These are also the default source locations. To reuse an existing installation,
set the same variables to its absolute paths instead. The benchmark loaders
select the source directly; installing two competing `libero` packages is unnecessary.
Evaluation needs simulator assets, task descriptions, and initial states, not
the training demonstration datasets.

### LIBERO and LIBERO-Plus

LIBERO includes its task definitions and initial states. For LIBERO-Plus,
download the additional [assets](https://huggingface.co/datasets/Sylvest/LIBERO-plus):

```bash
huggingface-cli download Sylvest/LIBERO-plus assets.zip --repo-type dataset \
  --local-dir "$LIBERO_PLUS_SOURCE_DIR/libero/libero"
unzip "$LIBERO_PLUS_SOURCE_DIR/libero/libero/assets.zip" \
  -d "$LIBERO_PLUS_SOURCE_DIR/libero/libero"
```

The final directory must be `$LIBERO_PLUS_SOURCE_DIR/libero/libero/assets/`,
containing `new_objects/`, `textures/`, and the other asset directories. Some
archive versions contain extra parent directories; move the extracted `assets`
directory to that location. See the [LIBERO-Plus installation guide](https://github.com/sylvestf/LIBERO-plus#-installation).

### RoboTwin

Install a CUDA 12.8 toolkit (`nvcc`) to build cuRobo against this environment's
PyTorch. Build on the target GPU, or set `TORCH_CUDA_ARCH_LIST` for it:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git \
  "$ROBOTWIN_ROOT/envs/curobo"
MAX_JOBS=8 python -m pip install --no-build-isolation --no-deps \
  -e "$ROBOTWIN_ROOT/envs/curobo"

(cd "$ROBOTWIN_ROOT" && bash script/_download_assets.sh)
```

Apply the SAPIEN/MPLib compatibility fixes used by the pinned RoboTwin version
and FasterWAM's evaluation environment:

```bash
python - <<'PY'
from importlib.util import find_spec
from pathlib import Path

fixes = {
    "sapien": ("wrapper/urdf_loader.py", [
        ('open(urdf_file, "r")', 'open(urdf_file, "r", encoding="utf-8")'),
        ('open(srdf_file, "r")', 'open(srdf_file, "r", encoding="utf-8")'),
    ]),
    "mplib": ("planner.py", [
        ("if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
         "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:"),
    ]),
}
for package, (relative, replacements) in fixes.items():
    path = Path(next(iter(find_spec(package).submodule_search_locations))) / relative
    text = path.read_text()
    for old, new in replacements:
        text = text.replace(old, new)
    path.write_text(text)
PY
```

RoboTwin uses expert planning to select solvable evaluation seeds, so cuRobo is
required even when only evaluating a learned policy. The current upstream
RoboTwin main branch uses XPolicyLab; these commands target this repository's
pinned RoboTwin 2.0 version. See the [upstream installation and asset guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)
for system-specific Vulkan setup.

## Model preparation

Set the base model, policy checkpoint, matching normalization statistics, and
GPU visibility. The following examples use FastWAM's native 20-step policy:

```bash
export WAN_MODEL_PATH=/path/to/Wan2.2-TI2V-5B
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHON_BIN="$(command -v python)"
```

`CKPT_PATH` is the full policy checkpoint (merge/export a distilled adapter first
if necessary). `DATASET_STATS_PATH` must match its benchmark and training data.
LIBERO-Plus uses the LIBERO policy and statistics. Download weights from the
corresponding model release; the benchmark scripts do not convert checkpoints.

`model=fastwam` and `model=realtimewam` select the existing native policy adapters.
`CONFIG_JSON` selects the model architecture and inference settings. The benchmark
schedule is independent of that choice:

| Policy | LIBERO / LIBERO-Plus config | RoboTwin config | Inference steps |
| --- | --- | --- | --- |
| FastWAM | `configs/fastwam/libero_i2va.json` | `configs/fastwam/robotwin_i2va.json` | 20 |
| RealtimeWAM on FastWAM | `configs/realtimewam/libero_fastwam_i2va.json` | `configs/realtimewam/robotwin_fastwam_i2va.json` | 1 |
| RealtimeWAM on FasterWAM | `configs/realtimewam/libero_fasterwam_i2va.json` | `configs/realtimewam/robotwin_fasterwam_i2va.json` | 1 |

The one-step profiles require the corresponding distilled checkpoints. For a
different policy implementation, the evaluator also accepts
`model.factory=your_module:factory`; its interface is defined in
[`sim/evaluator.py`](../../lightx2v_ros/src/simulator/simulator/sim/evaluator.py).

## Evaluation

### LIBERO

Run Spatial, Object, Goal, and Long (`libero_10`): 40 tasks, 50 trials per task.
The example uses WAM's 30 settling steps and executes 10 actions per plan.

```bash
CKPT_PATH=/path/to/libero/policy.pt \
DATASET_STATS_PATH=/path/to/libero/dataset_stats.json \
CONFIG_JSON=configs/fastwam/libero_i2va.json \
OUT=evaluate_results/libero/fastwam \
bash scripts/bench/robotics/run_libero.sh model=fastwam seed=42 \
  MULTIRUN.num_gpus=8 EVALUATION.num_trials=50 \
  EVALUATION.num_steps_wait=30 EVALUATION.replan_steps=10
```

### LIBERO-Plus

Use the same LIBERO checkpoint with the Plus source and assets. Following the
[official protocol](https://github.com/sylvestf/LIBERO-plus#-evaluation), run one
trial per perturbation task. Results include the seven perturbation categories.

```bash
CKPT_PATH=/path/to/libero/policy.pt \
DATASET_STATS_PATH=/path/to/libero/dataset_stats.json \
CONFIG_JSON=configs/fastwam/libero_i2va.json \
OUT=evaluate_results/libero_plus/fastwam \
bash scripts/bench/robotics/run_libero_plus.sh model=fastwam seed=42 \
  MULTIRUN.num_gpus=8 EVALUATION.num_trials=1 \
  EVALUATION.num_steps_wait=30 EVALUATION.replan_steps=10
```

### RoboTwin 2.0

Run all tasks in both clean and randomized settings, with 100 episodes per task
and setting. This FastWAM example uses unseen instructions and replan=24:

```bash
CKPT_PATH=/path/to/robotwin/policy.pt \
DATASET_STATS_PATH=/path/to/robotwin/dataset_stats.json \
CONFIG_JSON=configs/fastwam/robotwin_i2va.json \
OUT=evaluate_results/robotwin/fastwam \
bash scripts/bench/robotics/run_robotwin.sh model=fastwam seed=42 \
  MULTIRUN.num_gpus=8 EVALUATION.eval_num_episodes=100 \
  EVALUATION.instruction_type=unseen EVALUATION.replan_steps=24
```

For RealtimeWAM, change `model=fastwam` to `model=realtimewam`, select the matching
profile in the table above, and provide its distilled weights. For the FasterWAM
RoboTwin teacher-consistency future evaluation, use replan=28 instead of 24.
Overrides take precedence over the JSON profile, so update the explicit
`EVALUATION.replan_steps` argument as well.

### Scheduling and small runs

| Setting | LIBERO / LIBERO-Plus | RoboTwin |
| --- | --- | --- |
| `MULTIRUN.max_tasks_per_gpu` | 1 | 2 |
| `MULTIRUN.chunk_size` | 2 | 1 |
| Trial count | `EVALUATION.num_trials` | `EVALUATION.eval_num_episodes` |
| Task selection | `MULTIRUN.task_suite_names`, `MULTIRUN.task_ids` | `MULTIRUN.task_names`, `MULTIRUN.phases` |

`chunk_size` is the number of tasks a worker processes **sequentially** while
keeping its model loaded; it is not an inference batch size. For example, 40
LIBERO tasks form 20 jobs with chunk_size=2, allowing all eight GPUs to work.
For a single GPU, set both `CUDA_VISIBLE_DEVICES=0` and `MULTIRUN.num_gpus=1`.

Append `dry_run=true` to a command to discover tasks and write the resolved
configuration without loading the policy or running episodes. Simulator imports
and task assets are still required. For a real one-episode smoke, use the same
command with a separate `OUT` directory and these overrides:

```bash
# LIBERO / LIBERO-Plus
MULTIRUN.num_gpus=1 MULTIRUN.max_tasks_per_gpu=1 \
  'MULTIRUN.task_suite_names=[libero_spatial]' 'MULTIRUN.task_ids=[0]' \
  EVALUATION.num_trials=1

# RoboTwin
MULTIRUN.num_gpus=1 MULTIRUN.max_tasks_per_gpu=1 \
  'MULTIRUN.task_names=[adjust_bottle]' 'MULTIRUN.phases=[clean]' \
  EVALUATION.eval_num_episodes=1
```

These lines are arguments to append to the evaluation commands, not standalone
shell commands. Settling belongs to the evaluator: its default is 5 steps for
LIBERO/Plus; the examples explicitly select 30 for the WAM checkpoints.

## Results and resuming

Each run writes to `OUT` (or a new directory under `evaluate_results/`):

- `manifest.json`: resolved model/evaluation configuration and task list.
- `summary.json`: progress, overall success rate, suite/category/phase scores,
  and worker errors. `complete=true` indicates the full run finished successfully.
- `tasks/*.json`: per-task results with episode outcomes and actual seeds.
- `manager.log` and `jobs/*.log`: scheduler and worker logs.

Rates are successful episodes divided by completed episodes; partial results
are not final scores. The overall score is weighted by episode count, rather
than an unweighted average of category percentages.

To resume, use the **same configuration and OUT** with `EVALUATION.resume=true`.
Completed tasks are skipped; an interrupted task restarts from its first episode.
For RoboTwin comparisons, optionally add `EVALUATION.reuse_seed_cache=true` and
`EVALUATION.seed_cache_dir=/path/to/seeds`. Cached seeds are rechecked by the expert
and override the initial seed sequence; use separate copies when running models
concurrently and keep the original cache for reproducibility.

## References

- [FasterWAM environment and evaluation guide](https://github.com/hustvl/FasterWAM#evaluation)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus)
- [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)

# Robotics Evaluation

Evaluate LightX2V policies on LIBERO, LIBERO-Plus, and RoboTwin 2.0 using
`scripts/bench/robotics/run_*.sh`. Run all commands from the repository root;
ROS and colcon are not required.

## 1. Install the environment

Use Linux with an NVIDIA driver supporting CUDA 12.8. Both environments include
LightX2V, Python 3.10, and PyTorch 2.7.1. Install only the environment you need:

| Environment | Benchmarks | Simulator |
| --- | --- | --- |
| `lightx2v-libero` | LIBERO, LIBERO-Plus | MuJoCo 3.3.2 / robosuite 1.4.0 |
| `lightx2v-robotwin` | RoboTwin 2.0 | SAPIEN 3.0.0b1 / MPLib 0.2.1 / cuRobo |

```bash
# LIBERO and LIBERO-Plus
conda env create -f scripts/bench/robotics/environment_libero.yml
conda activate lightx2v-libero

# RoboTwin (separate environment)
conda env create -f scripts/bench/robotics/environment_robotwin.yml
conda activate lightx2v-robotwin
```

Rendering requires NVIDIA EGL for LIBERO and Vulkan for RoboTwin. Containers
must expose graphics, e.g. `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`.
For LIBERO-Plus, also install ImageMagick (`sudo apt-get install libmagickwand-dev`).

## 2. Prepare benchmark resources

### LIBERO / LIBERO-Plus

```bash
conda activate lightx2v-libero
git submodule update --init \
  lightx2v_ros/src/simulator/simulator/libero_node/LIBERO \
  lightx2v_ros/src/simulator/simulator/libero_node/LIBERO-plus

# Extra assets needed only for LIBERO-Plus
export LIBERO_PLUS_SOURCE_DIR="$PWD/lightx2v_ros/src/simulator/simulator/libero_node/LIBERO-plus"
huggingface-cli download Sylvest/LIBERO-plus assets.zip --repo-type dataset \
  --local-dir "$LIBERO_PLUS_SOURCE_DIR/libero/libero"
unzip "$LIBERO_PLUS_SOURCE_DIR/libero/libero/assets.zip" \
  -d "$LIBERO_PLUS_SOURCE_DIR/libero/libero"
```

Ensure the extracted assets are under `LIBERO-plus/libero/libero/assets/`;
move them there if the archive includes extra parent directories.
Both benchmarks share the environment; the launcher selects the correct source.
Training demonstration datasets are not needed for evaluation.

### RoboTwin

Use the pinned submodule, whose interface differs from current upstream main.
Install the CUDA 12.8 toolkit (`nvcc`) before building cuRobo on the target GPU:

```bash
conda activate lightx2v-robotwin
git submodule update --init lightx2v_ros/src/simulator/simulator/robotwin_node/RoboTwin
export ROBOTWIN_ROOT="$PWD/lightx2v_ros/src/simulator/simulator/robotwin_node/RoboTwin"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$ROBOTWIN_ROOT/envs/curobo"
MAX_JOBS=8 python -m pip install --no-build-isolation --no-deps -e "$ROBOTWIN_ROOT/envs/curobo"
(cd "$ROBOTWIN_ROOT" && bash script/_download_assets.sh)

# Compatibility fixes from the pinned RoboTwin installation instructions
ROBOTWIN_SITE=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
sed -i 's/open(urdf_file, "r")/open(urdf_file, "r", encoding="utf-8")/; s/open(srdf_file, "r")/open(srdf_file, "r", encoding="utf-8")/' "$ROBOTWIN_SITE/sapien/wrapper/urdf_loader.py"
sed -i 's/if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:/if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:/' "$ROBOTWIN_SITE/mplib/planner.py"
```

cuRobo is required for expert seed validation. For driver and asset details, see
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO),
[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus#-installation), and
[RoboTwin](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).
To reuse existing benchmark checkouts, set `LIBERO_SOURCE_DIR`,
`LIBERO_PLUS_SOURCE_DIR`, or `ROBOTWIN_ROOT` to their absolute paths.

## 3. Run evaluation

Activate the matching environment above. Set the base model and visible GPUs:

```bash
export WAN_MODEL_PATH=/path/to/Wan2.2-TI2V-5B
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHON_BIN="$(command -v python)"
```

The examples use FastWAM's 20-step policy. Supply its full checkpoint and matching
`dataset_stats.json`; LIBERO-Plus reuses the LIBERO weights and statistics.

**LIBERO** — four suites, 40 tasks × 50 trials:

```bash
CKPT_PATH=/path/to/libero/policy.pt \
DATASET_STATS_PATH=/path/to/libero/dataset_stats.json \
CONFIG_JSON=configs/fastwam/libero_i2va.json OUT=evaluate_results/libero/fastwam \
bash scripts/bench/robotics/run_libero.sh model=fastwam seed=42 \
  EVALUATION.num_trials=50 EVALUATION.num_steps_wait=30 EVALUATION.replan_steps=10
```

**LIBERO-Plus** — one trial per perturbation task, following its official protocol:

```bash
CKPT_PATH=/path/to/libero/policy.pt \
DATASET_STATS_PATH=/path/to/libero/dataset_stats.json \
CONFIG_JSON=configs/fastwam/libero_i2va.json OUT=evaluate_results/libero_plus/fastwam \
bash scripts/bench/robotics/run_libero_plus.sh model=fastwam seed=42 \
  EVALUATION.num_trials=1 EVALUATION.num_steps_wait=30 EVALUATION.replan_steps=10
```

**RoboTwin** — 50 tasks × clean/random × 100 episodes, unseen instructions:

```bash
CKPT_PATH=/path/to/robotwin/policy.pt \
DATASET_STATS_PATH=/path/to/robotwin/dataset_stats.json \
CONFIG_JSON=configs/fastwam/robotwin_i2va.json OUT=evaluate_results/robotwin/fastwam \
bash scripts/bench/robotics/run_robotwin.sh model=fastwam seed=42 \
  EVALUATION.eval_num_episodes=100 EVALUATION.instruction_type=unseen EVALUATION.replan_steps=24
```

To evaluate **RealtimeWAM**, use `model=realtimewam`, matching distilled weights,
and one of these profiles under `configs/realtimewam/`:

| Backbone | LIBERO / Plus | RoboTwin | RoboTwin replan |
| --- | --- | --- | --- |
| FastWAM | `libero_fastwam_i2va.json` | `robotwin_fastwam_i2va.json` | 24 |
| FasterWAM | `libero_fasterwam_i2va.json` | `robotwin_fasterwam_i2va.json` | 28 |

Both profiles use one inference step. Update the explicit replan argument when
switching to FasterWAM; command-line values override the profile.

Defaults: 8 GPUs; LIBERO/Plus uses 1 worker/GPU and `chunk_size=2`, RoboTwin uses
2 workers/GPU and `chunk_size=1`. A chunk is a sequence of tasks for one worker.
Append overrides to any command: `MULTIRUN.num_gpus=1` for a single visible GPU,
`dry_run=true` to inspect the task schedule, or the following for a one-episode
LIBERO smoke (use a separate `OUT`):

```text
MULTIRUN.num_gpus=1 MULTIRUN.task_suite_names=[libero_spatial] MULTIRUN.task_ids=[0] EVALUATION.num_trials=1
```

## 4. Read results or resume

Results are written to `OUT`: `summary.json` contains overall and per-suite,
perturbation-category, or clean/random scores; `manifest.json` records the resolved
configuration. Inspect `manager.log` and `jobs/*.log` for errors, and `tasks/*.json`
for individual episodes. Final results have `complete=true`.

Resume with the same configuration and `OUT`, appending `EVALUATION.resume=true`.
Completed tasks are skipped; interrupted tasks restart from their first episode.
For RoboTwin seed reuse, also set `EVALUATION.reuse_seed_cache=true` and
`EVALUATION.seed_cache_dir=/path/to/seeds`; use separate cache copies for concurrent runs.

# Robotics Evaluation

Evaluate LightX2V policies on LIBERO, LIBERO-Plus, and RoboTwin 2.0 using
`scripts/bench/robotics/run_*.sh`. Run all commands from the repository root;
ROS and colcon are not required.

## 1. Install the environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use Linux
with an NVIDIA driver supporting CUDA 12.8. System dependencies: Git, a C++
compiler, FFmpeg, unzip, and NVIDIA EGL/Vulkan libraries. LIBERO-Plus also needs
ImageMagick (`sudo apt-get install libmagickwand-dev`). In containers, expose
`NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`.

The installer creates independent Python 3.10 environments with LightX2V and
PyTorch 2.7.1, and initializes the corresponding benchmark submodules:

Python itself is stored in `.venvs/python/`. When sharing the repository between
machines, mount it at the same absolute path; GPU drivers remain machine-specific.

```bash
# LIBERO and LIBERO-Plus share this environment
bash scripts/bench/robotics/install_env.sh libero
source .venvs/libero/bin/activate

# RoboTwin: install CUDA Toolkit 12.8 first for the cuRobo build
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
bash scripts/bench/robotics/install_env.sh robotwin
source .venvs/robotwin/bin/activate
```

Install only the environment you need. RoboTwin setup builds cuRobo v0.7.8 and
applies the pinned benchmark's SAPIEN/MPLib fixes. Build on the target GPU, or
set `TORCH_CUDA_ARCH_LIST` for it. Dependencies are listed in
[`requirements_libero.txt`](../../scripts/bench/robotics/requirements_libero.txt)
and [`requirements_robotwin.txt`](../../scripts/bench/robotics/requirements_robotwin.txt).

## 2. Prepare benchmark resources

**LIBERO** includes task definitions and initial states in its submodule.
**LIBERO-Plus** additionally requires these assets:

```bash
source .venvs/libero/bin/activate
export LIBERO_PLUS_SOURCE_DIR="$PWD/lightx2v_ros/src/simulator/simulator/libero_node/LIBERO-plus"
huggingface-cli download Sylvest/LIBERO-plus assets.zip --repo-type dataset \
  --local-dir "$LIBERO_PLUS_SOURCE_DIR/libero/libero"
unzip "$LIBERO_PLUS_SOURCE_DIR/libero/libero/assets.zip" \
  -d "$LIBERO_PLUS_SOURCE_DIR/libero/libero"
```

Ensure the extracted assets are under `LIBERO-plus/libero/libero/assets/`;
move them there if the archive includes extra parent directories.

**RoboTwin** assets:

```bash
source .venvs/robotwin/bin/activate
export ROBOTWIN_ROOT="$PWD/lightx2v_ros/src/simulator/simulator/robotwin_node/RoboTwin"
(cd "$ROBOTWIN_ROOT" && bash script/_download_assets.sh)
```

Use the repository-pinned benchmark versions. To reuse existing checkouts, set
`LIBERO_SOURCE_DIR`, `LIBERO_PLUS_SOURCE_DIR`, or `ROBOTWIN_ROOT` to their absolute
paths. Training demonstration datasets are not needed for evaluation. See
[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus#-installation) and
[RoboTwin](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)
for asset and driver details.

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
For a single GPU, set `CUDA_VISIBLE_DEVICES=0` and append `MULTIRUN.num_gpus=1`.
Append `dry_run=true` to inspect the task schedule without running episodes.

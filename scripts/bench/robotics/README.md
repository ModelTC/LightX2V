# Robotics evaluation and ROS RealtimeWAM

This ports the simulator/evaluation support from [ModelTC/LightX2V#1562](https://github.com/ModelTC/LightX2V/pull/1562), using this repository's existing FastWAM and RealtimeWAM implementations. One CLI handles LIBERO, LIBERO-plus and RoboTwin; each simulator has one benchmark module alongside its existing ROS environment. Model code and inference kernels are unchanged.

## Offline evaluation

Activate an environment containing both LightX2V and the selected simulator's dependencies. ROS and a colcon build are not required. Initialize the relevant submodule and install its dependencies/assets following the upstream simulator instructions:

```bash
git submodule update --init --recursive lightx2v_ros/src/simulator/simulator/libero_node/LIBERO
# For LIBERO-plus, initialize libero_node/LIBERO-plus instead.
# For RoboTwin, initialize robotwin_node/RoboTwin and download its assets.
```

Run from the repository root. `CKPT_PATH` must be a dense or already merged checkpoint supported by LV's loader; the separate FasterWAM/LoRA loader from #1562 is not part of this port.

```bash
export WAN_MODEL_PATH=/absolute/path/to/Wan2.2-TI2V-5B
export CKPT_PATH=/absolute/path/to/robotwin_consistency_ts10_ema_step30000.pt
export DATASET_STATS_PATH=/absolute/path/to/robotwin_dataset_stats.json
export OUT=/absolute/path/to/results/robotwin-realtimewam
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python scripts/bench/robotics/run.py robotwin \
  model=realtimewam EVALUATION.num_inference_steps=1 \
  EVALUATION.replan_steps=24 EVALUATION.eval_num_episodes=100 \
  EVALUATION.reuse_seed_cache=true \
  EVALUATION.seed_cache_dir=/absolute/path/to/validated_seeds \
  MULTIRUN.num_gpus=8 MULTIRUN.max_tasks_per_gpu=2 seed=42
```

The seed directory uses `clean/<task>_seed.json` and `random/<task>_seed.json`, containing increasing lists of integers. Cached seeds override the initial scene seed and are revalidated by the real cuRobo expert. Use an isolated copy when comparing runs, since failed seeds are removed and new validated seeds are saved. Missing cuRobo fails evaluation rather than enabling the interactive fallback planner. Planner asset paths are relocated inside the selected RoboTwin tree without changing its files.

For LIBERO, set checkpoint/stats/output to the LIBERO paths and run:

```bash
python scripts/bench/robotics/run.py libero EVALUATION.num_trials=50
python scripts/bench/robotics/run.py libero_plus EVALUATION.num_trials=1
```

Use a separate output directory and process for each benchmark. Optional source overrides are `EVALUATION.libero_root=/path/to/LIBERO[-plus]` and `EVALUATION.robotwin_root=/path/to/RoboTwin`; defaults point to the repository submodules. LIBERO-plus uses its own task classification and suite-resolved initial states.

`model=fastwam` selects LV's original FastWAM policy. `config_json=/path/to/profile.json` reuses an existing model profile. RealtimeWAM defaults to one step with CUDA Graph and Triton acceleration; FastWAM follows its profile. Explicit `key=value` overrides take precedence. The resolved policy configuration is saved in `manifest.json`.

| Protocol | LIBERO / LIBERO-plus | RoboTwin |
| --- | --- | --- |
| Trials | 50 / 1 per task | 100 per task and clean/random phase |
| Actions per replan | Existing profile (10 by default) | **24**, independently of the interactive profile's 8 |
| Settling | 5 steps, open gripper | Expert validation, then reset the accepted seed |
| Observations | LIBERO orientation and 8-D state | 3 RGB cameras and 14-D qpos; skip unused observations within a chunk |
| Success | Simulator's success flag | `eval_success` after `take_action(qpos)`, without an extra `check_success()` |
| Limit | 400 steps; 700 for libero_10/libero_90 | Task's `step_lim` |

`EVALUATION.max_steps` explicitly overrides the limit and is recorded in results. Python/NumPy/Torch RNGs are seeded per task; selected RoboTwin instructions and accepted scene seeds are recorded per episode. Historical runs without a fixed Python random seed can select different instructions even with the same scene seed.

Use `dry_run=true` to discover tasks and save a manifest without model inference. Select a small smoke with `MULTIRUN.task_names=[adjust_bottle] MULTIRUN.phases=[clean] EVALUATION.eval_num_episodes=2` for RoboTwin, or `MULTIRUN.task_suite_names=[libero_spatial] MULTIRUN.task_ids=[0] EVALUATION.num_trials=1` for LIBERO.

Workers produce `jobs/*.log`, per-task JSON and `summary.json`. Success rates use total successes divided by completed trials. `EVALUATION.resume=true` reuses completed tasks only when configuration, checkpoint/stat metadata and source fingerprint match. Interrupted tasks restart from their first trial. Worker failures stop the manager, and normal completion exits without an infinite sleep.

## ROS

The existing `fastwam_node` accepts `model_cls:=realtimewam`; its default remains `fastwam`. Build/source the existing ROS workspace as usual, then:

```bash
ros2 run inference fastwam_node --ros-args \
  -p model_cls:=realtimewam -p env:=libero \
  -p config_json:=/absolute/path/to/libero_profile.json \
  -p model_path:=/absolute/path/to/Wan2.2-TI2V-5B
```

The JSON must contain absolute `adapter_model_path` and `dataset_stats_path` values; start from `configs/realtimewam/libero_i2va.json`. For RoboTwin use its existing FastWAM profile with `policy_profile=robotwin`, set the desired `action_infer_steps` and `actions_per_plan`, and enable `cuda_graph`, `triton_ops` and `layer_norm_type=Triton` for accelerated RealtimeWAM. The same topics, action queue and episode-reset handling serve both policy classes.

## Validation

```bash
python -m unittest discover -s tests -p test_robotics_benchmark.py -v
```

These CPU checks cover config precedence, ROS policy selection, observation orientation, settling/reset, action validation, 24-action replanning, success/observation boundaries, seed caching, micro aggregation, dry-run/resume and error cleanup. Full policy accuracy still requires the matching checkpoints and simulator assets.

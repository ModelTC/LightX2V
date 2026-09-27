# RealtimeWAM benchmark inference

All entrypoints, model inference, LoRA merging and rollout code run from this
repository. Neither FasterWAM nor MeanFlowWAM is imported. LIBERO, LIBERO-plus and RoboTwin
default to pinned repository submodules, matching the existing ROS LIBERO layout;
no external benchmark source directory or ROS is required. Simulator libraries,
benchmark assets and pretrained model files are still required.

## Model contract

RealtimeWAM here means **teacher-supervised few-step action distillation**:
`model.sampler=teacher_flow` uses the teacher's velocity parameterization. At one
step it computes `action_noise - predicted_velocity`. This is **not** the
`consistency_baseline` c_skip/c_out parameterization. Unsupported samplers fail.

- `base_ckpt`: original native checkpoint containing `mot` and `proprio_encoder`.
- `lora_path`: exported PEFT directory (`action/`, optionally `video/`) or training
  checkpoint directory with `config.yaml` and `ema_action.pt`/`student_action.pt`.
- `model.lora_weights=ema` (default) selects training-checkpoint EMA weights.
  Exported PEFT adapters already encode their selected weights.
- Alternatively `ckpt=/merged.pt` loads an already merged model. Do not combine
  this with `base_ckpt` or `lora_path`. Original files are never overwritten.
- `model.model_path`: Wan2.2 base components, including `Wan2.2_VAE.pth`,
  `models_t5_umt5-xxl-enc-bf16.pth`, and `google/umt5-xxl`.
- `model.backbone=fastwam`: dense action transformer, no KV fusion.
- `model.backbone=fasterwam`: sparse condition layers `[0,4,...,28]` with trained
  interval-weighted video KV fusion. Checkpoint structure is checked on load.
- `EVALUATION.action_infer_mode=first_frame`: observed-frame cache.
- `EVALUATION.action_infer_mode=one_pass_future_cache`: observed frame at t=0,
  noisy future latents at t=1000, first-frame-causal video attention; prefill once
  and reuse within action denoising. Default `model.num_video_frames=9` matches
  `(33 - 1) / 4 + 1` in the referenced training/evaluation configs.

Architecture and conditioning must match training. A successful load alone
does not establish that a different conditioning mode is appropriate.

## Runtime

Initialize the benchmark submodules once after cloning:

```bash
git submodule update --init --recursive \
  lightx2v_ros/src/simulator/simulator/libero_node/LIBERO \
  lightx2v_ros/src/simulator/simulator/libero_node/LIBERO-plus \
  lightx2v_ros/src/simulator/simulator/robotwin_node/RoboTwin
```

LIBERO-plus is pinned to `4976dc30028e805ff8094b55501d532c48fec182`, the version
used for the existing-weight smoke. Download any benchmark assets not shipped
in git according to that benchmark's instructions; initialization alone does
not install simulator dependencies or download every asset.
In particular, LIBERO-plus publishes `assets.zip` separately (see its bundled
README). Extract it so that scenes live under
`lightx2v_ros/src/simulator/simulator/libero_node/LIBERO-plus/libero/libero/assets/`.
These large assets are ignored by the upstream submodule, not stored in the
parent repository's Git objects. Keep actual asset files there, not a symlink
to another machine-specific benchmark installation.

`EVALUATION.libero_root` is optional. Without an override the entrypoint selects
`libero_node/LIBERO` or `libero_node/LIBERO-plus` within this repository.
`LIBERO_SOURCE_DIR`/`LIBERO_PLUS_SOURCE_DIR` remain optional environment overrides;
unset them to use the repository defaults. An explicit CLI path takes precedence.

Use an environment with this repository's LightX2V inference dependencies.
LIBERO/Plus additionally require their robosuite, MuJoCo, BDDL dependencies;
RoboTwin requires SAPIEN, its assets, motion planners (including curobo), and
working NVIDIA Vulkan/GL libraries. Keep incompatible simulator environments
separate. Optional acceleration import warnings do not imply SDPA is unavailable.

The CLI uses strict OmegaConf `key=value` overrides (Hydra-style syntax), not a
Hydra launcher. Unknown fields are rejected. `--help` prints the entrypoint
usage; `experiments/configs/eval.yaml` lists supported fields. `model.model_id`,
`redirect_common_files`, training scheduler options, and arbitrary legacy task
configs are not silently accepted.

## LIBERO-plus

```bash
cd /path/to/LightX2V_realtimewam-infer
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python -u experiments/libero_plus/run_libero_plus_manager.py \
  model=realtimewam model.backbone=fasterwam \
  base_ckpt=/path/to/original_fasterwam_libero.pt \
  lora_path=/path/to/teacher_distilled_lora \
  model.model_path=/path/to/Wan2.2-TI2V-5B \
  EVALUATION.dataset_stats_path=/path/to/libero/dataset_stats.json \
  EVALUATION.num_trials=1 EVALUATION.num_inference_steps=1 \
  EVALUATION.action_infer_mode=one_pass_future_cache \
  EVALUATION.sigma_shift=5.0 EVALUATION.replan_steps=10 \
  MULTIRUN.num_gpus=8 MULTIRUN.max_tasks_per_gpu=2 \
  MULTIRUN.chunk_size=20 MULTIRUN.task_sample_ratio=1.0 \
  seed=42 EVALUATION.output_dir=/path/to/new_results
```

## LIBERO

Use `experiments/libero/run_libero_manager.py` and set `EVALUATION.num_trials=50`.
The entrypoint automatically selects ordinary LIBERO. LIBERO and Plus run in
separate processes to prevent their identically named Python packages colliding.

The referenced WAM protocol uses five settle actions `[0,0,0,0,0,0,-1]`, 400
policy steps for Spatial/Object/Goal and 700 for LIBERO-10/90. These protocol
settings are stored in the run manifest; changing them changes the experiment.
Initial states come from each suite's `get_task_init_states` (important for Plus),
and trial indices cycle only when trials exceed available states.

## RoboTwin

Activate the RoboTwin-compatible environment and its NVIDIA GL/Vulkan setup.
For the dense FastWAM teacher-distilled checkpoint in the original command:

```bash
python -u experiments/robotwin/run_robotwin_manager.py \
  model=realtimewam model.backbone=fastwam \
  base_ckpt=/path/to/original_robotwin.pt \
  lora_path=/path/to/teacher_distilled_checkpoint_or_lora \
  model.model_path=/path/to/Wan2.2-TI2V-5B \
  EVALUATION.dataset_stats_path=/path/to/robotwin_dataset_stats.json \
  EVALUATION.action_infer_mode=first_frame \
  EVALUATION.num_inference_steps=1 EVALUATION.replan_steps=8 \
  EVALUATION.eval_num_episodes=100 \
  MULTIRUN.num_gpus=8 MULTIRUN.max_tasks_per_gpu=2 \
  'MULTIRUN.phases=[clean,random]' seed=42 \
  EVALUATION.output_dir=/path/to/new_results
```

For a FasterWAM-backed teacher-distilled RoboTwin model, explicitly change
`model.backbone=fasterwam` and
`EVALUATION.action_infer_mode=one_pass_future_cache` and use its matching base
checkpoint/statistics. RoboTwin candidate seeds start at `100000 * (1 + seed)`;
the expert filters infeasible layouts. Accepted seeds and actual instructions
are recorded per episode. Dry-run experts/fallback instructions are forbidden.

The default simulator is the existing repository submodule at
`lightx2v_ros/src/simulator/simulator/robotwin_node/RoboTwin`, pinned to
`bf44be51cf5717a5595ce59447f2cf5263d2aa95`. No `robotwin_root` argument is required.
`ROBOTWIN_ROOT` and `EVALUATION.robotwin_root` remain optional overrides; unset
the environment variable to use the repository default. Populate the submodule's
`assets/` and `task_config/` using the official setup instructions, including
embodiment files and expert motion-planner dependencies.
The independent planner adapter rebases downloaded Curobo asset references to
the selected repository and writes per-worker configs under the output's
`runtime/robotwin_planner/`. It does not edit upstream code or downloaded YAML,
and rejects unresolved references to assets outside the selected benchmark.

### RoboTwin observation performance

The RoboTwin entrypoint defaults to `EVALUATION.skip_get_obs_within_replan=true`.
Every action still executes the original physics and success/termination checks;
the evaluator requests fresh camera observations only at action-replanning
boundaries. With `replan_steps=8`, the seven intermediate observation captures
are avoided. Reset observations remain fresh. The simulator's own internal
rendering and terminal-success capture are unchanged. Expert checks and seed
cache validation are not skipped.

Set `EVALUATION.skip_get_obs_within_replan=false` for the previous per-step
observation path. This flag is RoboTwin-only. Already-written job configs that
lack the flag retain their old behavior, including newly launched queued jobs;
use a new result directory for optimized runs. Do not resume old results with
changed settings/source. No numerical accuracy or wall-clock speedup is implied
by this optimization without a matched-seed simulator comparison.

### RoboTwin seed cache

`EVALUATION.reuse_seed_cache=true` preserves the legacy load/save behavior and
JSON-list format: `<seed_cache_dir>/clean/<task>_seed.json` and
`<seed_cache_dir>/random/<task>_seed.json`. The directory defaults to
`ROBOTWIN_SEED_DIR` or the repository-local RoboTwin `my_seeds/`; an explicit
`EVALUATION.seed_cache_dir` takes precedence. Existing seed lists can be copied
there without conversion. Cache contents are **expert-validated scene seeds**,
not seeds selected for policy success.

Cached candidates are tried first and still run the real expert; failed
candidates are rejected. Once the cache is exhausted, sequential seed search
continues. Accepted seeds are saved atomically after rollout-environment setup,
under a file lock, preserving unused cache entries and concurrent additions.
As before, a cache write failure warns without aborting the evaluation.
Per-episode results record `seed_cache_hit`, the cache path, and the actual seed.

As in the old launcher, cached seeds override the initial seed selection.
For an independent seed experiment, set `EVALUATION.reuse_seed_cache=false`
(default), or use an empty, dedicated cache directory. With reuse disabled,
existing cache files are neither read nor overwritten; actual seeds are still
recorded in `tasks/*.json`. Output resume remains controlled independently by
`EVALUATION.resume=true`.

## Smoke, diagnostics, and results

- Add `MULTIRUN.num_gpus=1 MULTIRUN.max_tasks_per_gpu=1`, a single task selector
  (`MULTIRUN.task_ids=[0] MULTIRUN.task_suite_names=[libero_spatial]` or
  `MULTIRUN.task_names=[adjust_bottle] MULTIRUN.phases=[clean]`), and one trial.
- `EVALUATION.max_steps=20` is a short integration smoke, **not benchmark
  accuracy**. Omit it to run the normal full episode horizon.
- `dry_run=true` validates paths and writes a task manifest without loading the
  model or starting rollouts. It still imports the benchmark to enumerate tasks.
- `python experiments/common/smoke_policy.py libero <model/weight overrides>`
  checks synthetic observation -> finite action output, without a simulator.
- `manifest.json` records effective config, task list and weight file identity
  (path, size, mtime). `jobs/*.log` contains the original worker traceback.
- `tasks/*.json` contains each episode's seed, outcome and timing.
- `summary.json` reports successes/trials, categories, suites/phases and
  completeness. LIBERO-plus Overall is trial-weighted, not seven-category macro.
  Incomplete trials are not silently recorded as model failures or successes.
- Resume with identical config/assets/source and `EVALUATION.resume=true`; complete
  tasks are skipped, incomplete tasks restart. A lock prevents concurrent
  managers from writing to the same output. Use a new output for changed configs.
- Worker GPU IDs respect `CUDA_VISIBLE_DEVICES`, including remapped indices.
  Each worker sees only its assigned device as CUDA 0. Two workers/GPU means two
  model copies: reduce concurrency on smaller GPUs.

## Adding another model

Implement the `Policy` protocol in `experiments/common/interfaces.py`, then use
`model.factory=your_package:build_policy`. It receives the resolved config and
must return physical-unit `ActionChunk`s with an explicit action space. Image
preprocessing, state normalization, gripper conversion and recurrent state belong
to that adapter. No changes to benchmark rollout/success accounting are needed.

Run unit tests with `python -m unittest discover -s experiments/tests -v`.

See [SMOKE_TESTS.md](SMOKE_TESTS.md) for the existing-weight integration checks.

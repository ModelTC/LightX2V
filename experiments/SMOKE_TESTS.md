# Existing-weight smoke checks (2026-09-26)

Branch: `realtimewam-infer`, based on upstream/main `a4b8ce30`.
Worktree: `/mnt/miaohua/charles/codes/LightX2V_realtimewam-infer`.

All three checks loaded original weights plus teacher-distilled EMA LoRA,
performed native LightX2V inference, validated finite actions, and stepped the
real benchmark simulator. All managers exited 0 with no worker failures.

| Benchmark | Task | Policy steps | Model calls | Result directory under `evaluate_results/` |
| --- | --- | ---: | ---: | --- |
| LIBERO | Spatial task 0 | 20 | 2 | `smoke_libero_teacher_v2` |
| LIBERO-plus | Spatial task 0, Background Textures | 20 | 2 | `smoke_libero_plus_teacher_v3` |
| RoboTwin | adjust_bottle, clean | 20 | 3 | `smoke_robotwin_teacher_v4` |

These are deliberately truncated integration checks, **not accuracy results**.
None of the three truncated episodes completed its manipulation goal. Each
directory contains the effective `manifest.json`, `summary.json`, per-task
episode records, and worker logs. Earlier failed diagnostic runs are retained;
the table identifies the final passing runs.

## Repository-local benchmark rerun

After adding the LIBERO-plus submodule and defaulting both entrypoints to the
repository-local benchmarks, the same model + simulator checks were rerun with
`LIBERO_SOURCE_DIR` and `LIBERO_PLUS_SOURCE_DIR` unset and **without** passing
`EVALUATION.libero_root`:

| Benchmark | Result directory | Policy steps | Model calls | Worker errors |
| --- | --- | ---: | ---: | ---: |
| LIBERO | `smoke_libero_repo_local` | 20 | 2 | 0 |
| LIBERO-plus | `smoke_libero_plus_repo_local` | 20 | 2 | 0 |

Both managers exited 0. The manifests resolve benchmark paths beneath this
worktree's `lightx2v_ros/src/simulator/simulator/libero_node/`, not the external
benchmark directory. Original checkpoint, LoRA and base component paths remain
external user-supplied model inputs. These reruns remain truncated smoke checks,
not accuracy measurements or verification of every scene asset.

Full LIBERO-plus task discovery also passed: 10,030 unique suite/task keys with
nonempty, validated seven-category metadata. This validates the task catalog,
not successful execution of all 10,030 tasks.

### Repository-local RoboTwin and legacy seed cache

The pinned RoboTwin submodule (`bf44be51cf5717a5595ce59447f2cf5263d2aa95`)
was tested without `EVALUATION.robotwin_root`, with `ROBOTWIN_ROOT` and
`ROBOTWIN_SEED_DIR` unset. Both runs used the existing RoboTwin teacher weights
listed below, one trial, 20 steps, three model calls, and real expert checks:

| Result directory | Initial cache | Accepted environment seed | `seed_cache_hit` | Exit / errors |
| --- | --- | ---: | --- | --- |
| `smoke_robotwin_repo_local_cache_create` | empty | 4300001 | false | 0 / none |
| `smoke_robotwin_repo_local_cache_replay` | created by first run | 4300001 | true | 0 / none |

Both passed `EVALUATION.reuse_seed_cache=true` and shared
`EVALUATION.seed_cache_dir=<worktree>/evaluate_results/smoke_robotwin_repo_local_cache_create/seed_cache`.
The first run rejected unstable seed 4300000 before saving `[4300001]` to
`clean/adjust_bottle_seed.json`. The replay revalidated that candidate with the
expert; it did not bypass planning. Each result remains a truncated smoke,
not a successful full manipulation episode or benchmark accuracy result.

Generated Curobo configs under each run's `runtime/robotwin_planner/` point to
the repository-local URDF and collision-sphere assets, not the old external
MeanFlowWAM benchmark installation. The 100 preexisting clean/random cache files
were also copied into the submodule's default `my_seeds/` without overwriting
existing files, and compared equal to their original copies. Smoke runs used
their own cache directory and did not modify those historical caches.

## Weight inputs

LIBERO and LIBERO-plus:

```text
base_ckpt=/mnt/miaohua/charles/models/fasterwam_release/libero/step_021700.pt
lora_path=/mnt/miaohua/charles/codes/LightX2V_fastwam/exports/fasterwam_libero_teacher_consistency_future_ema_step30000/lora
EVALUATION.dataset_stats_path=/mnt/miaohua/charles/models/fasterwam_release/libero/dataset_stats.json
```

RoboTwin (also verifies loading EMA directly from a training checkpoint):

```text
base_ckpt=/mnt/miaohua/charles/models/fasterwam_release/robotwin/step_029355.pt
lora_path=/mnt/miaohua/charles/codes/LightX2V_fastwam/lightx2v_train/runs/fasterwam_robotwin_action_1step_consistency_ts10_teacher_future/checkpoint-000030000
EVALUATION.dataset_stats_path=/mnt/miaohua/charles/models/fasterwam_release/robotwin/dataset_stats.json
```

All three used `model.backbone=fasterwam`,
`EVALUATION.action_infer_mode=one_pass_future_cache`, one inference step,
sigma shift 5, seed 42, and
`model.model_path=/mnt/miaohua/charles/models/Wan2.2-TI2V-5B`.
Replan intervals were 10 for LIBERO/Plus and 8 for RoboTwin.
RoboTwin's expert rejected unstable candidate seed 4300000 and accepted 4300001;
this is recorded in its worker log and episode metadata.

## Environments and reproduction

The local `.eval_envs/libero` and `.eval_envs/robotwin` are isolated venv overlays
on the existing FastWAM and RoboTwin conda environments, respectively. Activate
the matching conda environment before using the overlay Python so native library
paths are preserved. They are machine-local, ignored by git, and not portable
environment archives. No original conda environment was modified.

Use the commands in README with the inputs above and these overrides:

```text
MULTIRUN.num_gpus=1 MULTIRUN.max_tasks_per_gpu=1 EVALUATION.max_steps=20
```

For LIBERO/Plus additionally select:

```text
EVALUATION.num_trials=1 MULTIRUN.task_ids=[0] MULTIRUN.task_suite_names=[libero_spatial]
```

For RoboTwin additionally select:

```text
EVALUATION.eval_num_episodes=1 MULTIRUN.task_names=[adjust_bottle] MULTIRUN.phases=[clean]
```

Use a fresh output directory. Full effective settings and asset paths are in
each run's manifest. LIBERO source directories were
`/mnt/miaohua/charles/codes/benchmark/LIBERO` and `LIBERO-plus`; RoboTwin assets
were `/mnt/afs_1/lvchengtao/code/wam/MeanFlowWAM/third_party/RoboTwin`.
The latter supplies only benchmark assets/code, not external WAM inference.

## Additional checks and limits

- Seventeen unit tests pass, covering LoRA merging, BF16 rounding, conditioning masks,
  action validation, reset/replanning, aggregation, config and GPU mapping,
  custom model options, and the RoboTwin step return contract.
- Repository-local path defaults, override priority, initialization hints, and
  reloading LIBERO's namespace package are covered by the additional tests.
- RoboTwin tests also cover default submodule paths, portable planner asset
  references, legacy seed-cache validation, real expert rechecks on cache hits,
  clean/random separation, disabled-cache behavior, preserving concurrent
  additions, and warning rather than aborting on optional cache write failures.
- Four real merged LoRA target tensors exactly match the existing PEFT-merged
  EMA checkpoint (`max_abs=0`); this is not a full reference-policy parity test.
- Python compilation, Ruff checks and git whitespace checks pass.
- These checks do not establish formal benchmark accuracy, dense-backbone
  numerical parity, all perturbation categories, or 8-GPU x 2-worker load behavior.

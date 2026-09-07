# RoboTwin distilled evaluation

Run from `/mnt/miaohua/charles/codes/LightX2V_fastwam`. Each command uses
eight GPUs with two task processes per GPU. Run the two variants separately.

```bash
OUT=/mnt/miaohua/charles/codes/LightX2V_fastwam/runs/robotwin_eval/dmd_$(date +%Y%m%d_%H%M%S) \
  bash lightx2v_train/scripts/eval_robotwin_distilled_1step.sh dmd
```

```bash
OUT=/mnt/miaohua/charles/codes/LightX2V_fastwam/runs/robotwin_eval/consistency_$(date +%Y%m%d_%H%M%S) \
  bash lightx2v_train/scripts/eval_robotwin_distilled_1step.sh consistency
```

The launcher sources `/mnt/miaohua/charles/envs/miniconda3/etc/profile.d/conda.sh`,
activates `RoboTwin`, and uses that environment's Python
and `/mnt/afs_1/lvchengtao/code/wam/MeanFlowWAM/experiments/robotwin/run_robotwin_manager.py`.
The manager runs both clean and randomized phases for every task.
Defaults are 100 episodes per task per phase, unseen instructions, one inference step,
sigma shift 5.0, timing enabled, and observation skipping within replans.
Extra arguments are passed to Hydra. `OUT` sets the manager output directory;
the manager log is `$OUT/manager.log`. The existing evaluator also writes
per-task results under the FastWAM checkout's `evaluate_results/robotwin/`.

## Weights and architecture

Both exports use `checkpoint-000030000` from the named training runs.
Export directory: `/mnt/miaohua/charles/models/fastwam_release/distilled/`.

| Variant | Merged evaluation checkpoint | PEFT adapter directory |
| --- | --- | --- |
| DMD v2 student | `robotwin_dmd_v2_step30000.pt` | `robotwin_dmd_v2_step30000_lora/action/` |
| Consistency EMA | `robotwin_consistency_ema_step30000.pt` | `robotwin_consistency_ema_step30000_lora/action/` |

Adapters include `adapter_model.safetensors` and `adapter_config.json`, with
rank and alpha 128. Consistency uses EMA weights, matching training evaluation.
The manager loads the merged checkpoints, so no additional adapter is applied.

The installed FastWAM task is `robotwin_uncond_3cam_384_distilled_1step`.
It uses the ordinary FastWAM action architecture: action/proprio dimension 14,
hidden dimension 1024, FFN dimension 4096, 30 layers, 24 heads, head dimension
128, and a flow-matching scheduler with shift 5.0. The task itself carries
these settings because the simulator recomposes the task configuration.

## Rendering environment

FastWAM [issue #23](https://github.com/yuantianyuan01/FastWAM/issues/23)
identifies SAPIEN 3.0.3 rendering as a cause of reduced release-checkpoint
scores. This environment uses `sapien==3.0.0b1` and the already compatible
`setuptools==69.5.1`, which supplies `pkg_resources`. The launcher checks the
SAPIEN version before starting. Pins are in
`../configs/infer/robotwin_runtime_constraints.txt`.

The launcher uses the environment's activation hook to configure
`ROBOTWIN_NVIDIA_GL_ROOT` and its library search paths. It selects
`$ROBOTWIN_NVIDIA_GL_ROOT/nvidia_icd_abs.json` and preloads
`$ROBOTWIN_NVIDIA_GL_ROOT/libGL.so.1.7.0`, matching the supplied working
release-checkpoint command. It no longer references the machine-local
`/opt/robotwin-nvidia-headless-550.90.07` directory. GPU and Vulkan information
are printed before evaluation starts. The original RoboTwin ray-tracing
camera settings remain in use.

On another machine, the environment's bundled graphics libraries must match
the NVIDIA driver, and the FastWAM checkout must be available at `FASTWAM_ROOT`.
Do not run the checkout's old `eval.sh` installer, which installs SAPIEN 3.0.3.

The RGB evaluation does not use point clouds. The environment's incompatible
optional PyTorch3D extension reports `missing pytorch3d`; point-cloud use would
require a separate compatible build.

## Validation on 2026-09-05

The results below precede the switch to the environment's bundled graphics
libraries. The updated launcher has not been rerun, as requested.

Both variants were tested separately on this machine's eight H100 80GB GPUs,
using two processes per GPU with the previous `/opt` graphics runtime. Each process
ran one complete `click_alarmclock`, `demo_clean` episode at seed 4300000,
with unseen instructions, one inference step, and sigma shift 5.0.

| Variant | Completed processes | Successful episodes | Elapsed | Peak total GPU memory |
| --- | --- | --- | --- | --- |
| DMD v2 student | 16/16 | 16/16 | 502 seconds | 69,094 MiB |
| Consistency EMA | 16/16 | 16/16 | 430 seconds | 68,802 MiB |

Memory measurements include unrelated jobs already running on the machine.
All test processes exited with status zero and produced result files and
videos. These repeated episodes establish concurrent execution for this
task and seed; full benchmark accuracy still requires the 100-episode runs.
No claim of reproducing the paper's 91.8% follows from this smoke test.

Reports, relative to the LightX2V checkout:

- `runs/robotwin_8gpu_check/parallel_20260905_035145/summary.json` (DMD)
- `runs/robotwin_8gpu_check/parallel_20260905_040032/summary.json` (consistency)

The standalone scene probe also completed expert planning and verified
nonconstant, finite RGB pixels for all three model cameras. Export validation
covered exact LoRA tensor equality, merged weight samples, evaluation-model
key/shape compatibility, and four passing exporter tests.

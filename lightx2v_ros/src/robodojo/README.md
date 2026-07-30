# FastWAM on RoboDojo

This package contains the observation/policy adapter used to evaluate
LightX2V's native `FastWAMPolicy` through RoboDojo/XPolicyLab.

The evaluator supplies three RGB observations:

- `cam_head` -> `head_camera`
- `cam_left_wrist` -> `left_camera`
- `cam_right_wrist` -> `right_camera`

The XPolicyLab policy wrapper should pack the robot state with its
`pack_robot_state` helper, call `FastWAMRoboDojoAdapter.predict`, and unpack the
returned 14-D absolute joint targets with `unpack_robot_state`. The released
baseline-compatible settings are a 32-action chunk, 24 executed actions per
plan, 10 action denoising steps, and z-score normalization.

Required configuration paths are `model_path`, `checkpoint_path` (or
`adapter_model_path`), and `dataset_stats_path`. When T5/tokenizer/VAE assets
live outside `model_path`, pass absolute `t5_model_path`, `tokenizer_path`, and
`vae_model_path` values. Set `sequential_aux_offload: true` when inference and
Isaac Sim share a 24 GiB GPU.

## Evaluation status

The integration has completed end-to-end RoboDojo evaluation. Current results
are close to the upstream FastWAM baseline: layout 0 succeeds, while layout 1
still fails. This is recorded as a known generalization limitation rather than
an integration failure.

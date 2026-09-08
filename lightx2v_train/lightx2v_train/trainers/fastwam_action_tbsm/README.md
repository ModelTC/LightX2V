# FastWAM action TBSM

Pure instant scattering (`lambda=1`, `rho=0`) on dataset-normalized action
chunks. No tracker, consistency loss, flow loss, or extra feature network.

`training.action_tbsm.positive_source` selects the positive action:

- `teacher`: a frozen original FastWAM teacher, sampled for `teacher_steps` steps
  with independent noise under the same observation.
- `data`: the paired ground-truth action. No teacher forward runs in the training
  loss; the teacher is still used for periodic evaluation.

Both student samples start from independent pure noise at the final training
timestep. The negative uses the current student without gradients; EMA is used
only for evaluation/checkpointing. Each batch builds its observation KV cache
once. Only action LoRA parameters are trained by the supplied configs.

The no-grad sampling scope disables the autocast weight cache while retaining
the outer autocast dtype. Otherwise, a negative sample can cache detached BF16
casts of FP32 LoRA weights and cause the subsequent student loss to have no
`grad_fn`. The main student forward retains normal autocast caching.

The loss follows [TBSM](https://github.com/sp12138/TBSM),
`imgnet/methods/tbsm.py` (Apache-2.0; see the repository-root `LICENSE`), using
identity features with `feature_norm=[]`. Adaptations preserve action magnitude,
mask padding, scale bearings by the square root of each chunk's valid coordinate
count, and average per-chunk MSE over nonempty chunks. All loss math is float32.
The optimized loss is `raw / raw.detach().clamp_min(1e-8)`;
`train/scattering_loss` logs **raw**, not the usually unit-valued normalized loss
or an estimate of energy distance. An entirely padded batch has zero loss and
zero loss gradients; the inherited optimizer/EMA iteration still advances.

## Two nodes, eight GPUs per node

Run this on **both nodes**, with the same mode, rendezvous ID, `MASTER_ADDR`
(the first node's reachable address), and `MASTER_PORT`. Set `WANDB_API_KEY`
externally. Use a fresh run ID for a new experiment.

```bash
cd /mnt/afs_1/lvchengtao/code/wam/lightx2v_distill/lightx2v_train
export PYTHONPATH="/mnt/afs_1/lvchengtao/code/wam/lightx2v_distill:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NNODES=2
export NPROC_PER_NODE=8
export WANDB_ENTITY=lvchengtao-nanyang-technological-university-singapore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# teacher or data; the two modes use separate configs, outputs and W&B IDs.
TBSM_SOURCE=teacher
export RDZV_ID="fastwam-robotwin-action-1step-tbsm-${TBSM_SOURCE}-16gpu-lora-only"
export WANDB_RUN_ID="${RDZV_ID}"

../.venv/bin/torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR:?Set MASTER_ADDR}:${MASTER_PORT:?Set MASTER_PORT}" \
  --rdzv_id="${RDZV_ID}" \
  train.py \
  --config "configs/train/fastwam_action_tbsm/robotwin_action_1step_tbsm_${TBSM_SOURCE}.yaml"
```

For the data run, use `TBSM_SOURCE=data` and repeat the ID exports and torchrun
command on both nodes. The default global batch is 256, with 30,000 updates,
LoRA rank 128, learning rate `1e-4`, EMA decay `0.995`, and 20 reference teacher
steps. `teacher_steps` affects both teacher labels and reference evaluation;
student inference always uses one action step.

Checkpoints retain the existing action-trainer layout: `student_action.pt`,
`ema_action.pt`, `training_state.pt`, `config.yaml`, and per-rank RNG files.
Set `resume.resume_ckpt_path` to a checkpoint from the same mode and model, or
enable `resume.auto_resume` in its existing output directory. Resuming requires
the same world size. The model checkpoint in the config supplies the frozen
teacher and base weights. This format does not make the DMD-specific export CLI
compatible with TBSM configs.

## Validation

From `lightx2v_train`, run:

```bash
PYTHONPATH="..:${PYTHONPATH:-}" ../.venv/bin/python -m pytest tests/test_fastwam_action_tbsm.py -q
```

The tests cover the loss and gradients, masks/bf16, both positive sources,
independent noise and shared caches, LoRA, EMA, checkpoint/RNG restoration,
consistency loss/metric regression, and two-process CPU Gloo training with both
FP32 and BF16 LoRA. Trainer tests use tiny CPU model substitutes plus real small
ActionDiT/video DiT/MoT models under BF16 autocast, isolating the installed CUDA
extensions. All 21 tests passed after fixing the autocast cache interaction.
The loss was also compared directly with the local TBSM reference on three
unpadded action shapes; raw loss, normalized loss and gradients matched.

An additional H100 CUDA smoke test passed two training updates per mode with
real small DiTs, BF16 and FP32 LoRA, using PyTorch attention and isolating external
FlashAttention extensions. Full pretrained-model CUDA/FlashAttention training
has not been validated here; the installed FlashAttention 3 extension previously
failed to import with `undefined symbol`. No long training run was started.
Six focused legacy regression checks passed; two
legacy config assertions still expect `target_steps=2`, whereas the workspace's
action/joint configs now specify `10`. Those existing files were preserved.

export WANDB_ENTITY=lvchengtao-nanyang-technological-university-singapore
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY in the environment before running this script}"

cd /mnt/afs_1/lvchengtao/code/wam/lightx2v_distill/lightx2v_train

export PYTHONPATH="/mnt/afs_1/lvchengtao/code/wam/lightx2v_distill:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NNODES=2
export NPROC_PER_NODE=8
export RDZV_ID=fastwam-robotwin-action-1step-consistency-ts10-16gpu-lora-only
export WANDB_RUN_ID=fastwam-robotwin-action-1step-consistency-ts10-16gpu-lora-only

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

../.venv/bin/torchrun \
--nnodes="${NNODES}" \
--nproc_per_node="${NPROC_PER_NODE}" \
--rdzv_backend=c10d \
--rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
--rdzv_id="${RDZV_ID}" \
train.py \
--config /mnt/afs_1/lvchengtao/code/wam/lightx2v_distill/lightx2v_train/configs/train/fastwam_action_dmd/robotwin_action_1step_consistency.yaml

sleep inf


# with video
export WANDB_ENTITY=lvchengtao-nanyang-technological-university-singapore
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY in the environment before running this script}"

cd /mnt/afs_1/lvchengtao/code/wam/lightx2v_distill/lightx2v_train

export PYTHONPATH="/mnt/afs_1/lvchengtao/code/wam/lightx2v_distill:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NNODES=2
export NPROC_PER_NODE=8
export RDZV_ID=fastwam-robotwin-joint-1step-consistency-ts10-8gpu-lora-only
export WANDB_RUN_ID=fastwam-robotwin-joint-1step-consistency-ts10-8gpu-lora-only

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

../.venv/bin/torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv_id="${RDZV_ID}" \
  train.py \
  --config /mnt/afs_1/lvchengtao/code/wam/lightx2v_distill/lightx2v_train/configs/train/fastwam_action_dmd/robotwin_action_1step_consistency_joint.yaml

# with video no flow matching
export WANDB_ENTITY=lvchengtao-nanyang-technological-university-singapore
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY in the environment before running this script}"

cd /mnt/afs_1/lvchengtao/code/wam/lightx2v_distill/lightx2v_train

export PYTHONPATH="/mnt/afs_1/lvchengtao/code/wam/lightx2v_distill:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NNODES=2
export NPROC_PER_NODE=8
export RDZV_ID=fastwam-robotwin-joint-1step-consistency-ts10-noflow-16gpu-lora-only
export WANDB_RUN_ID=fastwam-robotwin-joint-1step-consistency-ts10-noflow-16gpu-lora-only

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

../.venv/bin/torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv_id="${RDZV_ID}" \
  train.py \
  --config /mnt/afs_1/lvchengtao/code/wam/lightx2v_distill/lightx2v_train/configs/train/fastwam_action_dmd/robotwin_action_1step_consistency_joint_noflow.yaml

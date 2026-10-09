#!/usr/bin/env bash
set -Eeuo pipefail

# Diagnostics only; no training run/cache/checkpoint is changed.
NEW_ROOT=${H3_NEW_ROOT:-/data/nvme6/gushiqiao/codes/latest/LightX2V}
OLD_ROOT=${H3_OLD_ROOT:-/data/nvme6/gushiqiao/codes/LightX2V}
PYTHON=${H3_PYTHON:-/data/nvme6/gushiqiao/models/MiniMax-H3/local_diffusers/.venv/bin/python}
MODEL=${H3_MODEL_PATH:-/data/nvme6/gushiqiao/models/MiniMax-H3}
METADATA=${H3_REF2AV_CACHE:-/data/nvme7/gushiqiao/datasets/minimax_h3_ref2av_image_audio_1to5_match_fixed124_cache/metadata.jsonl}
RESULT=${H3_PARITY_OUTPUT:-/data/nvme6/gushiqiao/h3_old_new_parity_$(date +%Y%m%d_%H%M%S)_$$}
mkdir -p "$RESULT/tmp" "$RESULT/old" "$RESULT/new"
export TMPDIR="$RESULT/tmp"
export HF_HOME="$MODEL/local_diffusers/.cache/huggingface"
export HF_HUB_OFFLINE=1
export LOCAL_KERNELS="kernels-community/flash-attn3=$MODEL/local_diffusers/.cache/kernels/kernels--kernels-community--flash-attn3/snapshots/43f0bd269777115d94ff826e0d113ce9c1c9087b"
export OMP_NUM_THREADS=2
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
SCRIPT="$NEW_ROOT/lightx2v_train/tools/h3_old_new_fsdp_parity.py"

echo "Results: $RESULT; old=GPU0,1 new=GPU2,3; each FSDP2 world_size=2"
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="$OLD_ROOT/lightx2v_train" \
  "$PYTHON" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 --max_restarts=0 \
  "$SCRIPT" --implementation old --model "$MODEL" --metadata "$METADATA" \
  --output "$RESULT/old" --fake-cpu-offload "$@" > "$RESULT/old.log" 2>&1 &
old_pid=$!
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH="$NEW_ROOT/lightx2v_train" \
  "$PYTHON" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 --max_restarts=0 \
  "$SCRIPT" --implementation new --model "$MODEL" --metadata "$METADATA" \
  --output "$RESULT/new" --fake-cpu-offload "$@" > "$RESULT/new.log" 2>&1 &
new_pid=$!
old_rc=0
new_rc=0
wait "$old_pid" || old_rc=$?
wait "$new_pid" || new_rc=$?
if (( old_rc != 0 || new_rc != 0 )); then
  echo "Comparison workers failed: old=$old_rc new=$new_rc; inspect $RESULT/{old,new}.log" >&2
  exit 1
fi
"$PYTHON" "$NEW_ROOT/lightx2v_train/tools/compare_h3_parity_results.py" \
  --old "$RESULT/old" --new "$RESULT/new" --output "$RESULT/comparison.json"

#!/usr/bin/env bash
set -euo pipefail

train_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${H3_PYTHON:-python}
config=${H3_MOE_CONFIG:-$train_root/configs/train/dense2moe/minimax_h3_t2av_recovery_fsdp4.yaml}
export PYTHONPATH="$train_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTORCH_ALLOC_CONF=expandable_segments:True

run_dir=$("$python_bin" - "$config" <<'PY'
import os
import sys
import json
from pathlib import Path
from omegaconf import OmegaConf

config = OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=True)
output = Path(config["output_dir"])
source = Path(json.loads((Path(config["initialize_checkpoint"]) / "progress.json").read_text())["config"]["output_dir"])
(output / "logs").mkdir(parents=True, exist_ok=True)
for name in ("shared.pt", "partition.pt"):
    destination = output / name
    if not destination.exists():
        os.link(source / name, destination)
print(output)
PY
)
exec 9>"$run_dir/logs/train.lock"
flock -n 9
echo $$ > "$run_dir/logs/train.pid"
exec "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m lightx2v_train.trainers.dense2moe.run --config "$config" --stage moe \
    >> "$run_dir/logs/train_moe.log" 2>&1

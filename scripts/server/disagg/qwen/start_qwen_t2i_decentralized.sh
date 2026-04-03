#!/bin/bash
#
# Qwen Image T2I decentralized disagg — 4-GPU deployment:
#   GPU0: Controller (RDMA rings) + Encoder HTTP (port 8002) + Decoder pull worker
#   GPU1: Transformer pull worker (rank=1)
#   GPU2: Transformer pull worker (rank=2)
#   GPU3: Transformer pull worker (rank=3)
#
# Usage:
#   bash scripts/server/disagg/qwen/start_qwen_t2i_decentralized.sh
#
# Override paths:
#   LIGHTX2V_PATH=/path/to/LightX2V QWEN_IMAGE_MODEL_PATH=/path/to/model ./start_qwen_t2i_decentralized.sh
#
set -e

LIGHTX2V_PATH="${LIGHTX2V_PATH:-/path/to/LightX2V}"
MODEL_PATH="${QWEN_IMAGE_MODEL_PATH:-/path/to/model}"

CFG_CTRL="${LIGHTX2V_PATH}/configs/disagg/qwen/qwen_image_t2i_disagg_controller.json"
CFG_ENC="${LIGHTX2V_PATH}/configs/disagg/qwen/qwen_image_t2i_disagg_encoder_decentralized.json"
CFG_DEC="${LIGHTX2V_PATH}/configs/disagg/qwen/qwen_image_t2i_disagg_decode_decentralized.json"
CFG_TRANS_TMPL="${LIGHTX2V_PATH}/configs/disagg/qwen/qwen_image_t2i_disagg_transformer_decentralized.json"
TMP_DIR="${TMP_DIR:-/tmp/qwen_t2i_decentralized_cfg}"
mkdir -p "${TMP_DIR}"

export MOONCAKE_CONFIG_PATH="${LIGHTX2V_PATH}/configs/mooncake_config.json"

# Generate per-rank transformer configs (rank 1, 2, 3)
export LIGHTX2V_PATH TMP_DIR
python3 - <<'PY'
import json, os
base = os.environ['LIGHTX2V_PATH']
tmp = os.environ['TMP_DIR']
src = os.path.join(base, 'configs/disagg/qwen/qwen_image_t2i_disagg_transformer_decentralized.json')
with open(src, 'r', encoding='utf-8') as f:
    tmpl = json.load(f)
for rank, name in [(1, 'transformer_r1.json'), (2, 'transformer_r2.json'), (3, 'transformer_r3.json')]:
    data = json.loads(json.dumps(tmpl))
    data['disagg_config']['receiver_engine_rank'] = rank
    data['disagg_config']['transformer_engine_rank'] = rank
    dst = os.path.join(tmp, name)
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
print(f"Generated transformer configs in {tmp}")
PY

# --- 1. Start Controller (RDMA ring buffers only, no subprocess management) ---
(
set -e
export LIGHTX2V_PATH="${LIGHTX2V_PATH}"
export PYTHONPATH="${LIGHTX2V_PATH}:${PYTHONPATH:-}"
cd "${LIGHTX2V_PATH}"

python3 -u - <<PYCTRL
import json
from lightx2v.disagg.services.controller import ControllerService
cfg_path = '${CFG_CTRL}'.strip()
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)
dc = cfg.get('disagg_config', {})
cfg.setdefault('data_bootstrap_addr', dc.get('bootstrap_addr', '127.0.0.1'))
ControllerService().serve_rdma_dispatch_only(cfg)
PYCTRL
) &
PID_CTRL=$!

sleep 5

# --- 2. Start Encoder HTTP server on GPU 0 ---
(
set -e
export CUDA_VISIBLE_DEVICES=0
export LIGHTX2V_PATH="${LIGHTX2V_PATH}"
export PYTHONPATH="${LIGHTX2V_PATH}:${PYTHONPATH:-}"
cd "${LIGHTX2V_PATH}"
python3 -m lightx2v.server \
  --model_cls qwen_image \
  --task t2i \
  --model_path "${MODEL_PATH}" \
  --config_json "${CFG_ENC}" \
  --host 0.0.0.0 \
  --port 8002
) &
PID_ENC=$!

# --- 3. Start Decoder pull worker on GPU 0 ---
(
set -e
export CUDA_VISIBLE_DEVICES=0
export LIGHTX2V_PATH="${LIGHTX2V_PATH}"
export PYTHONPATH="${LIGHTX2V_PATH}:${PYTHONPATH:-}"
cd "${LIGHTX2V_PATH}"
python3 -m lightx2v.disagg.examples.qwen_t2i_queue_workers \
  --role decoder \
  --model_path "${MODEL_PATH}" \
  --config_json "${CFG_DEC}"
) &
PID_DEC=$!

# --- 4. Start Transformer pull workers on GPU 1, 2, 3 ---
start_trans() {
  local gpu="$1"
  local cfg="$2"
  (
  set -e
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export LIGHTX2V_PATH="${LIGHTX2V_PATH}"
  export PYTHONPATH="${LIGHTX2V_PATH}:${PYTHONPATH:-}"
  cd "${LIGHTX2V_PATH}"
  python3 -m lightx2v.disagg.examples.qwen_t2i_queue_workers \
    --role transformer \
    --model_path "${MODEL_PATH}" \
    --config_json "${cfg}"
  ) &
}

start_trans 1 "${TMP_DIR}/transformer_r1.json"
PID_T1=$!
start_trans 2 "${TMP_DIR}/transformer_r2.json"
PID_T2=$!
start_trans 3 "${TMP_DIR}/transformer_r3.json"
PID_T3=$!

cleanup() {
  echo "Stopping decentralized Qwen T2I services..."
  kill ${PID_CTRL} ${PID_ENC} ${PID_DEC} ${PID_T1} ${PID_T2} ${PID_T3} 2>/dev/null || true
  wait 2>/dev/null || true
  echo "Stopped."
}
trap cleanup EXIT INT TERM

echo ""
echo "=== Decentralized Qwen T2I services started ==="
echo "  Controller (RDMA rings) PID=${PID_CTRL}"
echo "  Encoder HTTP            PID=${PID_ENC}  GPU=0  port=8002"
echo "  Decoder worker          PID=${PID_DEC}  GPU=0"
echo "  Transformer workers     PID=${PID_T1} GPU=1, PID=${PID_T2} GPU=2, PID=${PID_T3} GPU=3"
echo "  Transformer configs:    ${TMP_DIR}/transformer_r*.json"
echo ""
echo "Press Ctrl+C to stop all services."
wait

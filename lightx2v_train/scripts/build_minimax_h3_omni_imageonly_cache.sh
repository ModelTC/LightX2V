#!/usr/bin/env bash
# One independent encoder per GPU; four nodes share one 32-shard namespace.
set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 2 || ! "$1" =~ ^[0-3]$ || ( $# -eq 2 && "$2" != --dry-run ) ]]; then
    echo "Usage: bash $0 NODE_INDEX_0_TO_3 [--dry-run]" >&2
    exit 2
fi
node_index="$1"
dry_run="${2:-}"
H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
H3_PYTHON="${H3_PYTHON:-python}"
H3_OMNI_METADATA="${H3_OMNI_METADATA:-/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/omni_r2v_image_only_100000_ir.jsonl}"
H3_MODEL_PATH="${H3_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
H3_SOURCE_MODEL_PATH="${H3_SOURCE_MODEL_PATH:-$H3_MODEL_PATH/Ref2VA}"
H3_MEDIA_ROOT="${H3_MEDIA_ROOT:-/mnt/lm_data_afs/gushiqiao/datasets/omini-r2v}"
H3_CACHE_DIR="${H3_CACHE_DIR:-/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/latent_match124_bf16}"
H3_CACHE_STAGE="${H3_CACHE_STAGE:-all}"
case "$H3_CACHE_STAGE" in all|text|references) ;; *) echo "H3_CACHE_STAGE must be all, text or references" >&2; exit 2 ;; esac
IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
if [[ "${#gpu_ids[@]}" -ne 8 ]]; then
    echo "Each node needs exactly eight CUDA_VISIBLE_DEVICES entries." >&2
    exit 2
fi
declare -A seen_devices=()
for gpu in "${gpu_ids[@]}"; do
    if [[ -z "$gpu" || -n "${seen_devices[$gpu]:-}" ]]; then
        echo "GPU entries must be nonempty and unique." >&2; exit 2
    fi
    seen_devices[$gpu]=1
done
builder="$H3_CODE_ROOT/lightx2v_train/data_process/minimax_h3/build_minimax_h3_ref2av_condition_caches.py"
command -v "$H3_PYTHON" >/dev/null
[[ -f "$builder" ]] || { echo "Missing builder: $builder" >&2; exit 1; }

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
command=("$H3_PYTHON" -u "$builder" "$H3_OMNI_METADATA"
    --output-dir "$H3_CACHE_DIR" --model-path "$H3_MODEL_PATH"
    --source-model-path "$H3_SOURCE_MODEL_PATH" --media-root "$H3_MEDIA_ROOT"
    --dtype bf16 --reference-latent-dtype bf16 --reference-resize-mode match
    --prompt-policy enhanced-or-original --target-policy fixed-768p
    --target-num-frames 124 --image-only --skip-invalid --num-shards 32
    --device cuda:0 --stage "$H3_CACHE_STAGE")

if [[ "$dry_run" == --dry-run ]]; then
    for local_index in {0..7}; do
        printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu_ids[$local_index]}"
        printf '%q ' "${command[@]}" --shard-index "$((node_index * 8 + local_index))"
        printf '\n'
    done
    exit 0
fi

[[ -s "$H3_OMNI_METADATA" ]] || { echo "Missing/empty input: $H3_OMNI_METADATA" >&2; exit 1; }
for required in "$H3_SOURCE_MODEL_PATH/text_encoder/config.json" "$H3_SOURCE_MODEL_PATH/video_vae/config.json" "$H3_MODEL_PATH/vae/config.json" "$H3_MODEL_PATH/audio_vae/config.json"; do
    [[ -f "$required" ]] || { echo "Missing model component: $required (set H3_MODEL_PATH/H3_SOURCE_MODEL_PATH)" >&2; exit 1; }
done
command -v flock >/dev/null
mkdir -p "$H3_CACHE_DIR/logs"
exec 9>"$H3_CACHE_DIR/.node-$node_index.lock"
flock -n 9 || { echo "An encoder already owns node $node_index in $H3_CACHE_DIR" >&2; exit 1; }

pids=()
stop_workers() {
    trap - INT TERM
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    exit 130
}
trap stop_workers INT TERM
run_stamp="$(date +%Y%m%d_%H%M%S)_$$"
for local_index in {0..7}; do
    shard_index=$((node_index * 8 + local_index))
    log="$H3_CACHE_DIR/logs/shard-$(printf '%03d' "$shard_index").$run_stamp.log"
    CUDA_VISIBLE_DEVICES="${gpu_ids[$local_index]}" "${command[@]}" --shard-index "$shard_index" >"$log" 2>&1 &
    pids+=("$!")
    echo "node=$node_index shard=$shard_index/32 gpu=${gpu_ids[$local_index]} pid=$! log=$log"
done
rc=0
for pid in "${pids[@]}"; do
    if wait "$pid"; then :; else rc=1; echo "Encoder pid=$pid failed; inspect its log above." >&2; fi
done
trap - INT TERM
if [[ "$rc" -eq 0 ]]; then
    echo "Node $node_index finished stage=$H3_CACHE_STAGE. After ALL FOUR nodes finish stage=all/references, merge the 32 manifests."
fi
exit "$rc"

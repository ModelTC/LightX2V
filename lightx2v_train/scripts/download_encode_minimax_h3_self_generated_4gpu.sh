#!/usr/bin/env bash
# Download once, then run four independent H3 encoders and publish verified caches.
set -Eeuo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi
H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
H3_PYTHON="${H3_PYTHON:-python}"
H3_MODEL_PATH="${H3_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
H3_REF_MODEL_PATH="${H3_REF_MODEL_PATH:-$H3_MODEL_PATH/Ref2VA}"
H3_SELFGEN_RAW="${H3_SELFGEN_RAW:-/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_self_generated}"
H3_SELFGEN_CACHE="${H3_SELFGEN_CACHE:-/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_self_generated_latents_124_bf16}"
H3_SELFGEN_SUBSETS="${H3_SELFGEN_SUBSETS:-all}"
H3_SELFGEN_REVISION="${H3_SELFGEN_REVISION:-master}"
H3_SELFGEN_DOWNLOAD_WORKERS="${H3_SELFGEN_DOWNLOAD_WORKERS:-8}"
H3_SELFGEN_ALLOW_FAILURES="${H3_SELFGEN_ALLOW_FAILURES:-0}"
H3_SELFGEN_ALLOW_MISSING_MEDIA="${H3_SELFGEN_ALLOW_MISSING_MEDIA:-1}"
case "$H3_SELFGEN_SUBSETS" in all|t2av|ref2av) ;; *) echo "H3_SELFGEN_SUBSETS must be all, t2av or ref2av." >&2; exit 2 ;; esac
case "$H3_SELFGEN_ALLOW_FAILURES" in 0|1) ;; *) echo "H3_SELFGEN_ALLOW_FAILURES must be 0 or 1." >&2; exit 2 ;; esac
case "$H3_SELFGEN_ALLOW_MISSING_MEDIA" in 0|1) ;; *) echo "H3_SELFGEN_ALLOW_MISSING_MEDIA must be 0 or 1." >&2; exit 2 ;; esac
if [[ ! "$H3_SELFGEN_DOWNLOAD_WORKERS" =~ ^[1-9][0-9]*$ || ( -n "${H3_SELFGEN_MAX_SAMPLES:-}" && ! "$H3_SELFGEN_MAX_SAMPLES" =~ ^[1-9][0-9]*$ ) ]]; then
    echo "Download workers and optional H3_SELFGEN_MAX_SAMPLES must be positive integers." >&2
    exit 2
fi

visible_devices="${CUDA_VISIBLE_DEVICES-0,1,2,3}"
if [[ -z "$visible_devices" || "$visible_devices" == ,* || "$visible_devices" == *, || "$visible_devices" == *,,* || "$visible_devices" == *[[:space:]]* ]]; then
    echo "CUDA_VISIBLE_DEVICES must contain four nonempty, unique GPU IDs without spaces." >&2
    exit 2
fi
IFS=',' read -r -a gpu_ids <<< "$visible_devices"
if [[ "${#gpu_ids[@]}" -ne 4 ]]; then
    echo "This launcher requires exactly four CUDA_VISIBLE_DEVICES entries." >&2
    exit 2
fi
declare -A seen_devices=()
for gpu in "${gpu_ids[@]}"; do
    if [[ -n "${seen_devices[$gpu]:-}" ]]; then
        echo "CUDA_VISIBLE_DEVICES entries must be unique." >&2
        exit 2
    fi
    seen_devices[$gpu]=1
done

helper="$H3_CODE_ROOT/lightx2v_train/data_process/minimax_h3/prepare_modelscope_h3_dataset.py"
encoder="$H3_CODE_ROOT/lightx2v_train/data_process/minimax_h3/encode_modelscope_h3_dataset.py"
command -v "$H3_PYTHON" >/dev/null
for source in "$helper" "$encoder"; do
    [[ -f "$source" ]] || { echo "Missing code: $source; sync the dataset tools first." >&2; exit 1; }
done
download=("$H3_PYTHON" -u "$helper" download --dataset-dir "$H3_SELFGEN_RAW"
    --revision "$H3_SELFGEN_REVISION" --download-workers "$H3_SELFGEN_DOWNLOAD_WORKERS")
prepare=("$H3_PYTHON" -u "$helper" prepare --dataset-dir "$H3_SELFGEN_RAW"
    --output-dir "$H3_SELFGEN_CACHE" --subsets "$H3_SELFGEN_SUBSETS")
encode=("$H3_PYTHON" -u "$encoder" --metadata "$H3_SELFGEN_CACHE/prepared.jsonl"
    --output-dir "$H3_SELFGEN_CACHE" --model-path "$H3_MODEL_PATH"
    --source-model-path "$H3_REF_MODEL_PATH" --device cuda:0 --dtype bf16 --num-shards 4)
merge=("$H3_PYTHON" -u "$helper" merge --output-dir "$H3_SELFGEN_CACHE" --num-shards 4)
if [[ -n "${H3_SELFGEN_MAX_SAMPLES:-}" ]]; then
    prepare+=(--max-samples-per-subset "$H3_SELFGEN_MAX_SAMPLES")
fi
if [[ "$H3_SELFGEN_ALLOW_MISSING_MEDIA" == 1 ]]; then
    prepare+=(--allow-missing-media)
fi
if [[ "$H3_SELFGEN_ALLOW_FAILURES" == 1 ]]; then
    encode+=(--allow-failures)
    merge+=(--allow-failures)
fi
print_command() { printf '%q ' "$@"; printf '\n'; }
if [[ "${1:-}" == --dry-run ]]; then
    echo "ModelScope is installed into H3_PYTHON's environment only if missing (actual run)."
    print_command "${download[@]}"
    print_command "${prepare[@]}"
    for shard in 0 1 2 3; do
        printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu_ids[$shard]}"
        print_command "${encode[@]}" --shard-index "$shard"
    done
    print_command "${merge[@]}"
    exit 0
fi

[[ -d "$H3_MODEL_PATH" ]] || { echo "Missing local H3 model directory: $H3_MODEL_PATH" >&2; exit 1; }
if [[ "$H3_SELFGEN_SUBSETS" != t2av ]]; then
    [[ -d "$H3_REF_MODEL_PATH" ]] || { echo "Missing local H3 Ref2VA directory: $H3_REF_MODEL_PATH" >&2; exit 1; }
fi
required_components=("$H3_MODEL_PATH/vae/config.json" "$H3_MODEL_PATH/audio_vae/config.json")
if [[ "$H3_SELFGEN_SUBSETS" != ref2av ]]; then
    required_components+=("$H3_MODEL_PATH/text_encoder/config.json" "$H3_MODEL_PATH/tokenizer/tokenizer_config.json" "$H3_MODEL_PATH/processor/preprocessor_config.json")
fi
if [[ "$H3_SELFGEN_SUBSETS" != t2av ]]; then
    required_components+=("$H3_REF_MODEL_PATH/text_encoder/config.json" "$H3_REF_MODEL_PATH/tokenizer/tokenizer_config.json" "$H3_REF_MODEL_PATH/processor/preprocessor_config.json" "$H3_REF_MODEL_PATH/video_vae/config.json" "$H3_REF_MODEL_PATH/audio_vae/config.json")
fi
for component in "${required_components[@]}"; do
    [[ -f "$component" ]] || { echo "Missing local H3 component: $component" >&2; exit 1; }
done
# Fail before a large download if the selected environment lacks the encoders.
"$H3_PYTHON" -c 'import av, torch; from PIL import Image; from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio; from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor'
command -v flock >/dev/null
mkdir -p "$H3_SELFGEN_RAW" "$H3_SELFGEN_CACHE/logs"
exec 9>"$H3_SELFGEN_CACHE/.download-encode.lock"
flock -n 9 || { echo "Another download/encode job owns $H3_SELFGEN_CACHE" >&2; exit 1; }
export PYTHONPATH="$H3_CODE_ROOT/lightx2v_train:$H3_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cleanup_workers() {
    # jobs reports only running children owned by this shell, including after wait -n.
    local pid
    while read -r pid; do
        [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true
    done < <(jobs -pr)
    wait 2>/dev/null || true
}
trap cleanup_workers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! "$H3_PYTHON" -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("modelscope") is None)'; then
    echo "Installing missing ModelScope download dependency into $H3_PYTHON's environment."
    "$H3_PYTHON" -m pip install modelscope
fi
echo "[1/4] Downloading/resuming DiffSynth-Studio/MiniMax-H3-Self-Generated-Dataset to $H3_SELFGEN_RAW"
"${download[@]}"
echo "[2/4] Preparing and checking local media metadata"
"${prepare[@]}"
echo "[3/4] Encoding on GPUs $visible_devices (one independent shard per GPU)"
run_stamp="$(date +%Y%m%d_%H%M%S)_$$"
pids=()
for shard in 0 1 2 3; do
    log="$H3_SELFGEN_CACHE/logs/shard-$(printf '%05d' "$shard").$run_stamp.log"
    CUDA_VISIBLE_DEVICES="${gpu_ids[$shard]}" "${encode[@]}" --shard-index "$shard" >"$log" 2>&1 &
    pids+=("$!")
    echo "shard=$shard/4 gpu=${gpu_ids[$shard]} pid=$! log=$log"
done
for _ in 0 1 2 3; do
    if wait -n; then
        :
    else
        status=$?
        # Bash 5.0 can report 127 if every short-lived worker already exited.
        # Explicit waits below still recover every recorded child's status.
        if [[ "$status" -eq 127 ]]; then break; fi
        echo "An encoder failed (status=$status); stopping this launcher's remaining workers. Inspect $H3_SELFGEN_CACHE/logs." >&2
        exit "$status"
    fi
done
for pid in "${pids[@]}"; do
    if wait "$pid"; then :; else
        status=$?
        echo "Encoder pid=$pid failed (status=$status); manifests were not merged." >&2
        exit "$status"
    fi
done
echo "[4/4] Verifying all four shards and publishing training manifests"
"${merge[@]}"
echo "Finished: $H3_SELFGEN_CACHE"
echo "CM T2AV 768p manifest: $H3_SELFGEN_CACHE/metadata.jsonl"
echo "All native resolutions: $H3_SELFGEN_CACHE/all_metadata.jsonl"

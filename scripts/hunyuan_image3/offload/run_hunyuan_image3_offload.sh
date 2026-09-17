#!/bin/bash
set -eo pipefail

task="${1:?Usage: run_hunyuan_image3_offload.sh t2i|ti2i private|host|numa [infer arguments]}"
scope="${2:?Missing offload scope: private, host or numa}"
shift 2
case "${task}" in t2i|ti2i) ;; *) echo "Unsupported task: ${task}" >&2; exit 2 ;; esac
case "${scope}" in private|host|numa) ;; *) echo "Unsupported scope: ${scope}" >&2; exit 2 ;; esac
for argument in "$@"; do
    case "${argument}" in
        --config_json|--config_json=*|--model_cls|--model_cls=*|--task|--task=*)
            echo "Use CONFIG_JSON and the matching task entry to change the validated launch configuration." >&2
            exit 2 ;;
    esac
done

# Preserve caller-relative paths when switching to the repository below.
absolute_path() {
    case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s/%s\n' "${PWD}" "$1" ;; esac
}

lightx2v_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="$(absolute_path "${HUNYUAN_IMAGE3_MODEL_PATH:-${lightx2v_path}/../HunyuanImage-3-Instruct}")"
export HUNYUAN_IMAGE3_REPO_PATH="$(absolute_path "${HUNYUAN_IMAGE3_REPO_PATH:-${lightx2v_path}/../HunyuanImage-3.0}")"
suffix="block_shared_${scope}"
if [[ "${scope}" == private ]]; then suffix=block; fi
config_json="$(absolute_path "${CONFIG_JSON:-${lightx2v_path}/configs/hunyuan_image3/offload/hunyuan_image3_${task}_${suffix}_tp2_sp2_cfg2.json}")"

# Resolve the launcher topology from the selected config, before GPU startup.
# NPROC_PER_NODE is an optional consistency check; CONFIG_JSON changes topology.
nproc="$(python - "${config_json}" "${scope}" "${NPROC_PER_NODE:-}" <<'PY'
import json
import sys

path, scope, requested = sys.argv[1:]
with open(path) as source:
    config = json.load(source)
if not config.get('cpu_offload') or config.get('offload_granularity') != 'block':
    raise SystemExit('The selected configuration must enable CPU block offload')
if bool(config.get('shared_cpu_weights')) != (scope != 'private'):
    raise SystemExit('shared_cpu_weights does not match this launch entry')
if scope != 'private' and config.get('shared_cpu_weight_scope') != scope:
    raise SystemExit(f'This entry requires shared_cpu_weight_scope={scope!r}')
parallel = config.get('parallel') or {}
size = 1
for key in ('tensor_p_size', 'seq_p_size', 'cfg_p_size'):
    value = parallel.get(key, 1)
    if type(value) is not int or value < 1:
        raise SystemExit(f'Invalid parallel.{key}: {value!r}')
    size *= value
if requested and int(requested) != size:
    raise SystemExit(f'NPROC_PER_NODE={requested} does not match configured TP*SP*CFG={size}; select a matching CONFIG_JSON')
print(size)
PY
)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(python -c 'import sys; print(",".join(map(str, range(int(sys.argv[1])))))' "${nproc}")}"
export PYTHONPATH="${HUNYUAN_IMAGE3_REPO_PATH}:${PYTHONPATH:-}"
source "${lightx2v_path}/scripts/base/base.sh"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# The leader may need to read ~150 GiB from remote storage while peers wait.
export LIGHTX2V_DIST_TIMEOUT_SECONDS="${LIGHTX2V_DIST_TIMEOUT_SECONDS:-3600}"

if [[ "${task}" == ti2i ]]; then
    default_prompt='新年宠物海报，Q版圆润的可爱标题“新年快乐汪”，副标题“HAPPY NEW YEAR”。鱼眼镜头，背景是房间门口，上传的主体歪头笑，围着红色围巾，戴着红色毛线帽，高清绒毛细节，面部特写，宝丽莱相纸，写实胶片摄影，复古颗粒感。'
    image_args=(--image_path "$(absolute_path "${HUNYUAN_IMAGE3_IMAGE_PATH:-${HUNYUAN_IMAGE3_REPO_PATH}/assets/demo_instruct_imgs/input_0_0.png}")")
else
    default_prompt='生成图片：一辆汽车行驶在高速公路上，驾驶员在打电话，副驾驶坐着一只狗'
    image_args=()
fi
save_path="$(absolute_path "${HUNYUAN_IMAGE3_SAVE_RESULT_PATH:-${lightx2v_path}/save_results/hunyuan_image3_${task}_${suffix}.png}")"
mkdir -p -- "$(dirname -- "${save_path}")"
cd -- "${lightx2v_path}"
exec python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${nproc}" -m lightx2v.infer \
    --model_cls hunyuan_image3 \
    --task "${task}" \
    --model_path "${model_path}" \
    --config_json "${config_json}" \
    --prompt "${HUNYUAN_IMAGE3_PROMPT:-${default_prompt}}" \
    "${image_args[@]}" \
    --save_result_path "${save_path}" \
    --seed "${SEED:-42}" \
    "$@"

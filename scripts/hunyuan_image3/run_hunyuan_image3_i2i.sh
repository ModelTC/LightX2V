#!/bin/bash

export lightx2v_path=${lightx2v_path:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export model_path=${model_path:-/path/to/HunyuanImage-3-Instruct}
export image_path=${image_path:-/path/to/input.png}

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count=$(nvidia-smi -L | wc -l)
    if [ "$gpu_count" -gt 0 ]; then
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpu_count - 1)))
    fi
fi

source ${lightx2v_path}/scripts/base/base.sh

config_json="${lightx2v_path}/configs/hunyuan_image3/hunyuan_image3_i2i.json"
echo "config_json=${config_json}"

python -m lightx2v.infer \
    --model_cls hunyuan_image3 \
    --task i2i \
    --model_path $model_path \
    --config_json "$config_json" \
    --prompt "新年宠物海报，Q版圆润的可爱标题“新年快乐汪”，副标题“HAPPY NEW YEAR”。 鱼眼镜头，背景是房间门口，近景，上传的主体歪头笑，围着红色围巾，戴着红色毛线帽，高清，绒毛细节，面部特写。" \
    --image_path "${image_path}" \
    --save_result_path ${lightx2v_path}/save_results/hunyuan_image3_i2i.png \
    --seed 42

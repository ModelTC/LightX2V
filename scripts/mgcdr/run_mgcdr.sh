#!/bin/bash

source /kaiwu_vepfs/kaiwu/huangxinchi/metavdt/bin/activate

export PYTHONPATH=/kaiwu_vepfs/kaiwu/huangxinchi/metavdt/lib/python3.10/site-packages:$PYTHONPATH

# set path and first
lightx2v_path=/kaiwu_vepfs/kaiwu/huangxinchi/lightx2v
magicdrivedit_path=/kaiwu_vepfs/kaiwu/huangxinchi/drivescapedit
model_path='/kaiwu_vepfs/kaiwu/xujin2/code_hsy/magicdrivedit/outputs/zhiji_0509/MagicDriveSTDiT3-XL-2_zhiji_0509_20250513-0620/epoch0-global_step512/ema.pt'

export CUDA_VISIBLE_DEVICES=7

# check section
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    cuda_devices=0
    echo "Warn: CUDA_VISIBLE_DEVICES is not set, using default value: ${cuda_devices}, change at shell script or set env variable."
    export CUDA_VISIBLE_DEVICES=${cuda_devices}
fi

if [ -z "${lightx2v_path}" ]; then
    echo "Error: lightx2v_path is not set. Please set this variable first."
    exit 1
fi

if [ -z "${model_path}" ]; then
    echo "Error: model_path is not set. Please set this variable first."
    exit 1
fi

export TOKENIZERS_PARALLELISM=false

export PYTHONPATH=${lightx2v_path}:${magicdrivedit_path}:${magicdrivedit_path}/magicdrivedit/datasets:$PYTHONPATH
echo $PYTHONPATH
export DTYPE=BF16
export ENABLE_PROFILING_DEBUG=true
export ENABLE_GRAPH_MODE=false

python -m lightx2v.infer_magicdrive \
--model_cls mgcdr \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/mgcdr/config.json \
--dataset_params_json ${lightx2v_path}/configs/mgcdr/dataset.json \
--camera_params_json ${lightx2v_path}/configs/mgcdr/cam_params \
--raw_meta_files "/kaiwu_vepfs/kaiwu/hule/codes/drivescapedit/outputs_infer/test_checked" \
--save_video_path ${lightx2v_path}/save_results/output_lightx2v_mgcdr.mp4

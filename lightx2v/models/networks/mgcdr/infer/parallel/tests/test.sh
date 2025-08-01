lightx2v_path=/kaiwu_vepfs/kaiwu/huangxinchi/lightx2v


# check section
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    cuda_devices=0,1,2,3,4,5,6,7
    echo "Warn: CUDA_VISIBLE_DEVICES is not set, using default value: ${cuda_devices}, change at shell script or set env variable."
    export CUDA_VISIBLE_DEVICES=${cuda_devices}
fi

if [ -z "${lightx2v_path}" ]; then
    echo "Error: lightx2v_path is not set. Please set this variable first."
    exit 1
fi


export PYTHONPATH=${lightx2v_path}:$PYTHONPATH


source /kaiwu_vepfs/kaiwu/huangxinchi/metavdt/bin/activate


torchrun --nproc_per_node=8 test.py
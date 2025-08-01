export PYTHONPATH=/kaiwu_vepfs/kaiwu/huangxinchi/lightx2v:$PYTHONPATH
export ENABLE_PROFILING_DEBUG=True
export DTYPE='BF16'

source /kaiwu_vepfs/kaiwu/huangxinchi/metavdt/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 test_dist.py
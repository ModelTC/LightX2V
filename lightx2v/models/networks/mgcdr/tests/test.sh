export PYTHONPATH=/kaiwu_vepfs/kaiwu/huangxinchi/lightx2v:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export DTYPE='BF16'

# pip3 install qtorch --target /kaiwu_vepfs/kaiwu/huangxinchi/metavdt/lib/python3.10/site-packages/

# ENV_FJ="/kaiwu_vepfs/kaiwu/xujin2/envs/metavdt"
# echo "Activating environment at $ENV_FJ"
# source $ENV_FJ/bin/activate
source /kaiwu_vepfs/kaiwu/huangxinchi/metavdt/bin/activate


python3 test.py
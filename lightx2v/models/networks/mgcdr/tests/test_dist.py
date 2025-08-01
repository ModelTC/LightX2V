from lightx2v.common.ops import *
import json
from lightx2v.common.transformer_infer import transformer_infer
import torch
import torch.distributed as dist
import pickle
from lightx2v.models.networks.mgcdr.infer.pre_infer import MagicDrivePreInfer
from lightx2v.models.networks.mgcdr.weights.pre_weights import MagicDrivePreWeights
from lightx2v.models.networks.mgcdr.infer.transformer_infer import MagicDriveTransformerInfer
from lightx2v.models.networks.mgcdr.weights.transformer_weights import MagicDriveTransformerWeight
from lightx2v.models.networks.mgcdr.infer.post_infer import MagicDrivePostInfer
from lightx2v.models.networks.mgcdr.weights.post_weights import MagicDrivePostWeights
from lightx2v.utils.profiler import ProfilingContext4Debug
from lightx2v.models.networks.mgcdr.infer.parallel.coordinator import SequenceParallelCoordinator
from lightx2v.models.networks.mgcdr.infer.parallel.attn import sequence_split_before_transformer_infer, sequence_gather_after_transformer_infer

dist.init_process_group(backend="nccl")
torch.cuda.set_device(dist.get_rank())


coordinator = SequenceParallelCoordinator(
    config = {
        "cfg_parallel_size": 1
    }
)


def main():
    with open('/kaiwu_vepfs/kaiwu/huangxinchi/test_inputs/_model_args.pkl', 'rb') as f:
        _model_args = pickle.load(f)
        for k, v in _model_args.items():
            if k == 'bbox':
                for _k, _v in _model_args[k].items():
                    _model_args[k][_k] = _v.to('cuda')
            elif k == "mv_order_map" or k == "t_order_map":
                pass
            else:
                _model_args[k] = v.to('cuda')

    with open('/kaiwu_vepfs/kaiwu/huangxinchi/test_inputs/t.pkl', 'rb') as f:
        t = pickle.load(f)
        t = t.to('cuda')
    with open('/kaiwu_vepfs/kaiwu/huangxinchi/test_inputs/z.pkl', 'rb') as f:
        z = pickle.load(f)
        z = z.to('cuda')
    with open('/kaiwu_vepfs/kaiwu/huangxinchi/test_inputs/a_res.pkl', 'rb') as f:
        a_res = pickle.load(f) 
    
    model_path = '/kaiwu_vepfs/kaiwu/xujin2/code_hsy/magicdrivedit/outputs/zhiji_0509/MagicDriveSTDiT3-XL-2_zhiji_0509_20250513-0620/epoch0-global_step512/ema.pt'
    model_state_dict = torch.load(model_path, map_location="cuda")
    for k, v in model_state_dict.items():
        model_state_dict[k] = v.bfloat16()
    
    with open('/kaiwu_vepfs/kaiwu/huangxinchi/lightx2v/configs/mgcdr/config.json', 'r', encoding='UTF-8') as f:
        config = json.load(f)
    infer_config = {
        'attention_type': 'flash_attn2_base',
        'mm_config': {}
    }
    config.update(infer_config)
    
    pre_weights = MagicDrivePreWeights(config=config)
    pre_infer = MagicDrivePreInfer(config=config)
    pre_weights.load(model_state_dict)

    transformer_weights = MagicDriveTransformerWeight(config)
    transfomer_infer = MagicDriveTransformerInfer(config)
    transfomer_infer.set_coordinator(coordinator)
    transformer_weights.load(model_state_dict)
    
    post_weights = MagicDrivePostWeights(config)
    post_infer = MagicDrivePostInfer(config)
    post_weights.load(model_state_dict)
    
    # x = run(z, t, _model_args, pre_weights, pre_infer, transformer_weights, transfomer_infer, post_weights, post_infer)
    
    x_sp = run_sp(z, t, _model_args, pre_weights, pre_infer, transformer_weights, transfomer_infer, post_weights, post_infer)
    
    if dist.get_rank() == 1:
        import pdb; pdb.set_trace()
    else:
        import time; time.sleep(9999)    
    
    print()
    

def run(z, t, _model_args, pre_weights, pre_infer, transformer_weights, transfomer_infer, post_weights, post_infer):
    x, y, c, t, t_mlp, y_lens, x_mask, t0, t0_mlp, T, H, W, S, NC, Tx, Hx, Wx, mv_order_map = pre_infer.infer(pre_weights, x=z, timestep=t, **_model_args)
    x = transfomer_infer.infer(transformer_weights, x, y, c, t_mlp, y_lens, x_mask, t0_mlp, T, S, NC, mv_order_map)
    return x
    # x.shape: torch.Size([7, 34816, 1152])
    # x = post_infer.infer(post_weights, x, t, x_mask, t0, S, NC, T, H, W, Tx, Hx, Wx)
    # torch.save(x, 'x.pt')
    

def run_sp(z, t, _model_args, pre_weights, pre_infer, transformer_weights, transfomer_infer, post_weights, post_infer):
    x, y, c, t, t_mlp, y_lens, x_mask, t0, t0_mlp, T, H, W, S, NC, Tx, Hx, Wx, mv_order_map = pre_infer.infer(pre_weights, x=z, timestep=t, **_model_args)
    
    if coordinator.is_seq_parallel():
        x = sequence_split_before_transformer_infer(x, group=coordinator.seq_group, dim=2)
        c = sequence_split_before_transformer_infer(c, group=coordinator.seq_group, dim=2)
    x = transfomer_infer.infer(transformer_weights, x, y, c, t_mlp, y_lens, x_mask, t0_mlp, T, S, NC, mv_order_map)
    if coordinator.is_seq_parallel():
        x = sequence_gather_after_transformer_infer(x, group=coordinator.seq_group, dim=1)
        c = sequence_gather_after_transformer_infer(c, group=coordinator.seq_group, dim=1)
    return x
    # x = post_infer.infer(post_weights, x, t, x_mask, t0, S, NC, T, H, W, Tx, Hx, Wx)


if __name__ == "__main__":
    main()

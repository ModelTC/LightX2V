from lightx2v.common.ops import *
import json
import torch
import pickle
from lightx2v.models.networks.mgcdr.infer.pre_infer import MagicDrivePreInfer
from lightx2v.models.networks.mgcdr.weights.pre_weights import MagicDrivePreWeights
from lightx2v.models.networks.mgcdr.infer.transformer_infer import MagicDriveTransformerInfer
from lightx2v.models.networks.mgcdr.weights.transformer_weights import MagicDriveTransformerWeight
from lightx2v.models.networks.mgcdr.infer.post_infer import MagicDrivePostInfer
from lightx2v.models.networks.mgcdr.weights.post_weights import MagicDrivePostWeights


def main():
    with open('/kaiwu_vepfs/kaiwu/huangxinchi/test_inputs/_model_args.pkl', 'rb') as f:
        _model_args = pickle.load(f)
    with open('/kaiwu_vepfs/kaiwu/huangxinchi/test_inputs/t.pkl', 'rb') as f:
        t = pickle.load(f)
    with open('/kaiwu_vepfs/kaiwu/huangxinchi/test_inputs/z.pkl', 'rb') as f:
        z = pickle.load(f)
    with open('/kaiwu_vepfs/kaiwu/huangxinchi/test_inputs/a_res.pkl', 'rb') as f:
        a_res = pickle.load(f) 
    
    model_path = '/kaiwu_vepfs/kaiwu/xujin2/code_hsy/magicdrivedit/outputs/zhiji_0509/MagicDriveSTDiT3-XL-2_zhiji_0509_20250513-0620/epoch0-global_step512/ema.pt'
    model_state_dict = torch.load(model_path, map_location='cuda')
    for k, v in model_state_dict.items():
        model_state_dict[k] = v.bfloat16()
    # import pdb; pdb.set_trace()
    with open('/kaiwu_vepfs/kaiwu/huangxinchi/lightx2v/configs/mgcdr/config.json', 'r', encoding='UTF-8') as f:
        config = json.load(f)
    infer_config = {
        'attention_type': 'flash_attn2_base',
        'mm_config': {}
    }
    config.update(infer_config)
    
    # import pdb; pdb.set_trace()
    
    pre_weights = MagicDrivePreWeights(config=config)
    pre_infer = MagicDrivePreInfer(config=config)
    
    pre_weights.load(model_state_dict)
    pre_weights.bbox_embedder_attn.load(model_state_dict)
    # pre_weights.bbox_embedder_attn.set_rotary_emb()
    pre_weights.frame_embedder_attn.load(model_state_dict)
    # pre_weights.frame_embedder_attn.set_rotary_emb()

    x, y, c, t, t_mlp, y_lens, x_mask, t0, t0_mlp, T, H, W, S, NC, Tx, Hx, Wx, mv_order_map = pre_infer.infer(pre_weights, x=z, timestep=t, **_model_args)
    
    
    
    
    transformer_weights = MagicDriveTransformerWeight(config)
    transofmer_infer = MagicDriveTransformerInfer(config)
    transformer_weights.load(model_state_dict)
    import pdb; pdb.set_trace()
    x = transofmer_infer.infer(transformer_weights, x, y, c, t_mlp, y_lens, x_mask, t0_mlp, T, S, NC, mv_order_map)
    
    
    
    post_weights = MagicDrivePostWeights(config)
    post_infer = MagicDrivePostInfer(config)
    post_weights.load(model_state_dict)
    x = post_infer.infer(post_weights, x, t, x_mask, t0, S, NC, T, H, W, Tx, Hx, Wx)
    
    
    

if __name__ == "__main__":
    main()

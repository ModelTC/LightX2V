import torch
import functools
import torch.nn as nn
from typing import Optional
from rotary_embedding_torch import RotaryEmbedding
from lightx2v.utils.registry_factory import (
    MM_WEIGHT_REGISTER,
    LN_WEIGHT_REGISTER,
    RMS_WEIGHT_REGISTER,
    CONV3D_WEIGHT_REGISTER,
    CONV2D_WEIGHT_REGISTER,
    TENSOR_REGISTER,
    ATTN_WEIGHT_REGISTER
)
from lightx2v.common.modules.weight_module import WeightModule


class FourierEmbedder:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            if self.kwargs['norm_x']:
                embed_fns.append(lambda x: x / 1000.0)
            else:
                embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class PositionEmbedding2D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0, 'dim must be divisible by 4'
        half_dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def _get_sin_cos_emb(self, t: torch.Tensor):
        out = torch.einsum('i,d->id', t, self.inv_freq.to(t.device))
        emb_cos = torch.cos(out)
        emb_sin = torch.sin(out)
        return torch.cat((emb_sin, emb_cos), dim=-1)

    @functools.lru_cache(maxsize=512)
    def _get_cached_emb(
        self,
        device: torch.device,
        dtype: torch.dtype,
        h: int,
        w: int,
        scale: float = 1.0,
        base_size: Optional[int] = None,
    ):
        grid_h = torch.arange(h, device=device) / scale
        grid_w = torch.arange(w, device=device) / scale
        if base_size is not None:
            grid_h *= base_size / h
            grid_w *= base_size / w
        grid_h, grid_w = torch.meshgrid(
            grid_w,
            grid_h,
            indexing='ij',
        )  # here w goes first
        grid_h = grid_h.t().reshape(-1)
        grid_w = grid_w.t().reshape(-1)
        emb_h = self._get_sin_cos_emb(grid_h)
        emb_w = self._get_sin_cos_emb(grid_w)
        return torch.concat([emb_h, emb_w], dim=-1).unsqueeze(0).to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        h: int,
        w: int,
        scale: Optional[float] = 1.0,
        base_size: Optional[int] = None,
    ) -> torch.Tensor:
        return self._get_cached_emb(x.device, x.dtype, h, w, scale, base_size)


class MagicDriveSelfAttention(WeightModule):
    def __init__(self, config, prefix=''):
        super().__init__()
        self.config = config
        param_prefix = f'{prefix}_param'
        if param_prefix in config and 'num_heads' in config[param_prefix]:
            self.num_heads = config[param_prefix]['num_heads']
        else:
            self.num_heads = config['num_heads']
        
        if param_prefix in config and 'hidden_size' in config[param_prefix]:
            self.hidden_size = config[param_prefix]['hidden_size']
        else:
            self.hidden_size = config['hidden_size']
            
        self.head_dim = self.hidden_size // self.num_heads
        self.attn_drop = 0.0
        self.softmax_scale = self.head_dim**-0.5
        self.is_causal = False
        
        self.add_module(
            'attn_qkv',
            MM_WEIGHT_REGISTER['Flinear'](f'{prefix}.attn.qkv.weight', f'{prefix}.attn.qkv.bias')
        )
        self.add_module(
            'attn_q_norm',
            RMS_WEIGHT_REGISTER['Mgcdr'](f'{prefix}.attn.q_norm.weight')
        )
        self.add_module(
            'attn_k_norm',
            RMS_WEIGHT_REGISTER['Mgcdr'](f'{prefix}.attn.k_norm.weight')
        )
        self.add_module(
            'attn',
            ATTN_WEIGHT_REGISTER['flash_attn2_base']()
        )
        self.add_module(
            'attn_proj',
            MM_WEIGHT_REGISTER['Flinear'](f'{prefix}.attn.proj.weight', f'{prefix}.attn.proj.bias')
        )
        
        self.register_parameter(
            'rope_freqs',
            TENSOR_REGISTER['Default'](f'{prefix}.rope.freqs')
        )
        
    def load(self, weight_dict):
        super().load(weight_dict)
        self._set_rotary_emb()
        
    def _set_rotary_emb(self):
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim, custom_freqs=self.rope_freqs.tensor)
        

class MagicDrivePreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.pos_embedding = PositionEmbedding2D(self.config.get('hidden_size')).to(torch.device('cuda'), torch.bfloat16)
        fourier_kwargs = {'input_dims': 3, 'num_freqs': 4, 'max_freq_log2': 3, 'include_input': True, 'norm_x': False, 'log_sampling': True, 'periodic_fns': [torch.sin, torch.cos]}
        self.fourier_embedder = FourierEmbedder(**fourier_kwargs)
        
        self.bbox_embedder_attn = MagicDriveSelfAttention(config=config, prefix='bbox_embedder')
        self.frame_embedder_attn = MagicDriveSelfAttention(config=config, prefix='frame_embedder')
        
        self.add_module(
            'bbox_embedder_attn', self.bbox_embedder_attn
        )
        self.add_module(
            'frame_embedder_attn', self.frame_embedder_attn
        )
        
        self.add_module(
            't_embedder_0',
            MM_WEIGHT_REGISTER['Flinear']('t_embedder.mlp.0.weight', 't_embedder.mlp.0.bias')
        )
        self.add_module(
            't_embedder_2',
            MM_WEIGHT_REGISTER['Flinear']('t_embedder.mlp.2.weight', 't_embedder.mlp.2.bias')
        )

        self.add_module(
            't_block_1',
            MM_WEIGHT_REGISTER['Flinear']('t_block.1.weight', 't_block.1.bias')
        )

        self.add_module(
            'fps_embedder_0',
            MM_WEIGHT_REGISTER['Flinear']('fps_embedder.mlp.0.weight', 'fps_embedder.mlp.0.bias')
        )
        self.add_module(
            'fps_embedder_2',
            MM_WEIGHT_REGISTER['Flinear']('fps_embedder.mlp.2.weight', 'fps_embedder.mlp.2.bias')
        )

        self.add_module(
            'y_embedder_fc1',
            MM_WEIGHT_REGISTER['Flinear']('y_embedder.y_proj.fc1.weight', 'y_embedder.y_proj.fc1.bias')
        )
        self.add_module(
            'y_embedder_fc2',
            MM_WEIGHT_REGISTER['Flinear']('y_embedder.y_proj.fc2.weight', 'y_embedder.y_proj.fc2.bias')
        )

        self.add_module(
            'bbox_embedder_bbox_proj',
            MM_WEIGHT_REGISTER['Flinear']('bbox_embedder.bbox_proj.weight', 'bbox_embedder.bbox_proj.bias')
        )
        self.add_module(
            'bbox_embedder_second_linear_0',
            MM_WEIGHT_REGISTER['Flinear']('bbox_embedder.second_linear.0.weight', 'bbox_embedder.second_linear.0.bias')
        )
        self.add_module(
            'bbox_embedder_second_linear_2',
            MM_WEIGHT_REGISTER['Flinear']('bbox_embedder.second_linear.2.weight', 'bbox_embedder.second_linear.2.bias')
        )
        self.add_module(
            'bbox_embedder_second_linear_4',
            MM_WEIGHT_REGISTER['Flinear']('bbox_embedder.second_linear.4.weight', 'bbox_embedder.second_linear.4.bias')
        )
        self.add_module(
            'bbox_embedder_norm1',
            LN_WEIGHT_REGISTER['Apex']()
        )
        self.add_module(
            'bbox_embedder_norm2',
            LN_WEIGHT_REGISTER['Apex']()
        )
        self.add_module(
            'bbox_embedder_mlp_fc1',
            MM_WEIGHT_REGISTER['Flinear']('bbox_embedder.mlp.fc1.weight', 'bbox_embedder.mlp.fc1.bias')
        )
        self.add_module(
            'bbox_embedder_mlp_fc2',
            MM_WEIGHT_REGISTER['Flinear']('bbox_embedder.mlp.fc2.weight', 'bbox_embedder.mlp.fc2.bias')
        )
        self.add_module(
            'bbox_embedder_final_proj',
            MM_WEIGHT_REGISTER['Flinear']('bbox_embedder.final_proj.weight', 'bbox_embedder.final_proj.bias')
        )
        
        self.add_module(
            'camera_embedder_emb2token',
            MM_WEIGHT_REGISTER['Flinear']('camera_embedder.emb2token.weight', 'camera_embedder.emb2token.bias')
        )
        self.add_module(
            'camera_embedder_after_proj',
            MM_WEIGHT_REGISTER['Flinear']('camera_embedder.after_proj.weight', 'camera_embedder.after_proj.bias')
        )
        
        self.add_module(
            'frame_embedder_emb2token',
            MM_WEIGHT_REGISTER['Flinear']('frame_embedder.emb2token.weight', 'frame_embedder.emb2token.bias')
        )
        self.add_module(
            'frame_embedder_norm1',
            LN_WEIGHT_REGISTER['Apex']()
        )
        self.add_module(
            'frame_embedder_norm2',
            LN_WEIGHT_REGISTER['Apex']()
        )
        self.add_module(
            'frame_embedder_mlp_fc1',
            MM_WEIGHT_REGISTER['Flinear']('frame_embedder.mlp.fc1.weight', 'frame_embedder.mlp.fc1.bias')
        )
        self.add_module(
            'frame_embedder_mlp_fc2',
            MM_WEIGHT_REGISTER['Flinear']('frame_embedder.mlp.fc2.weight', 'frame_embedder.mlp.fc2.bias')
        )
        self.add_module(
            'frame_embedder_final_proj',
            MM_WEIGHT_REGISTER['Flinear']('frame_embedder.final_proj.weight', 'frame_embedder.final_proj.bias')
        )
        
        self.add_module(
            'controlnet_cond_embedder_conv_in',
            CONV2D_WEIGHT_REGISTER['Default']('controlnet_cond_embedder.conv_in.weight', 'controlnet_cond_embedder.conv_in.bias', stride=(1,1), padding=(1,1))
        )
        self.add_module(
            'controlnet_cond_embedder_blocks_0',
            CONV2D_WEIGHT_REGISTER['Default']('controlnet_cond_embedder.blocks.0.weight', 'controlnet_cond_embedder.blocks.0.bias', stride=(1,1), padding=(1,1))
        )
        self.add_module(
            'controlnet_cond_embedder_blocks_1',
            CONV2D_WEIGHT_REGISTER['Default']('controlnet_cond_embedder.blocks.1.weight', 'controlnet_cond_embedder.blocks.1.bias', stride=(2,2), padding=(1,1))
        )
        self.add_module(
            'controlnet_cond_embedder_blocks_2',
            CONV2D_WEIGHT_REGISTER['Default']('controlnet_cond_embedder.blocks.2.weight', 'controlnet_cond_embedder.blocks.2.bias', stride=(1,1), padding=(1,1))
        )
        self.add_module(
            'controlnet_cond_embedder_blocks_3',
            CONV2D_WEIGHT_REGISTER['Default']('controlnet_cond_embedder.blocks.3.weight', 'controlnet_cond_embedder.blocks.3.bias', stride=(2,2), padding=(1,1))
        )
        self.add_module(
            'controlnet_cond_embedder_blocks_4',
            CONV2D_WEIGHT_REGISTER['Default']('controlnet_cond_embedder.blocks.4.weight', 'controlnet_cond_embedder.blocks.4.bias', stride=(1,1), padding=(1,1))
        )
        self.add_module(
            'controlnet_cond_embedder_blocks_5',
            CONV2D_WEIGHT_REGISTER['Default']('controlnet_cond_embedder.blocks.5.weight', 'controlnet_cond_embedder.blocks.5.bias', stride=(2,2), padding=(1,1))
        )
        self.add_module(
            'controlnet_cond_embedder_conv_out',
            CONV2D_WEIGHT_REGISTER['Default']('controlnet_cond_embedder.conv_out.weight', 'controlnet_cond_embedder.conv_out.bias', stride=(1,1), padding=(1,1))
        )
        self.add_module(
            'controlnet_cond_embedder_temp_conv_blocks_1_conv',
            CONV2D_WEIGHT_REGISTER['Default']('controlnet_cond_embedder_temp.conv_blocks.1.conv.weight', 'controlnet_cond_embedder_temp.conv_blocks.1.conv.bias', stride=(1,1))
        )
        self.add_module(
            'controlnet_cond_embedder_temp_conv_blocks_3_conv',
            CONV2D_WEIGHT_REGISTER['Default']('controlnet_cond_embedder_temp.conv_blocks.3.conv.weight', 'controlnet_cond_embedder_temp.conv_blocks.3.conv.bias', stride=(1,1))
        )
        self.add_module(
            'controlnet_cond_patchifier_proj',
            CONV3D_WEIGHT_REGISTER['Default']('controlnet_cond_patchifier.proj.weight', 'controlnet_cond_patchifier.proj.bias', stride=(1,2,2))
        )
        self.add_module(
            'x_embedder_proj',
            CONV3D_WEIGHT_REGISTER['Default']('x_embedder.proj.weight', 'x_embedder.proj.bias', stride=(1,2,2))
        )
        self.add_module(
            'x_control_embedder_proj',
            CONV3D_WEIGHT_REGISTER['Default']('x_control_embedder.proj.weight', 'x_control_embedder.proj.bias', stride=(1,2,2))
        )
        self.add_module(
            'before_proj',
            MM_WEIGHT_REGISTER['Flinear']('before_proj.weight', 'before_proj.bias')
        )

        self.register_parameter(
            'y_embedder_y_embedding', 
            TENSOR_REGISTER['Default']('y_embedder.y_embedding')
        )
        self.register_parameter(
            'bbox_embedder_class_tokens',
            TENSOR_REGISTER['Default']('bbox_embedder._class_tokens')
        )
        self.register_parameter(
            'bbox_embedder_mean_var',
            TENSOR_REGISTER['Default']('bbox_embedder.mean_var')
        )
        self.register_parameter(
            'bbox_embedder_null_class_feature',
            TENSOR_REGISTER['Default']('bbox_embedder.null_class_feature')
        )
        self.register_parameter(
            'bbox_embedder_mask_class_feature',
            TENSOR_REGISTER['Default']('bbox_embedder.mask_class_feature')
        )
        self.register_parameter(
            'bbox_embedder_null_pos_feature',
            TENSOR_REGISTER['Default']('bbox_embedder.null_pos_feature')
        )
        self.register_parameter(
            'bbox_embedder_mask_pos_feature',
            TENSOR_REGISTER['Default']('bbox_embedder.mask_pos_feature')
        )
        self.register_parameter(
            'bbox_embedder_scale_shift_table',
            TENSOR_REGISTER['Default']('bbox_embedder.scale_shift_table')
        )
        
        self.register_parameter(
            'base_token',
            TENSOR_REGISTER['Default']('base_token')
        )
        self.register_parameter(
            'camera_embedder_uncond_cam',
            TENSOR_REGISTER['Default']('camera_embedder.uncond_cam')
        )
        self.register_parameter(
            'frame_embedder_scale_shift_table',
            TENSOR_REGISTER['Default']('frame_embedder.scale_shift_table')
        )
        self.register_parameter(
            'frame_embedder_uncond_cam',
            TENSOR_REGISTER['Default']('frame_embedder.uncond_cam')
        )
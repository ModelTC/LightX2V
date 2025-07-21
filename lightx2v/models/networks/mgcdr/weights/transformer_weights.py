import torch
import os
from lightx2v.utils.registry_factory import (
    MM_WEIGHT_REGISTER,
    LN_WEIGHT_REGISTER,
    RMS_WEIGHT_REGISTER,
    TENSOR_REGISTER,
    ATTN_WEIGHT_REGISTER,
)
from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from rotary_embedding_torch import RotaryEmbedding


class MagicDriveTransformerAttnBlock(WeightModule):
    def __init__(self, block_idx, config, prefix, tensor_name_list, mm_name_list, mm_nobias_name_list,  ln_name_list, rms_name_list):
        super().__init__()
        self.num_heads = config['num_heads']
        self.hidden_size = config['hidden_size']
        self.head_dim = self.hidden_size // self.num_heads
        self.attn_drop = 0.0
        self.softmax_scale = self.head_dim**-0.5
        self.is_causal = False

        self.ori_weight_prefix = f"{prefix}.{block_idx}"
        # self.new_weight_prefix = f"{prefix}_{block_idx}"
        self.block_index = block_idx
        self.config = config
        self.tensor_name_list = tensor_name_list
        self.mm_name_list = mm_name_list
        self.mm_nobias_name_list = mm_nobias_name_list
        self.ln_name_list = ln_name_list
        self.rms_name_list = rms_name_list
        
        self._init_parameters()
        self._init_modules()
        
        self.add_module(
            'attn',
            ATTN_WEIGHT_REGISTER['flash_attn2_base']()
        )
        self.add_module(
            'attn_xformers',
            ATTN_WEIGHT_REGISTER['xformers']()
        )
        self.add_module(
            'attn_naive',
            ATTN_WEIGHT_REGISTER['naive']()
        )
        self.register_parameter(
            'rope_freqs',
            TENSOR_REGISTER['Default'](f'rope.freqs')
        )
        
    def load(self, weight_dict):
        super().load(weight_dict)
        self._set_rotary_emb()
    
    def _set_rotary_emb(self):
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim, custom_freqs=self.rope_freqs.tensor)

    def _init_parameters(self):
        # import pdb; pdb.set_trace()
        for tensor_name in self.tensor_name_list:
            lightx2v_tensor_name = tensor_name.replace('.', '_')
            self.register_parameter(
                f'{lightx2v_tensor_name}',
                TENSOR_REGISTER['Default'](f'{self.ori_weight_prefix}.{tensor_name}')
            )

    def _init_modules(self):
        for mm_name in self.mm_name_list:
            lightx2v_mm_name = mm_name.replace('.', '_')
            self.add_module(
                f'{lightx2v_mm_name}',
                MM_WEIGHT_REGISTER['Flinear'](f'{self.ori_weight_prefix}.{mm_name}.weight', f'{self.ori_weight_prefix}.{mm_name}.bias')
            )
        for mm_name in self.mm_nobias_name_list:
            lightx2v_mm_name = mm_name.replace('.', '_')
            self.add_module(
                f'{lightx2v_mm_name}',
                MM_WEIGHT_REGISTER['Flinear'](f'{self.ori_weight_prefix}.{mm_name}.weight', None)
            )
        for ln_name in self.ln_name_list:
            lightx2v_ln_name = ln_name.replace('.', '_')
            self.add_module(
                f'{lightx2v_ln_name}',
                LN_WEIGHT_REGISTER['Apex']()
        )
        for rms_name in self.rms_name_list:
            lightx2v_rms_name = rms_name.replace('.', '_')
            self.add_module(
                f'{lightx2v_rms_name}',
                RMS_WEIGHT_REGISTER['Mgcdr'](f'{self.ori_weight_prefix}.{rms_name}.weight')
        )

class MagicDriveTransformerWeight(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base_blocks_s_num = self.config.get('base_blocks_s_num', 28)
        self.base_blocks_s_prefix = "base_blocks_s"
        self.base_blocks_t_num = self.config.get('base_blocks_t_num', 28)
        self.base_blocks_t_prefix = "base_blocks_t"
        self.control_blocks_s_num = self.config.get('control_blocks_s_num', 13)
        self.control_blocks_s_prefix = "control_blocks_s"
        self.control_blocks_t_num = self.config.get('control_blocks_t_num', 13)
        self.control_blocks_t_prefix = "control_blocks_t"
        
        self._init_base_blocks_s()
        self._init_base_blocks_t()
        self._init_control_blocks_s()
        self._init_control_blocks_t()
        
        self.add_module(
            'base_blocks_s_list', self.base_blocks_s_list
        )
        self.add_module(
            'base_blocks_t_list', self.base_blocks_t_list
        )
        self.add_module(
            'control_blocks_s_list', self.control_blocks_s_list
        )
        self.add_module(
            'control_blocks_t_list', self.control_blocks_t_list
        )
        
    def load(self, weight_dict):
        super().load(weight_dict)
        # self._split_cross_view_attn_qkv_linear_weight()
        self._split_attn_qkv_linear_weight()
        
    def _split_attn_qkv_linear_weight(self):
        for module in self.base_blocks_s_list:
            # attn_qkv_weight = module.attn_qkv.weight.t()
            # attn_qkv_bias = module.attn_qkv.bias
            cross_view_attn_qkv_weight = module.cross_view_attn_qkv.weight
            tiny_dict = {
                # 'attn.q.weight':attn_qkv_weight[:module.hidden_size, :],
                # 'attn.q.bias': attn_qkv_bias[:module.hidden_size],
                # 'attn.kv.weight': attn_qkv_weight[module.hidden_size:, :],
                # 'attn.kv.bias': attn_qkv_bias[module.hidden_size:],
                'cross_view_attn.q.weight': cross_view_attn_qkv_weight[:module.hidden_size, :],
                'cross_view_attn.kv.weight': cross_view_attn_qkv_weight[module.hidden_size:, :]
            }
            # module.add_module(
            #     'attn_q',
            #     MM_WEIGHT_REGISTER['Flinear']('attn.q.weight', 'attn.q.bias')
            # )
            # module.add_module(
            #     'attn_kv',
            #     MM_WEIGHT_REGISTER['Flinear']('attn.kv.weight', 'attn.kv.bias')
            # )
            module.add_module(
                'cross_view_attn_q',
                MM_WEIGHT_REGISTER['Flinear']('cross_view_attn.q.weight', None)
            )
            module.add_module(
                'cross_view_attn_kv',
                MM_WEIGHT_REGISTER['Flinear']('cross_view_attn.kv.weight', None)
            )
            # module.attn_q.load(tiny_dict)
            # module.attn_kv.load(tiny_dict)
            module.cross_view_attn_q.load(tiny_dict)
            module.cross_view_attn_kv.load(tiny_dict)
        
    def _init_base_blocks_s(self):
        tensor_name_list = ['scale_shift_table', 'scale_shift_table_mva', ]
        mm_name_list = ['attn.qkv', 'attn.proj', 'cross_attn.q_linear', 'cross_attn.kv_linear', 'cross_attn.proj', 'mlp.fc1', 'mlp.fc2', 'cross_view_attn.proj', 'mva_proj']
        mm_nobias_name_list = ['cross_view_attn.qkv']
        ln_name_list = ['norm1', 'norm2', 'norm3']
        rms_name_list = ['attn.q_norm', 'attn.k_norm', 'cross_view_attn.q_norm', 'cross_view_attn.k_norm']
        self.base_blocks_s_list = WeightModuleList(MagicDriveTransformerAttnBlock(i, self.config, self.base_blocks_s_prefix, tensor_name_list=tensor_name_list, mm_name_list=mm_name_list, mm_nobias_name_list=mm_nobias_name_list, ln_name_list=ln_name_list, rms_name_list=rms_name_list) for i in range(self.base_blocks_s_num))
    
    def _init_base_blocks_t(self):
        tensor_name_list = ['scale_shift_table']
        mm_name_list = ['attn.qkv', 'attn.proj', 'cross_attn.q_linear', 'cross_attn.kv_linear', 'cross_attn.proj', 'mlp.fc1', 'mlp.fc2']
        mm_nobias_name_list = []
        ln_name_list = ['norm1', 'norm2', 'norm3']
        rms_name_list = ['attn.q_norm', 'attn.k_norm']
        self.base_blocks_t_list = WeightModuleList(MagicDriveTransformerAttnBlock(i, self.config, self.base_blocks_t_prefix, tensor_name_list=tensor_name_list, mm_name_list=mm_name_list, mm_nobias_name_list=mm_nobias_name_list, ln_name_list=ln_name_list, rms_name_list=rms_name_list) for i in range(self.base_blocks_t_num))
    
    def _init_control_blocks_s(self):
        tensor_name_list = ['scale_shift_table']
        mm_name_list = ['attn.qkv', 'attn.proj', 'cross_attn.q_linear', 'cross_attn.kv_linear', 'cross_attn.proj', 'mlp.fc1', 'mlp.fc2', 'after_proj']
        mm_nobias_name_list = []
        ln_name_list = ['norm1', 'norm2']
        rms_name_list = ['attn.q_norm', 'attn.k_norm']
        self.control_blocks_s_list = WeightModuleList(MagicDriveTransformerAttnBlock(i, self.config, self.control_blocks_s_prefix, tensor_name_list=tensor_name_list, mm_name_list=mm_name_list, mm_nobias_name_list=mm_nobias_name_list, ln_name_list=ln_name_list, rms_name_list=rms_name_list) for i in range(self.control_blocks_s_num))
        
    def _init_control_blocks_t(self):
        tensor_name_list = ['scale_shift_table']
        mm_name_list = ['attn.qkv', 'attn.proj', 'cross_attn.q_linear', 'cross_attn.kv_linear', 'cross_attn.proj', 'mlp.fc1', 'mlp.fc2', 'after_proj']
        mm_nobias_name_list = []
        ln_name_list = ['norm1', 'norm2']
        rms_name_list = ['attn.q_norm', 'attn.k_norm']
        self.control_blocks_t_list = WeightModuleList(MagicDriveTransformerAttnBlock(i, self.config, self.control_blocks_t_prefix, tensor_name_list=tensor_name_list, mm_name_list=mm_name_list, mm_nobias_name_list=mm_nobias_name_list, ln_name_list=ln_name_list, rms_name_list=rms_name_list) for i in range(self.control_blocks_t_num))
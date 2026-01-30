from dataclasses import dataclass
from typing import Optional


class NaiveCache:
    def __init__(self, num_layers):
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}

    @property
    def num_layers(self):
        return len(self.key_cache)

    @property
    def seq_lens(self):
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        else:
            return 0


# Modified from https://github.com/Shenyi-Z/TaylorSeer/blob/main/TaylorSeers-xDiT/taylorseer_flux/cache_functions/cache_init.py
def cache_init(self, num_steps: int):
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1]={}
    cache_index[-1]={}
    cache_index['layer_index']={}
    cache[-1]['layers_stream']={}
    cache_dic['cache_counter'] = 0

    for j in range(len(self.language_model.model.layers)):
        cache[-1]['layers_stream'][j] = {}
        cache_index[-1][j] = {}

    cache_dic['Delta-DiT'] = False
    cache_dic['cache_type'] = 'random'
    cache_dic['cache_index'] = cache_index
    cache_dic['cache'] = cache
    cache_dic['fresh_ratio_schedule'] = 'ToCa' 
    cache_dic['fresh_ratio'] = 0.0
    cache_dic['fresh_threshold'] = 3
    cache_dic['soft_fresh_weight'] = 0.0
    cache_dic['taylor_cache'] = True
    cache_dic['max_order'] = 6
    cache_dic['first_enhance'] = 5

    current = {}
    current['activated_steps'] = [0]
    current['step'] = 0
    current['num_steps'] = num_steps

    return cache_dic, current



@dataclass
class BagelInputs:
    image_shapes: tuple = None
    gen_context: dict = None
    cfg_text_precontext: dict = None
    cfg_img_precontext: dict = None
    model_pred_cache_dic: dict = None
    model_pred_current: dict = None
    model_pred_text_cache_dic: dict = None
    model_pred_text_current: dict = None
    model_pred_img_cache_dic: dict = None
    model_pred_img_current: dict = None
    generation_input: dict = None
    generation_input_cfg_text: dict = None
    generation_input_cfg_img: dict = None
    cfg_text_past_key_values: Optional[NaiveCache] = None
    cfg_img_past_key_values: Optional[NaiveCache] = None

    
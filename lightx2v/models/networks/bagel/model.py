import gc
import os
from copy import deepcopy
from PIL import Image

import torch
from torch.nn import functional as F
import math

from lightx2v.utils.envs import *
from lightx2v.utils.utils import *

from lightx2v.models.networks.bagel.weights.pre_weights import Qwen2PreWeights
from lightx2v.models.networks.bagel.weights.transformer_weights import Qwen2TransformerWeights
from lightx2v.models.networks.bagel.weights.post_weights import Qwen2PostWeights
from lightx2v.models.networks.bagel.infer.pre_infer import BagelPreInfer
from lightx2v.models.networks.bagel.infer.transformer_infer import BagelTransformerInfer
from lightx2v.models.networks.bagel.infer.post_infer import BagelPostInfer
from lightx2v.models.networks.bagel.model_io import BagelInputs, NaiveCache, cache_init
from lightx2v.models.networks.bagel.tokenization_qwen2 import Qwen2Tokenizer
from lightx2v.models.networks.bagel.modeling_utils import PositionEmbedding
from lightx2v.models.networks.bagel.data_utils import add_special_tokens

VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''


class BagelModel:
    pre_weight_class = Qwen2PreWeights
    transformer_weight_class = Qwen2TransformerWeights
    post_weight_class = Qwen2PostWeights
    
    def __init__(self, config):
        self.config = config
        self.model_path = config['model_path']
        # init llm config
        llm_config = self.config['llm_config']
        with self.config.temporarily_unlocked():
            llm_config.update(
                self.config['llm_config_update']
            )
        self.llm_config = llm_config
        self.use_moe = "Mo" in self.llm_config["layer_module"]
        self.num_heads = self.llm_config['num_attention_heads'] 
        self.hidden_size = self.llm_config['hidden_size']
        self.think = config.get("think", False)
        self.understanding_output = config.get("understanding_output", False)
        self.inference_hyper = config['inference_hyper']
        self.do_sample = config.get('do_sample', False)
        self.text_temperature = config.get("text_temperature", 0.3)
        self.max_think_token_n = config.get("max_think_token_n", 1000)
        self.enable_taylorseer = False
        
        self.cpu_offload = config.get("cpu_offload", False)
        self.offload_granularity = self.config.get("offload_granularity", "block")
        self.device = torch.device("cpu") if self.cpu_offload else torch.device(AI_DEVICE)
        self._init_infer_class()
        self._init_weights()
        self._init_infer()
        self._init_modules()
        
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.pre_infer.set_scheduler(scheduler)
        self.transformer_infer.set_scheduler(scheduler)
        self.post_infer.set_scheduler(scheduler)
    
    def _init_infer_class(self):
        self.pre_infer_class = BagelPreInfer
        self.transformer_infer_class = BagelTransformerInfer
        self.post_infer_class = BagelPostInfer
    
    def _apply_weights(self, weight_dict=None):
        if weight_dict is not None:
            self.original_weight_dict = weight_dict
            del weight_dict
            gc.collect()
        # Load weights into containers
        self.pre_weight.load(self.original_weight_dict)
        self.transformer_weights.load(self.original_weight_dict)
        self.post_weight.load(self.original_weight_dict)

        del self.original_weight_dict
        torch.cuda.empty_cache()
        gc.collect()
    
    def _init_weights(self):
        self.pre_weight = self.pre_weight_class(self.config)
        self.transformer_weights = self.transformer_weight_class(self.config, self.llm_config)
        self.post_weight = self.post_weight_class(self.config)
        weight_dict = safetensors.torch.load_file(os.path.join(self.config['model_path'], "ema.safetensors"), device=AI_DEVICE)
        self._apply_weights(weight_dict)
    
    def _init_infer(self):
        self.transformer_infer = self.transformer_infer_class(self.config, self.llm_config)
        self.pre_infer = self.pre_infer_class(self.config, self.llm_config)
        self.post_infer = self.post_infer_class(self.config, self.llm_config)
    
    def _init_modules(self):
        tokenizer = Qwen2Tokenizer.from_pretrained(self.model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids    

        if self.config.visual_gen:
            self.latent_patch_size = self.config.latent_patch_size
            self.timestep_shift = self.config.timestep_shift
            self.latent_downsample = self.config.vae_config['downsample'] * self.config.latent_patch_size
            
            self.latent_channel = self.config.vae_config['z_channels']
            self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel
            self.max_latent_size = self.config['max_latent_size_update']
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)
            self.frequency_embedding_size=256
            

    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.llm_config.num_hidden_layers),
        }
        return gen_context

    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ):       
        packed_query_position_embeddings = self.pre_infer.infer(
            self.pre_weight,
            packed_query_sequence,
            packed_query_position_ids
        )
        
        extra_inputs = {}
        if self.use_moe:
            extra_inputs.update(mode=mode)
            if mode == 'gen':
                assert packed_vae_token_indexes is not None
                assert packed_text_indexes is not None
                extra_inputs.update(
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_text_indexes=packed_text_indexes,
                )
        
        packed_query_sequence, past_key_values = self.transformer_infer.infer(
            self.transformer_weights.blocks,
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            **extra_inputs,
        )

        packed_query_sequence = self.post_infer.infer(
            self.post_weight,
            packed_query_sequence,
            packed_text_indexes,
            packed_vae_token_indexes,
            mode,
        )
     
        return packed_query_sequence, past_key_values
        
    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):  
        packed_text_embedding = self.pre_infer.embed_tokens(self.pre_weight, packed_text_ids)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        packed_query_sequence, past_key_values = self.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        return past_key_values

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference, 
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input, kv_lens, ropes = self.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )
        past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    def gen_text(self):
        assert NotImplementedError
    
    @torch.no_grad()
    def prepare_inputs(self, input_info, scheduler):
        gen_context = self.transformer_infer.gen_context
        cfg_text_context = self.transformer_infer.cfg_text_context
        cfg_img_context = self.transformer_infer.cfg_img_context
        
        input_lists = [input_info.prompt]
        output_list = []
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if self.think:
                if self.understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)
            for input_term in input_lists:
                if isinstance(input_term, str):  # True
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context) 
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)
                elif isinstance(input_term, Image.Image):
                    assert NotImplementedError
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")
            
            max_think_token_n=1000
            if self.understanding_output:
                assert NotImplementedError
                gen_text = self.gen_text(gen_context, do_sample=self.do_sample, temperature=self.text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)
            else:
                if self.think:
                    gen_text = self.gen_text(gen_context, do_sample=self.do_sample, temperature=self.text_temperature, max_length=max_think_token_n)
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)
                else:
                    gen_text = None

        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = scheduler.prepare_vae_latent(
            curr_kvlens=kv_lens,             
            curr_rope=ropes,                 
            image_sizes=[(1024, 1024)],        
            new_token_ids=self.new_token_ids, 
        )

        # text cfg
        cfg_text_past_key_values = cfg_text_context['past_key_values']
        kv_lens_cfg = cfg_text_context['kv_lens']
        ropes_cfg = cfg_text_context['ropes']
        generation_input_cfg_text = scheduler.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[(1024, 1024)], 
        )
       

        # img cfg
        cfg_img_past_key_values = cfg_img_context['past_key_values']
        kv_lens_cfg = cfg_img_context['kv_lens']
        ropes_cfg = cfg_img_context['ropes']
        generation_input_cfg_img = scheduler.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[(1024, 1024)], 
        )
        
        scheduler.generation_input = generation_input
        scheduler.generation_input_cfg_text = generation_input_cfg_text
        scheduler.generation_input_cfg_image = generation_input_cfg_img
        scheduler.latents = generation_input['packed_init_noises']

        num_timesteps = scheduler.infer_steps
        if self.enable_taylorseer:
            model_pred_cache_dic, model_pred_current = cache_init(self, num_timesteps)
            model_pred_text_cache_dic, model_pred_text_current = cache_init(self, num_timesteps)
            model_pred_img_cache_dic, model_pred_img_current = cache_init(self, num_timesteps)
        else:
            model_pred_cache_dic, model_pred_current = None, None
            model_pred_text_cache_dic, model_pred_text_current = None, None
            model_pred_img_cache_dic, model_pred_img_current = None, None
                
        bagel_inputs = BagelInputs(
            image_shapes=input_info.image_shapes,
            gen_context=gen_context,
            cfg_text_precontext=cfg_text_context,
            cfg_img_precontext=cfg_img_context,
            model_pred_cache_dic=model_pred_cache_dic,
            model_pred_current=model_pred_current,
            model_pred_text_cache_dic=model_pred_text_cache_dic,
            model_pred_text_current=model_pred_text_current,
            model_pred_img_cache_dic=model_pred_img_cache_dic,
            model_pred_img_current=model_pred_img_current,
            generation_input=generation_input,
            generation_input_cfg_text=generation_input_cfg_text,
            generation_input_cfg_img=generation_input_cfg_img,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
        )
        return bagel_inputs, scheduler

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    @torch.no_grad()
    def time_embedder(self, weights, t):
        t = t.to(AI_DEVICE)
        t = self.timestep_embedding(t, self.frequency_embedding_size)
        t = t.to(torch.bfloat16)
        t = weights.mlp_0.apply(t)
        t = F.silu(t)
        t = weights.mlp_2.apply(t)
        return t

    @torch.no_grad()
    def vae2llm(self, x):
        x = self.pre_infer.vae2llm(self.pre_weight, x)
        return x

    @torch.no_grad()
    def llm2vae(self, x):
        x = self.post_infer.llm2vae(self.post_weight, x)
        return x

    @torch.no_grad
    def infer(self, inputs):
        t = self.scheduler.timesteps[self.scheduler.step_index]
        x_t = self.scheduler.latents.to(torch.bfloat16).to(AI_DEVICE)
        timestep = torch.tensor([t] * x_t.shape[0])
        
        if t > self.inference_hyper['cfg_interval'][0] and t <= self.inference_hyper['cfg_interval'][1]:
            cfg_text_scale = self.inference_hyper['cfg_text_scale']
            cfg_img_scale = self.inference_hyper['cfg_img_scale']
        else:
            cfg_text_scale = 1.0
            cfg_img_scale = 1.0
        
        packed_text_ids = inputs.generation_input['packed_text_ids']
        packed_seqlens = inputs.generation_input['packed_seqlens']
        packed_text_indexes = inputs.generation_input['packed_text_indexes']
        packed_text_embedding = self.pre_infer.embed_tokens(self.pre_weight, packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.latent_pos_embed(inputs.generation_input['packed_vae_position_ids']).to(AI_DEVICE).to(torch.bfloat16)
        packed_timestep_embeds = self.time_embedder(self.pre_weight, timestep)

        packed_pos_embed = packed_pos_embed.to(AI_DEVICE)
        x_t = x_t.to(AI_DEVICE)
        x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[inputs.generation_input['packed_vae_token_indexes']] = x_t

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": inputs.generation_input['packed_vae_token_indexes'],
                "packed_text_indexes": packed_text_indexes
            }
        
        if self.enable_taylorseer:
            self.scheduler.cache_dic = inputs.model_pred_cache_dic
            self.scheduler.current = inputs.model_pred_current
        
        output = self.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=inputs.generation_input['packed_position_ids'],
            packed_query_indexes=inputs.generation_input['packed_indexes'],
            past_key_values=inputs.gen_context['past_key_values'],
            key_values_lens=inputs.generation_input['key_values_lens'],
            packed_key_value_indexes=inputs.generation_input['packed_key_value_indexes'],
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )


        v_t = self.llm2vae(output[0])
        v_t = v_t[inputs.generation_input['packed_vae_token_indexes']]

        if cfg_text_scale > 1.0:
            if self.enable_taylorseer:
                self.cache_dic = inputs.model_pred_text_cache_dic
                self.current = inputs.model_pred_text_current
            
            cfg_text_output = self.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=inputs.generation_input_cfg_text['cfg_packed_position_ids'],
                packed_query_indexes=inputs.generation_input_cfg_text['cfg_packed_query_indexes'],
                past_key_values=inputs.cfg_text_past_key_values,
                key_values_lens=inputs.generation_input_cfg_text['cfg_key_values_lens'],
                packed_key_value_indexes=inputs.generation_input_cfg_text['cfg_packed_key_value_indexes'],
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_text_v_t = self.llm2vae(cfg_text_output[0])
            cfg_text_v_t = cfg_text_v_t[inputs.generation_input['packed_vae_token_indexes']]

        if cfg_img_scale > 1.0:
            if self.enable_taylorseer:
                self.cache_dic = inputs.model_pred_text_cache_dic
                self.current = inputs.model_pred_text_current
            
            cfg_img_output = self.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=inputs.generation_input_cfg_img['cfg_packed_position_ids'],
                packed_query_indexes=inputs.generation_input_cfg_img['cfg_packed_query_indexes'],
                past_key_values=inputs.cfg_img_past_key_values,
                key_values_lens=inputs.generation_input_cfg_img['cfg_key_values_lens'],
                packed_key_value_indexes=inputs.generation_input_cfg_img['cfg_packed_key_value_indexes'],
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_img_v_t = self.llm2vae(cfg_img_output[0])
            cfg_img_v_t = cfg_img_v_t[inputs.generation_input['packed_vae_token_indexes']]

        if cfg_text_scale > 1.0:
            if self.inference_hyper['cfg_renorm_type'] == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=self.inference_hyper['cfg_renorm_min'], max=1.0)
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                
                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                if self.inference_hyper['cfg_renorm_type'] == "global":
                    norm_v_t = torch.norm(v_t, dtype=torch.float32)
                    norm_v_t_ = torch.norm(v_t_, dtype=torch.float32)
                elif self.inference_hyper['cfg_renorm_type'] == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{self.inference_hyper['cfg_renorm_min']} is not suppoprted")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=self.inference_hyper['cfg_renorm_min'], max=1.0)
                v_t = v_t_ * scale
        else:
            # No CFG
            pass

        self.scheduler.noise_pred = v_t
        return v_t
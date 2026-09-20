import torch
import torch.distributed as dist

from lightx2v.common.kvcache.manager import KVCacheManager
from lightx2v.models.networks.base_model import BaseTransformerModel

from .infer.post_infer import QwenImage21PostInfer
from .infer.pre_infer import QwenImage21PreInfer
from .infer.transformer_infer import QwenImage21TransformerInfer
from .weights.post_weights import QwenImage21PostWeights
from .weights.pre_weights import QwenImage21PreWeights
from .weights.transformer_weights import QwenImage21TransformerWeights


class QwenImage21TransformerModel(BaseTransformerModel):
    pre_weight_class = QwenImage21PreWeights
    transformer_weight_class = QwenImage21TransformerWeights
    post_weight_class = QwenImage21PostWeights

    def __init__(self, model_path, config, device):
        super().__init__(model_path, config, device)
        self._init_infer_class()
        self._init_weights()
        self._init_infer()
        self.kv_cache_manager = None

    def _init_infer_class(self):
        self.pre_infer_class = QwenImage21PreInfer
        self.transformer_infer_class = QwenImage21TransformerInfer
        self.post_infer_class = QwenImage21PostInfer

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class()
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.post_infer = self.post_infer_class()

    def _seq_parallel_pre_process(self, state):
        world_size = dist.get_world_size(self.seq_p_group)
        rank = dist.get_rank(self.seq_p_group)
        target_length = state.hidden_states.shape[0]
        if target_length % world_size:
            raise ValueError(f"Qwen-Image-2.1 target token count ({target_length}) must be divisible by seq_p_size ({world_size}); choose another output size or SP size")
        local_length = target_length // world_size
        begin = rank * local_length
        end = begin + local_length
        state.hidden_states = state.hidden_states[begin:end].contiguous()
        if state.rotary_positions is None:
            state.rotary = state.rotary[begin:end].contiguous()
        else:
            # FlashInfer keeps the full lookup table and indexes the rank-local
            # target positions from it.
            state.rotary_positions = state.rotary_positions[begin:end].contiguous()
        return state

    def _seq_parallel_post_process(self, hidden, state):
        target_length = state.layout.target_len
        world_size = dist.get_world_size(self.seq_p_group)
        expected_local_length = target_length // world_size
        if hidden.shape[0] != expected_local_length:
            raise RuntimeError(f"Qwen-Image-2.1 local target length changed from {expected_local_length} to {hidden.shape[0]}")
        gathered = [torch.empty_like(hidden) for _ in range(world_size)]
        dist.all_gather(gathered, hidden.contiguous(), group=self.seq_p_group)
        return torch.cat(gathered, dim=0)

    @torch.no_grad()
    def prefill_condition_kv(self, inputs):
        """Create and fill condition caches owned by this model for one request."""
        self.clear_condition_kv()
        cache_config = {
            "num_layers": self.config["num_layers"],
            "num_heads": self.config["num_attention_heads"],
            "dim": self.config["num_attention_heads"] * self.config["attention_head_dim"],
        }
        self.kv_cache_manager = KVCacheManager(cache_config, device=self.device)
        try:
            for name in ("cond", "uncond"):
                if name in inputs:
                    branch = inputs[name]
                    cache = self.kv_cache_manager.create_self_attn_kv_cache(name, branch["layout"].prefix_len, kv_cache_scheme="static", step_kv_cache=False)
                    state = self.pre_infer.infer_condition(self.pre_weight, branch["prompt_embeds"], inputs.get("image_latents"), branch["layout"])
                    self.transformer_infer.prefill(self.transformer_weights, state, cache)
        except Exception:
            self.clear_condition_kv()
            raise

    def clear_condition_kv(self):
        if self.kv_cache_manager is not None:
            for cache in self.kv_cache_manager.self_attn_kv_caches.values():
                cache.reset()
        self.kv_cache_manager = None

    def _infer_cond_uncond(self, inputs, infer_condition=True):
        name = "cond" if infer_condition else "uncond"
        branch = inputs[name]
        cache = self.kv_cache_manager.get_self_attn_kv_cache(name)
        state = self.pre_infer.infer_target(self.pre_weight, self.scheduler.latents[0], branch["layout"])
        if self.config["seq_parallel"]:
            state = self._seq_parallel_pre_process(state)
        hidden = self.transformer_infer.infer(self.transformer_weights, state, cache)
        if self.config["seq_parallel"]:
            hidden = self._seq_parallel_post_process(hidden, state)
        noise = self.post_infer.infer(self.post_weight, hidden, state)
        return noise.unsqueeze(0)

    @torch.no_grad()
    def infer(self, inputs):
        if self.kv_cache_manager is None:
            raise RuntimeError("Condition KV must be prefilled before denoising")
        if self.config["enable_cfg"]:
            positive = self._infer_cond_uncond(inputs, True)
            negative = self._infer_cond_uncond(inputs, False)
            self.scheduler.noise_pred = negative + self.scheduler.sample_guide_scale * (positive - negative)
        else:
            self.scheduler.noise_pred = self._infer_cond_uncond(inputs)

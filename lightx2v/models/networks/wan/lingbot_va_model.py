import torch

from lightx2v.models.networks.wan.infer.lingbot_va.pre_infer import LingbotVAPreInfer
from lightx2v.models.networks.wan.infer.lingbot_va.transformer_infer import LingbotVATransformerInfer
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.weights.lingbot_va.pre_weights import LingbotVAPreWeights
from lightx2v.models.networks.wan.weights.lingbot_va.transformer_weights import LingbotVATransformerWeights
from lightx2v.utils.custom_compiler import compiled_method


class WanLingbotVAModel(WanModel):
    pre_weight_class = LingbotVAPreWeights
    transformer_weight_class = LingbotVATransformerWeights

    def __init__(self, *args, **kwargs):
        self.kv_cache_manager = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def build_grid_id(f, h, w, t, f_w=1, f_shift=0, action=False):
        f_idx = torch.arange(f_shift, f + f_shift) * f_w
        h_idx = torch.arange(h)
        w_idx = torch.arange(w)
        ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")
        if action:
            ff_offset = (torch.ones([h]).cumsum(0) / (h + 1)).view(1, -1, 1)
            ff = ff + ff_offset
            hh = torch.ones_like(hh) * -1
            ww = torch.ones_like(ww) * -1
        grid_id = torch.cat([ff.unsqueeze(0), hh.unsqueeze(0), ww.unsqueeze(0)], dim=0).flatten(1)
        return torch.cat([grid_id, torch.full_like(grid_id[:1], t)], dim=0)

    def _init_infer_class(self):
        self.pre_infer_class = LingbotVAPreInfer
        self.post_infer_class = WanPostInfer
        self.transformer_infer_class = LingbotVATransformerInfer

    def clear_cache(self, cache_name):
        for name in self._cache_names(cache_name):
            if self.kv_cache_manager is not None:
                cache = self.kv_cache_manager.get_self_attn_kv_cache(name)
                if cache is not None:
                    cache.reset()

    def clear_pred_cache(self, cache_name):
        for name in self._cache_names(cache_name):
            if self.kv_cache_manager is not None:
                cache = self.kv_cache_manager.get_self_attn_kv_cache(name)
                if cache is not None and hasattr(cache, "clear_pred"):
                    cache.clear_pred()

    @staticmethod
    def cfg_cache_name(cache_name, infer_condition):
        return f"{cache_name}_{'cond' if infer_condition else 'uncond'}"

    @classmethod
    def _cache_names(cls, cache_name):
        return (cache_name, cls.cfg_cache_name(cache_name, True), cls.cfg_cache_name(cache_name, False))

    def _to_cuda_for_lingbot_va(self):
        if self.cpu_offload:
            if self.offload_granularity == "model":
                self.to_cuda()
            else:
                self.pre_weight.to_cuda()
                self.transformer_weights.non_block_weights_to_cuda()

    def _to_cpu_for_lingbot_va(self):
        if self.cpu_offload:
            if self.offload_granularity == "model":
                self.to_cpu()
            else:
                self.pre_weight.to_cpu()
                self.transformer_weights.non_block_weights_to_cpu()

    @torch.no_grad()
    def _infer_lingbot_va_once(self, inputs, action_mode=False, update_cache=0, cache_name="pos"):
        pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs, action_mode=action_mode)
        return self.transformer_infer.infer(
            self.transformer_weights,
            pre_infer_out,
            action_mode=action_mode,
            update_cache=update_cache,
            cache_name=cache_name,
        )

    @compiled_method()
    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True, action_mode=False, update_cache=0, cache_name="pos"):
        model_inputs = inputs
        if not infer_condition:
            model_inputs = dict(inputs)
            model_inputs["text_emb"] = model_inputs["negative_text_emb"]
        return self._infer_lingbot_va_once(
            model_inputs,
            action_mode=action_mode,
            update_cache=update_cache,
            cache_name=cache_name,
        )

    @torch.no_grad()
    def _infer_lingbot_va(self, inputs, action_mode=False, update_cache=0, cache_name="pos"):
        self._to_cuda_for_lingbot_va()
        try:
            return self._infer_lingbot_va_once(
                inputs,
                action_mode=action_mode,
                update_cache=update_cache,
                cache_name=cache_name,
            )
        finally:
            self._to_cpu_for_lingbot_va()

    @torch.no_grad()
    def _infer_lingbot_va_guided(self, inputs, action_mode=False, update_cache=0, cache_name="pos", guide_scale=1.0):
        self._to_cuda_for_lingbot_va()
        try:
            noise_pred_cond = self._infer_cond_uncond(
                inputs,
                infer_condition=True,
                action_mode=action_mode,
                update_cache=update_cache,
                cache_name=self.cfg_cache_name(cache_name, True),
            )
            noise_pred_uncond = self._infer_cond_uncond(
                inputs,
                infer_condition=False,
                action_mode=action_mode,
                update_cache=update_cache,
                cache_name=self.cfg_cache_name(cache_name, False),
            )
            return noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
        finally:
            self._to_cpu_for_lingbot_va()

    @torch.no_grad()
    def infer_latent(self, inputs, update_cache=0, cache_name="pos", enable_cfg=False, guide_scale=1.0):
        if enable_cfg:
            return self._infer_lingbot_va_guided(
                inputs,
                action_mode=False,
                update_cache=update_cache,
                cache_name=cache_name,
                guide_scale=guide_scale,
            )
        return self._infer_lingbot_va(inputs, action_mode=False, update_cache=update_cache, cache_name=cache_name)

    @torch.no_grad()
    def infer_action(self, inputs, update_cache=0, cache_name="pos", enable_cfg=False, guide_scale=1.0):
        if enable_cfg:
            return self._infer_lingbot_va_guided(
                inputs,
                action_mode=True,
                update_cache=update_cache,
                cache_name=cache_name,
                guide_scale=guide_scale,
            )
        return self._infer_lingbot_va(inputs, action_mode=True, update_cache=update_cache, cache_name=cache_name)

    @torch.no_grad()
    def infer(self, inputs):
        action_mode = bool(inputs.get("action_mode", False))
        update_cache = int(inputs.get("update_cache", 0))
        cache_name = inputs.get("cache_name", "pos")
        enable_cfg = bool(inputs.get("enable_cfg", False))
        guide_scale = float(inputs.get("guide_scale", 1.0))
        if enable_cfg:
            noise_pred = self._infer_lingbot_va_guided(
                inputs,
                action_mode=action_mode,
                update_cache=update_cache,
                cache_name=cache_name,
                guide_scale=guide_scale,
            )
        else:
            noise_pred = self._infer_lingbot_va(inputs, action_mode=action_mode, update_cache=update_cache, cache_name=cache_name)
        self.scheduler.noise_pred = noise_pred
        return noise_pred

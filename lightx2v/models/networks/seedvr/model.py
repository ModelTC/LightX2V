import os

import torch

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.seedvr.infer.post_infer import SeedVRPostInfer
from lightx2v.models.networks.seedvr.infer.pre_infer import SeedVRPreInfer
from lightx2v.models.networks.seedvr.infer.transformer_infer import SeedVRTransformerInfer
from lightx2v.models.networks.seedvr.utils import na as na_utils
from lightx2v.models.networks.seedvr.utils.utils import classifier_free_guidance_dispatcher
from lightx2v.models.networks.seedvr.weights.post_weights import SeedVRPostWeights
from lightx2v.models.networks.seedvr.weights.pre_weights import SeedVRPreWeights
from lightx2v.models.networks.seedvr.weights.transformer_weights import SeedVRTransformerWeights
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


class SeedVRNaDiTModel(BaseTransformerModel):
    """SeedVR model using LightX2V weight wrappers + inference pipeline."""

    pre_weight_class = SeedVRPreWeights
    transformer_weight_class = SeedVRTransformerWeights
    post_weight_class = SeedVRPostWeights

    def __init__(self, model_path, config, device, model_type="seedvr", lora_path=None, lora_strength=1.0):
        super().__init__(model_path, config, device, model_type, lora_path, lora_strength)
        self._apply_seedvr_defaults()
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _apply_seedvr_defaults(self):
        defaults = {
            "vid_in_channels": 33,
            "vid_out_channels": 16,
            "vid_dim": 2560,
            "txt_in_dim": 5120,
            "txt_dim": 2560,
            "emb_dim": 6 * 2560,
            "heads": 20,
            "head_dim": 128,
            "expand_ratio": 4,
            "norm": "fusedrms",
            "norm_eps": 1.0e-5,
            "ada": "single",
            "qk_bias": False,
            "qk_norm": "fusedrms",
            "patch_size": (1, 2, 2),
            "num_layers": 32,
            "block_type": ["mmdit_sr"] * 32,
            "mm_layers": 10,
            "mlp_type": "swiglu",
            "rope_type": "mmrope3d",
            "rope_dim": 128,
            "window": [(4, 3, 3)] * 32,
            "window_method": ["720pwin_by_size_bysize", "720pswin_by_size_bysize"] * 16,
            "vid_out_norm": "fusedrms",
            "rms_norm_type": "torch",
            "layer_norm_type": "torch",
            "timestep_sinusoidal_dim": 256,
            "seq_parallel": False,
            "seedvr_has_vid_in": True,
        }
        for key, value in defaults.items():
            self.config.setdefault(key, value)

        if "emb_dim" not in self.config:
            self.config["emb_dim"] = 6 * self.config["vid_dim"]

        self.config.setdefault("dit_quant_scheme", "Default")
        self.config.setdefault("dit_quantized", False)

    def _init_infer_class(self):
        self.pre_infer_class = SeedVRPreInfer
        self.transformer_infer_class = SeedVRTransformerInfer
        self.post_infer_class = SeedVRPostInfer

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        # SeedVR weights are typically in .pth/.pt format, not safetensors.
        ckpt_path = self.config.get("dit_original_ckpt") or self.model_path
        if ckpt_path and os.path.isfile(ckpt_path):
            ckpt_lower = str(ckpt_path).lower()
            if not ckpt_lower.endswith(".safetensors"):
                try:
                    state = torch.load(ckpt_path, map_location=AI_DEVICE)
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    remove_keys = self.remove_keys if hasattr(self, "remove_keys") else []
                    weight_dict = {}
                    for key, tensor in state.items():
                        if any(remove_key in key for remove_key in remove_keys):
                            continue
                        if unified_dtype or all(s not in key for s in sensitive_layer):
                            weight_dict[key] = tensor.to(GET_DTYPE())
                        else:
                            weight_dict[key] = tensor.to(GET_SENSITIVE_DTYPE())
                    return weight_dict
                except Exception:
                    # Fall back to BaseTransformerModel loader
                    pass

        # Fallback to BaseTransformerModel safetensors loader
        return super()._load_ckpt(unified_dtype, sensitive_layer)

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        return pre_infer_out

    @torch.no_grad()
    def _seq_parallel_post_process(self, x):
        return x

    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True):
        pass

    @torch.no_grad()
    def _infer_dit(self, vid, txt, vid_shape, txt_shape, timestep):
        pre_infer_out = self.pre_infer.infer(
            weights=self.pre_weight,
            vid=vid,
            txt=txt,
            vid_shape=vid_shape,
            txt_shape=txt_shape,
            timestep=timestep,
        )

        if self.config.get("seq_parallel", False):
            pre_infer_out = self._seq_parallel_pre_process(pre_infer_out)

        vid, txt, vid_shape, txt_shape = self.transformer_infer.infer(
            self.transformer_weights.blocks,
            pre_infer_out,
        )

        vid = self.post_infer.infer(self.post_weight, vid, pre_infer_out)

        if self.config.get("seq_parallel", False):
            vid = self._seq_parallel_post_process(vid)

        return vid

    @torch.no_grad()
    def infer(self, inputs):
        noises = inputs.get("noises", None)
        conditions = inputs.get("conditions", None)
        texts_pos = inputs["text_encoder_output"]["texts_pos"]
        texts_neg = inputs["text_encoder_output"]["texts_neg"]

        cfg_scale = 1.0
        cfg_rescale = 0.0
        cfg_partial = 1.0

        text_pos_embeds, text_pos_shapes = na_utils.flatten(texts_pos)
        text_neg_embeds, text_neg_shapes = na_utils.flatten(texts_neg)
        latents, latents_shapes = na_utils.flatten(noises)
        latents_cond, _ = na_utils.flatten(conditions)
        batch_size = len(noises)

        latents = self.scheduler.sampler.sample(
            x=latents,
            f=lambda args: classifier_free_guidance_dispatcher(
                pos=lambda: self._infer_dit(
                    vid=torch.cat([args.x_t, latents_cond], dim=-1),
                    txt=text_pos_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_pos_shapes,
                    timestep=args.t.repeat(batch_size),
                ),
                neg=lambda: self._infer_dit(
                    vid=torch.cat([args.x_t, latents_cond], dim=-1),
                    txt=text_neg_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_neg_shapes,
                    timestep=args.t.repeat(batch_size),
                ),
                scale=(cfg_scale if (args.i + 1) / len(self.scheduler.sampler.timesteps) <= cfg_partial else 1.0),
                rescale=cfg_rescale,
            ),
        )

        latents_list = na_utils.unflatten(latents, latents_shapes)
        self.scheduler.latents = latents_list
        return

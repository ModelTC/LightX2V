

import torch
import torch.distributed as dist

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.seedvr.dit_v2.nadit import NaDiT
from lightx2v.models.networks.seedvr.dit_v2 import na as na_utils
from lightx2v.models.networks.seedvr.dit_v2.diffusion.utils import classifier_free_guidance_dispatcher
from lightx2v.models.networks.seedvr.dit_v2.rotary_embedding_torch import RotaryEmbedding
from torch import nn


class SeedVRNaDiTModel(BaseTransformerModel):
    """SeedVR model using the original NaDiT implementation (no LightX2V weights)."""

    pre_weight_class = None
    transformer_weight_class = None
    post_weight_class = None

    def __init__(self, model_path, config, device, model_type="seedvr", lora_path=None, lora_strength=1.0):
        super().__init__(model_path, config, device, model_type, lora_path, lora_strength)

        self._init_model()
        # NaDiT path does not use pre/transformer/post infer modules from BaseTransformerModel.
        self.pre_infer = None
        self.transformer_infer = None
        self.post_infer = None

    def _init_model(self):
        window = [(4, 3, 3)] * 32
        window_method = ["720pwin_by_size_bysize", "720pswin_by_size_bysize"] * 16

        self.dit = NaDiT(
            vid_in_channels=33,
            vid_out_channels=16,
            vid_dim=2560,
            vid_out_norm="fusedrms",
            txt_in_dim=5120,
            txt_in_norm="fusedln",
            txt_dim=2560,
            emb_dim=6*2560,
            heads=20,
            head_dim=128,
            expand_ratio=4,
            norm="fusedrms",
            norm_eps=1.0e-05,
            ada="single",
            qk_bias=False,
            qk_norm="fusedrms",
            patch_size=[1, 2, 2],
            num_layers=32,
            block_type=["mmdit_sr"] * 32,
            mm_layers=10,
            mlp_type="swiglu",
            msa_type=None,
            rope_type="mmrope3d",
            rope_dim=128,
            window=window,
            window_method=window_method,
        )

        state = torch.load(self.model_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.dit.load_state_dict(state, strict=True, assign=True)

        self.dit = meta_non_persistent_buffer_init_fn(self.dit)

        self.dit.to(self.device, dtype=torch.bfloat16)
        self.dit.eval()

    def _init_infer_class(self):
        pass

    def _init_infer(self):
        pass

    def set_scheduler(self, scheduler):
        # Override BaseTransformerModel behavior (which assumes pre/transformer/post infer).
        self.scheduler = scheduler

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

        was_training = self.dit.training
        self.dit.eval()

        latents = self.scheduler.sampler.sample(
            x=latents,
            f=lambda args: classifier_free_guidance_dispatcher(
                pos=lambda: self.dit(
                    vid=torch.cat([args.x_t, latents_cond], dim=-1),
                    txt=text_pos_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_pos_shapes,
                    timestep=args.t.repeat(batch_size),
                ).vid_sample,
                neg=lambda: self.dit(
                    vid=torch.cat([args.x_t, latents_cond], dim=-1),
                    txt=text_neg_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_neg_shapes,
                    timestep=args.t.repeat(batch_size),
                ).vid_sample,
                scale=(cfg_scale if (args.i + 1) / len(self.scheduler.sampler.timesteps) <= cfg_partial else 1.0),
                rescale=cfg_rescale,
            ),
        )

        self.dit.train(was_training)

        latents_list = na_utils.unflatten(latents, latents_shapes)

        self.scheduler.latents = latents_list

        return


def meta_non_persistent_buffer_init_fn(module: nn.Module) -> nn.Module:
    """
    Used for materializing `non-persistent tensor buffers` while model resuming.

    Since non-persistent tensor buffers are not saved in state_dict,
    when initializing model with meta device, user should materialize those buffers manually.

    Currently, only `rope.dummy` is this special case.
    """
    with torch.no_grad():
        for submodule in module.modules():
            if not isinstance(submodule, RotaryEmbedding):
                continue
            for buffer_name, buffer in submodule.named_buffers(recurse=False):
                if buffer.is_meta and "dummy" in buffer_name:
                    materialized_buffer = torch.zeros_like(buffer, device="cpu")
                    setattr(submodule, buffer_name, materialized_buffer)
    assert not any(b.is_meta for n, b in module.named_buffers())
    return module

import torch
import torch.nn.functional as F

try:
    from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed
except ImportError:
    Flux2PosEmbed = None

from .module_io import Flux2KleinPreInferModuleOutput


class Flux2KleinPreInfer:
    """Pre-processing inference for Flux2Klein.

    Maps Flux2KleinPipeline.__call__ preprocessing logic to inference:
    - Embed image latents via x_embedder
    - Embed text embeddings via context_embedder
    - Compute timestep embeddings
    - Collect position IDs and rotary embeddings from scheduler
    """

    def __init__(self, config):
        self.config = config
        self.attention_kwargs = {}
        self.cpu_offload = config.get("cpu_offload", False)

        rope_theta = config.get("rope_theta", 2000)
        axes_dims_rope = config.get("axes_dims_rope", (32, 32, 32, 32))
        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, hidden_states, encoder_hidden_states, txt_ids=None, img_ids=None):
        hidden_states = weights.x_embedder.apply(hidden_states.squeeze(0))

        encoder_hidden_states = weights.context_embedder.apply(encoder_hidden_states.squeeze(0))

        timesteps_proj = self.scheduler.timesteps_proj
        timestep_embed = weights.timestep_embedder_linear_1.apply(timesteps_proj)
        timestep_embed = F.silu(timestep_embed)
        timestep_embed = weights.timestep_embedder_linear_2.apply(timestep_embed)

        txt_ids_final = txt_ids if txt_ids is not None else getattr(self.scheduler, "txt_ids", None)
        img_ids_final = img_ids if img_ids is not None else getattr(self.scheduler, "latent_image_ids", None)

        image_rotary_emb = None
        if img_ids_final is not None and txt_ids_final is not None:
            if img_ids_final.ndim == 3:
                img_ids_final = img_ids_final[0]
            if txt_ids_final.ndim == 3:
                txt_ids_final = txt_ids_final[0]

            image_rope = self.pos_embed(img_ids_final)
            text_rope = self.pos_embed(txt_ids_final)

            freqs_cos = torch.cat([text_rope[0], image_rope[0]], dim=0)
            freqs_sin = torch.cat([text_rope[1], image_rope[1]], dim=0)

        if self.config.get("rope_type", "flashinfer") == "flashinfer":
            cos_half = freqs_cos[:, ::2].contiguous()  # [L, D/2]
            sin_half = freqs_sin[:, ::2].contiguous()  # [L, D/2]
            image_rotary_emb = torch.cat([cos_half, sin_half], dim=-1)  # [L, D]
        else:
            image_rotary_emb = (freqs_cos, freqs_sin)

        return Flux2KleinPreInferModuleOutput(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep_embed,
            txt_ids=txt_ids_final,
            img_ids=img_ids_final,
            image_rotary_emb=image_rotary_emb,
        )

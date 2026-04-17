import torch
import torch.nn.functional as F

try:
    from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed
except (ImportError, ModuleNotFoundError):
    Flux2PosEmbed = None

from lightx2v.models.networks.flux2_klein.infer.module_io import Flux2KleinPreInferModuleOutput


class Flux2DevPreInfer:
    """Pre-processing inference for Flux2Dev.

    Extends Flux2Klein's pre-infer with guidance embedding computation.
    Flux2 dev uses embedded guidance (guidance tensor added to timestep embedding)
    instead of classifier-free guidance with two forward passes.
    """

    def __init__(self, config):
        self.config = config
        self.attention_kwargs = {}
        self.cpu_offload = config.get("cpu_offload", False)

        if Flux2PosEmbed is None:
            raise ImportError(
                "Flux2PosEmbed is not available. Please upgrade diffusers to a version "
                "that supports transformer_flux2: pip install --upgrade diffusers"
            )

        rope_theta = config.get("rope_theta", 2000)
        axes_dims_rope = config.get("axes_dims_rope", (32, 32, 32, 32))
        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, hidden_states, encoder_hidden_states, txt_ids=None, img_ids=None):
        hidden_states = weights.x_embedder.apply(hidden_states.squeeze(0))

        encoder_hidden_states = weights.context_embedder.apply(encoder_hidden_states.squeeze(0))

        # Timestep embedding
        timesteps_proj = self.scheduler.timesteps_proj
        timestep_embed = weights.timestep_embedder_linear_1.apply(timesteps_proj)
        timestep_embed = F.silu(timestep_embed)
        timestep_embed = weights.timestep_embedder_linear_2.apply(timestep_embed)

        # Guidance embedding (Flux2 dev specific)
        guidance_proj = self.scheduler.guidance_proj
        guidance_embed = weights.guidance_embedder_linear_1.apply(guidance_proj)
        guidance_embed = F.silu(guidance_embed)
        guidance_embed = weights.guidance_embedder_linear_2.apply(guidance_embed)

        # Combine timestep and guidance embeddings
        timestep_embed = timestep_embed + guidance_embed

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
            cos_half = freqs_cos[:, ::2].contiguous()
            sin_half = freqs_sin[:, ::2].contiguous()
            image_rotary_emb = torch.cat([cos_half, sin_half], dim=-1)
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

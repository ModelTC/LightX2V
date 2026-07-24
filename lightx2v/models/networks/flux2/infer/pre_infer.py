import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed
except (ImportError, ModuleNotFoundError):
    Flux2PosEmbed = None

from .module_io import Flux2PreInferModuleOutput


class Flux2PreInfer:
    """Pre-processing inference for Flux2 (base, used by Klein).

    Maps pipeline preprocessing logic to inference:
    - Embed image latents via x_embedder
    - Embed text embeddings via context_embedder
    - Compute timestep embeddings
    - Collect position IDs and rotary embeddings from scheduler
    """

    def __init__(self, config):
        self.config = config
        self.attention_kwargs = {}
        self.cpu_offload = config.get("cpu_offload", False)

        if Flux2PosEmbed is None:
            raise ImportError("Flux2PosEmbed is not available. Please upgrade diffusers to a version that supports transformer_flux2: pip install --upgrade diffusers")

        rope_theta = config.get("rope_theta", 2000)
        axes_dims_rope = config.get("axes_dims_rope", (32, 32, 32, 32))
        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)
        self.rope = None
        self.head_dim = config["attention_head_dim"]
        if config.get("seq_parallel", False):
            self.seq_p_group = config.get("device_mesh").get_group(mesh_dim="seq_p")
        else:
            self.seq_p_group = None
        self.clear_rope_cache()

    def clear_rope_cache(self):
        self._rope_cache = {True: None, False: None}

    def set_rope(self, rope):
        self.rope = rope
        self.clear_rope_cache()

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.clear_rope_cache()

    def _prepare_sequence_parallel(self, image_rotary_emb, num_txt_tokens):
        if self.seq_p_group is None:
            return image_rotary_emb

        world_size = dist.get_world_size(self.seq_p_group)
        cur_rank = dist.get_rank(self.seq_p_group)
        freqs_cos, freqs_sin = image_rotary_emb

        txt_cos = freqs_cos[:num_txt_tokens]
        img_cos = freqs_cos[num_txt_tokens:]
        txt_sin = freqs_sin[:num_txt_tokens]
        img_sin = freqs_sin[num_txt_tokens:]

        seqlen = img_cos.shape[0]
        padding_size = (world_size - (seqlen % world_size)) % world_size
        if padding_size > 0:
            img_cos = F.pad(img_cos, (0, 0, 0, padding_size))
            img_sin = F.pad(img_sin, (0, 0, 0, padding_size))
        img_cos = torch.chunk(img_cos, world_size, dim=0)[cur_rank]
        img_sin = torch.chunk(img_sin, world_size, dim=0)[cur_rank]

        return torch.cat([txt_cos, img_cos], dim=0), torch.cat([txt_sin, img_sin], dim=0)

    def prepare_rope_cache(self, txt_ids, img_ids, num_txt_tokens):
        if self.rope is None:
            raise RuntimeError("Flux2 RoPE must be set before inference.")

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        image_rope = self.pos_embed(img_ids)
        text_rope = self.pos_embed(txt_ids)
        image_rotary_emb = (
            torch.cat([text_rope[0], image_rope[0]], dim=0),
            torch.cat([text_rope[1], image_rope[1]], dim=0),
        )
        image_rotary_emb = self._prepare_sequence_parallel(image_rotary_emb, num_txt_tokens)
        image_rotary_emb = self.rope.prepare_freqs(image_rotary_emb, rotary_dim=self.head_dim)
        image_rotary_positions = self.rope.prepare_positions(image_rotary_emb)
        return image_rotary_emb, image_rotary_positions

    def get_rope_cache(self, txt_ids, img_ids, num_txt_tokens):
        if txt_ids is None or img_ids is None:
            return None, None

        infer_condition = bool(getattr(self.scheduler, "infer_condition", True))
        cached = self._rope_cache[infer_condition]
        if cached is not None and cached[0][0] is txt_ids and cached[0][1] is img_ids and cached[0][2] == num_txt_tokens:
            return cached[1]

        rope_cache = self.prepare_rope_cache(txt_ids, img_ids, num_txt_tokens)
        self._rope_cache[infer_condition] = ((txt_ids, img_ids, num_txt_tokens), rope_cache)
        return rope_cache

    def infer(self, weights, hidden_states, encoder_hidden_states, txt_ids=None, img_ids=None):
        hidden_states = weights.x_embedder.apply(hidden_states.squeeze(0))

        encoder_hidden_states = weights.context_embedder.apply(encoder_hidden_states.squeeze(0))

        timesteps_proj = self.scheduler.timesteps_proj
        timestep_embed = weights.timestep_embedder_linear_1.apply(timesteps_proj)
        timestep_embed = F.silu(timestep_embed)
        timestep_embed = weights.timestep_embedder_linear_2.apply(timestep_embed)

        txt_ids_final = txt_ids if txt_ids is not None else getattr(self.scheduler, "txt_ids", None)
        img_ids_final = img_ids if img_ids is not None else getattr(self.scheduler, "latent_image_ids", None)

        image_rotary_emb, image_rotary_positions = self.get_rope_cache(txt_ids_final, img_ids_final, encoder_hidden_states.shape[0])
        if img_ids_final is not None and img_ids_final.ndim == 3:
            img_ids_final = img_ids_final[0]
        if txt_ids_final is not None and txt_ids_final.ndim == 3:
            txt_ids_final = txt_ids_final[0]

        return Flux2PreInferModuleOutput(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep_embed,
            txt_ids=txt_ids_final,
            img_ids=img_ids_final,
            image_rotary_emb=image_rotary_emb,
            image_rotary_positions=image_rotary_positions,
        )

    def infer_partial(self, weights, hidden_states, encoder_hidden_states, txt_ids=None, img_ids=None):
        """Compute timestep embedding and RoPE only (skip x_embedder / context_embedder).

        Used by non-first pipeline stages that receive already-embedded
        hidden_states and encoder_hidden_states via P2P.
        """
        timesteps_proj = self.scheduler.timesteps_proj
        timestep_embed = weights.timestep_embedder_linear_1.apply(timesteps_proj)
        timestep_embed = F.silu(timestep_embed)
        timestep_embed = weights.timestep_embedder_linear_2.apply(timestep_embed)

        txt_ids_final = txt_ids if txt_ids is not None else getattr(self.scheduler, "txt_ids", None)
        img_ids_final = img_ids if img_ids is not None else getattr(self.scheduler, "latent_image_ids", None)

        num_txt_tokens = encoder_hidden_states.shape[0] if encoder_hidden_states is not None else 0
        image_rotary_emb, image_rotary_positions = self.get_rope_cache(txt_ids_final, img_ids_final, num_txt_tokens)
        if img_ids_final is not None and img_ids_final.ndim == 3:
            img_ids_final = img_ids_final[0]
        if txt_ids_final is not None and txt_ids_final.ndim == 3:
            txt_ids_final = txt_ids_final[0]

        return Flux2PreInferModuleOutput(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep_embed,
            txt_ids=txt_ids_final,
            img_ids=img_ids_final,
            image_rotary_emb=image_rotary_emb,
            image_rotary_positions=image_rotary_positions,
        )


class Flux2DevPreInfer(Flux2PreInfer):
    """Pre-processing inference for Flux2 Dev.

    Extends base with guidance embedding computation.
    Flux2 dev uses embedded guidance (guidance tensor added to timestep embedding)
    instead of classifier-free guidance with two forward passes.
    """

    def infer(self, weights, hidden_states, encoder_hidden_states, txt_ids=None, img_ids=None):
        hidden_states = weights.x_embedder.apply(hidden_states.squeeze(0))

        encoder_hidden_states = weights.context_embedder.apply(encoder_hidden_states.squeeze(0))

        timesteps_proj = self.scheduler.timesteps_proj
        timestep_embed = weights.timestep_embedder_linear_1.apply(timesteps_proj)
        timestep_embed = F.silu(timestep_embed)
        timestep_embed = weights.timestep_embedder_linear_2.apply(timestep_embed)

        guidance_proj = self.scheduler.guidance_proj
        guidance_embed = weights.guidance_embedder_linear_1.apply(guidance_proj)
        guidance_embed = F.silu(guidance_embed)
        guidance_embed = weights.guidance_embedder_linear_2.apply(guidance_embed)

        timestep_embed = timestep_embed + guidance_embed

        txt_ids_final = txt_ids if txt_ids is not None else getattr(self.scheduler, "txt_ids", None)
        img_ids_final = img_ids if img_ids is not None else getattr(self.scheduler, "latent_image_ids", None)

        image_rotary_emb, image_rotary_positions = self.get_rope_cache(txt_ids_final, img_ids_final, encoder_hidden_states.shape[0])
        if img_ids_final is not None and img_ids_final.ndim == 3:
            img_ids_final = img_ids_final[0]
        if txt_ids_final is not None and txt_ids_final.ndim == 3:
            txt_ids_final = txt_ids_final[0]

        return Flux2PreInferModuleOutput(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep_embed,
            txt_ids=txt_ids_final,
            img_ids=img_ids_final,
            image_rotary_emb=image_rotary_emb,
            image_rotary_positions=image_rotary_positions,
        )


# Backward-compatible aliases
Flux2KleinPreInfer = Flux2PreInfer

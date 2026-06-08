import math

import torch
import torch.nn.functional as F

from lightx2v.models.networks.hidream_o1_image.infer.module_io import HidreamPreInferOutput
from lightx2v.models.networks.hidream_o1_image.infer.vision_infer import HidreamO1ImageVisionInfer


class HidreamO1ImagePreInfer:
    def __init__(self, config):
        self.config = config
        self.vision_infer = HidreamO1ImageVisionInfer(config)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, sample, z_in, t_pixeldit, precomputed_image_embeds=None, precomputed_deepstack_image_embeds=None):
        input_ids = sample["input_ids"]
        inputs_embeds = weights.input_embeddings.apply(input_ids)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        cond_image_embeds = None
        cond_deepstack_image_embeds = None
        if "pixel_values" in sample and sample["pixel_values"] is not None:
            if precomputed_image_embeds is not None and precomputed_deepstack_image_embeds is not None:
                image_embeds = precomputed_image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                deepstack_image_embeds = [item.to(inputs_embeds.device, inputs_embeds.dtype) for item in precomputed_deepstack_image_embeds]
            else:
                image_embeds_list, deepstack_image_embeds = self.vision_infer.infer(
                    weights.visual,
                    sample["pixel_values"],
                    sample["image_grid_thw"],
                )
                image_embeds = torch.cat(image_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

            image_mask = (input_ids == weights.model_config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            if image_mask.sum() // inputs_embeds.shape[-1] != image_embeds.shape[0]:
                raise ValueError(f"Image placeholder count mismatch: placeholders={image_mask.sum() // inputs_embeds.shape[-1]}, image_embeds={image_embeds.shape[0]}")
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            visual_pos_masks = image_mask[..., 0]
            deepstack_visual_embeds = deepstack_image_embeds
            cond_image_embeds = image_embeds
            cond_deepstack_image_embeds = deepstack_image_embeds

        timestep = t_pixeldit.reshape(-1).to(inputs_embeds.device)
        t_emb = self._timestep_embedding(timestep * 1000, weights.frequency_embedding_size)
        t_emb = weights.t_embedder_linear_1.apply(t_emb.to(weights.t_embedder_linear_1.weight.dtype))
        t_emb = F.silu(t_emb)
        t_emb = weights.t_embedder_linear_2.apply(t_emb)
        tms_mask = input_ids == weights.tms_token_id
        tms_mask = tms_mask.unsqueeze(-1).expand_as(inputs_embeds)
        t_emb = t_emb.unsqueeze(1).expand_as(inputs_embeds)
        inputs_embeds = torch.where(tms_mask, t_emb, inputs_embeds)

        if isinstance(z_in, list):
            z_in = torch.cat(z_in, dim=0)
        z_in = z_in.to(inputs_embeds.device)
        z_shape = z_in.shape
        z_flat = z_in.reshape(-1, z_shape[-1])
        vinputs_embedded = weights.x_embedder_proj1.apply(z_flat)
        vinputs_embedded = weights.x_embedder_proj2.apply(vinputs_embedded).to(inputs_embeds.dtype)
        vinputs_embedded = vinputs_embedded.reshape(*z_shape[:-1], vinputs_embedded.shape[-1])
        inputs_embeds = torch.cat([inputs_embeds, vinputs_embedded], dim=1)

        if visual_pos_masks is not None:
            if visual_pos_masks.shape[0] != inputs_embeds.shape[0]:
                visual_pos_masks = visual_pos_masks.expand(inputs_embeds.shape[0], -1)
            pad = torch.zeros(
                visual_pos_masks.shape[0],
                vinputs_embedded.shape[1],
                dtype=visual_pos_masks.dtype,
                device=visual_pos_masks.device,
            )
            visual_pos_masks = torch.cat([visual_pos_masks, pad], dim=1)

        token_types = sample["token_types"]
        if isinstance(token_types, list):
            token_types = torch.cat(token_types, dim=0)
        token_types = token_types.to(inputs_embeds.device)
        if token_types.dim() == 1:
            token_types = token_types.unsqueeze(0)
        elif token_types.dim() == 2 and token_types.shape[-1] == 1 and token_types.shape[0] == inputs_embeds.shape[1]:
            token_types = token_types.squeeze(-1).unsqueeze(0)
        if token_types.shape[0] == 1 and inputs_embeds.shape[0] > 1:
            token_types = token_types.expand(inputs_embeds.shape[0], -1)

        return HidreamPreInferOutput(
            inputs_embeds=inputs_embeds,
            position_ids=sample["position_ids"],
            token_types=token_types,
            vinput_mask=sample["vinput_mask"],
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            cond_image_embeds=cond_image_embeds,
            cond_deepstack_image_embeds=cond_deepstack_image_embeds,
            tgt_image_len=sample.get("tgt_image_len"),
        )

    def _timestep_embedding(self, t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

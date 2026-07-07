import torch
import torch.nn.functional as F

from lightx2v.models.networks.hunyuan_image3.infer.module_io import HunyuanImage3PreInferOutput
from lightx2v.models.networks.hunyuan_image3.infer.utils import apply_linear, apply_timestep_embedder, first_weight_device, to_device


class HunyuanImage3PreInfer:
    def __init__(self, config):
        self.config = config

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def _apply_resblock(self, block, x, emb):
        h = block.in_norm.apply_group_norm(x)
        h = F.silu(h)
        h = block.in_conv.apply(h)
        emb_out = apply_linear(block.emb_proj, F.silu(emb))
        while emb_out.ndim < h.ndim:
            emb_out = emb_out[..., None]
        scale, shift = emb_out.chunk(2, dim=1)
        h = block.out_norm.apply_group_norm(h) * (1.0 + scale) + shift
        h = F.silu(h)
        h = block.out_conv.apply(h)
        if block.skip_connection is None:
            return x + h
        return block.skip_connection.apply(x) + h

    def _patch_embed(self, weights, images, timesteps):
        t_emb = apply_timestep_embedder(weights.time_embed, timesteps)
        x = weights.patch_embed.input_conv.apply(images)
        for block in weights.patch_embed.blocks:
            x = self._apply_resblock(block, x, t_emb)
        token_h, token_w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x, token_h, token_w

    def _instantiate_image_tokens(self, weights, hidden_states, images, image_mask, timesteps):
        if hidden_states is None:
            image_seq, token_h, token_w = self._patch_embed(weights, images, timesteps)
            if image_mask is not None:
                batch, _, hidden = image_seq.shape
                image_mask = image_mask.to(device=image_seq.device)
                out = image_seq.new_zeros(batch, image_mask.shape[1], hidden)
                index = torch.arange(image_mask.shape[1], device=image_seq.device).unsqueeze(0).repeat(batch, 1)
                image_scatter_index = index.masked_select(image_mask.bool()).reshape(batch, -1)
                if image_scatter_index.shape[1] != image_seq.shape[1]:
                    raise ValueError(
                        f"HunyuanImage3 image_mask selects {image_scatter_index.shape[1]} tokens, "
                        f"but image patch embed produced {image_seq.shape[1]} tokens."
                    )
                out.scatter_(
                    dim=1,
                    index=image_scatter_index.unsqueeze(-1).repeat(1, 1, hidden),
                    src=image_seq,
                )
                return out, (token_h, token_w)
            timestep_emb = apply_timestep_embedder(weights.timestep_emb, timesteps).reshape(images.size(0), -1, self.config["hidden_size"])
            return torch.cat([timestep_emb, image_seq], dim=1), (token_h, token_w)

        batch, seqlen, hidden = hidden_states.shape
        index = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).repeat(batch, 1)
        if isinstance(images, torch.Tensor):
            image_seq, token_h, token_w = self._patch_embed(weights, images, timesteps)
            image_scatter_index = index.masked_select(image_mask.bool()).reshape(batch, -1)
            hidden_states.scatter_(
                dim=1,
                index=image_scatter_index.unsqueeze(-1).repeat(1, 1, hidden),
                src=image_seq,
            )
            return hidden_states, (token_h, token_w)

        token_hw = None
        for batch_idx, (image, timestep) in enumerate(zip(images, timesteps)):
            timestep = timestep.to(hidden_states.device)
            if isinstance(image, torch.Tensor):
                image = image.to(hidden_states.device)
                image_seq, token_h, token_w = self._patch_embed(weights, image, timestep)
                token_hw = (token_h, token_w)
            elif isinstance(image, list):
                image_seq_parts = []
                for image_idx, image_item in enumerate(image):
                    image_item = image_item.unsqueeze(0).to(hidden_states.device)
                    image_seq_item, token_h, token_w = self._patch_embed(weights, image_item, timestep[image_idx : image_idx + 1])
                    token_hw = (token_h, token_w)
                    image_seq_parts.append(image_seq_item)
                image_seq = torch.cat(image_seq_parts, dim=1)
            else:
                raise TypeError(f"HunyuanImage3 image item should be a tensor or list, got {type(image)}")

            image_scatter_index = index[batch_idx : batch_idx + 1].masked_select(image_mask[batch_idx : batch_idx + 1].bool()).reshape(1, -1)
            hidden_states[batch_idx : batch_idx + 1].scatter_(
                dim=1,
                index=image_scatter_index.unsqueeze(-1).repeat(1, 1, hidden),
                src=image_seq.reshape(1, -1, hidden),
            )
        return hidden_states, token_hw

    def _instantiate_vit_image_tokens(self, hidden_states, image_embeds, image_mask):
        if image_embeds is None or image_mask is None:
            return hidden_states

        batch, seqlen, hidden = hidden_states.shape
        index = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).repeat(batch, 1)
        if isinstance(image_embeds, torch.Tensor):
            image_embeds = image_embeds.to(device=hidden_states.device, dtype=hidden_states.dtype)
            image_scatter_index = index.masked_select(image_mask.bool()).reshape(batch, -1)
            hidden_states.scatter_(
                dim=1,
                index=image_scatter_index.unsqueeze(-1).repeat(1, 1, hidden),
                src=image_embeds.reshape(batch, -1, hidden),
            )
            return hidden_states

        for batch_idx, embeds in enumerate(image_embeds):
            embeds = embeds.to(device=hidden_states.device, dtype=hidden_states.dtype).reshape(1, -1, hidden)
            image_scatter_index = index[batch_idx : batch_idx + 1].masked_select(image_mask[batch_idx : batch_idx + 1].bool()).reshape(1, -1)
            hidden_states[batch_idx : batch_idx + 1].scatter_(
                dim=1,
                index=image_scatter_index.unsqueeze(-1).repeat(1, 1, hidden),
                src=embeds,
            )
        return hidden_states

    def _instantiate_continuous_tokens(self, hidden_states, embedder_weights, values, scatter_index):
        if values is None or scatter_index is None:
            return hidden_states

        batch, _, hidden = hidden_states.shape
        if isinstance(values, list):
            for batch_idx, value in enumerate(values):
                token_emb = apply_timestep_embedder(embedder_weights, value)
                hidden_states[batch_idx : batch_idx + 1].scatter_(
                    dim=1,
                    index=scatter_index[batch_idx].unsqueeze(0).unsqueeze(-1).repeat(1, 1, hidden),
                    src=token_emb.reshape(1, -1, hidden),
                )
            return hidden_states

        token_emb = apply_timestep_embedder(embedder_weights, values.reshape(-1))
        hidden_states.scatter_(
            dim=1,
            index=scatter_index.unsqueeze(-1).repeat(1, 1, hidden),
            src=token_emb.reshape(batch, -1, hidden),
        )
        return hidden_states

    @torch.no_grad()
    def infer(self, weights, inputs):
        device = first_weight_device(weights)
        input_ids = inputs.get("input_ids")
        input_ids = to_device(input_ids, device)
        hidden_states = inputs.get("inputs_embeds")
        hidden_states = to_device(hidden_states, device)
        if hidden_states is None and input_ids is not None:
            hidden_states = weights.token_embedding.apply(input_ids)

        token_hw = None
        images = inputs.get("images")
        images = to_device(images, device)
        if images is not None:
            hidden_states, token_hw = self._instantiate_image_tokens(
                weights,
                hidden_states,
                images,
                to_device(inputs.get("image_mask"), device),
                to_device(inputs.get("timesteps"), device),
            )

        cond_vae_images = to_device(inputs.get("cond_vae_images"), device)
        if cond_vae_images is not None:
            hidden_states, _ = self._instantiate_image_tokens(
                weights,
                hidden_states,
                cond_vae_images,
                to_device(inputs.get("cond_vae_image_mask"), device),
                to_device(inputs.get("cond_timesteps"), device),
            )

        hidden_states = self._instantiate_vit_image_tokens(
            hidden_states,
            to_device(inputs.get("cond_vit_embeds"), device),
            to_device(inputs.get("cond_vit_image_mask"), device),
        )

        hidden_states = self._instantiate_continuous_tokens(
            hidden_states,
            weights.timestep_emb,
            to_device(inputs.get("timesteps"), device),
            to_device(inputs.get("timesteps_index"), device),
        )
        if self.config.get("cfg_distilled", False):
            hidden_states = self._instantiate_continuous_tokens(
                hidden_states,
                weights.guidance_emb,
                to_device(inputs.get("guidance"), device),
                to_device(inputs.get("guidance_index"), device),
            )
        if self.config.get("use_meanflow", False):
            hidden_states = self._instantiate_continuous_tokens(
                hidden_states,
                weights.timestep_r_emb,
                to_device(inputs.get("timesteps_r"), device),
                to_device(inputs.get("timesteps_r_index"), device),
            )
        hidden_states = self._instantiate_continuous_tokens(
            hidden_states,
            weights.timestep_emb,
            to_device(inputs.get("cond_timesteps"), device),
            to_device(inputs.get("cond_timestep_index"), device),
        )

        if hidden_states is None:
            raise ValueError("HunyuanImage3 pre_infer requires input_ids, inputs_embeds, or images.")

        return HunyuanImage3PreInferOutput(
            hidden_states=hidden_states,
            attention_mask=to_device(inputs.get("attention_mask"), device),
            position_ids=to_device(inputs.get("position_ids"), device),
            custom_pos_emb=to_device(inputs.get("custom_pos_emb"), device),
            past_key_values=inputs.get("past_key_values"),
            use_cache=bool(inputs.get("use_cache", False)),
            image_mask=to_device(inputs.get("image_mask"), device),
            timesteps=to_device(inputs.get("timesteps"), device),
            token_hw=token_hw,
            first_step=inputs.get("first_step"),
        )

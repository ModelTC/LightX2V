import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.models.networks.hunyuan_image3.infer.utils import apply_linear, apply_timestep_embedder, first_weight_device, to_device


class HunyuanImage3PostInfer:
    def __init__(self, config):
        self.config = config
        if config.get("tensor_parallel", False):
            self.tp_group = config["device_mesh"].get_group(mesh_dim="tensor_p")
            self.tp_size = dist.get_world_size(self.tp_group)
        else:
            self.tp_group = None
            self.tp_size = 1

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

    def _final_layer(self, weights, image_output, timesteps, token_h, token_w):
        t_emb = apply_timestep_embedder(weights.time_embed_2, timesteps)
        x = image_output.reshape(image_output.shape[0], token_h, token_w, image_output.shape[-1])
        x = x.permute(0, 3, 1, 2).contiguous()
        for block in weights.final_layer.blocks:
            x = self._apply_resblock(block, x, t_emb)
        x = weights.final_layer.out_norm.apply_group_norm(x)
        x = F.silu(x)
        return weights.final_layer.output_conv.apply(x)

    @torch.no_grad()
    def infer(self, weights, hidden_states, pre_infer_out):
        device = first_weight_device(weights)
        hidden_states = to_device(hidden_states, device)
        if pre_infer_out.image_mask is not None and pre_infer_out.timesteps is not None and pre_infer_out.token_hw is not None:
            token_h, token_w = pre_infer_out.token_hw
            image_mask = to_device(pre_infer_out.image_mask, device)
            timesteps = to_device(pre_infer_out.timesteps, device)
            if pre_infer_out.first_step is False:
                special_tokens = 1 + int(self.config.get("cfg_distilled", False)) + int(self.config.get("use_meanflow", False))
                image_output = hidden_states[:, special_tokens:, :]
            else:
                hidden = hidden_states.shape[-1]
                image_output = hidden_states.masked_select(image_mask.unsqueeze(-1).bool()).reshape(-1, token_h * token_w, hidden)
            return {"diffusion_prediction": self._final_layer(weights, image_output, timesteps, token_h, token_w)}

        # Autoregressive generation only consumes the last position. Limiting
        # the TP vocabulary projection to that token avoids materializing and
        # gathering prompt-length logits on every rank.
        logits_hidden_states = hidden_states[:, -1:, :] if self.tp_size > 1 else hidden_states
        normed = weights.final_norm.apply(logits_hidden_states)
        logits = apply_linear(weights.lm_head, normed.reshape(-1, normed.shape[-1])).reshape(*normed.shape[:-1], -1)
        if self.tp_size > 1:
            local_logits = logits.contiguous()
            gathered_logits = [torch.empty_like(local_logits) for _ in range(self.tp_size)]
            dist.all_gather(gathered_logits, local_logits, group=self.tp_group)
            logits = torch.cat(gathered_logits, dim=-1)
        return {"logits": logits}

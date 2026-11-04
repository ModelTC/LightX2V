"""Native StreamRender-H3 embedding and per-row time preparation."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from lightx2v.models.networks.minimax_h3.infer.autoregressive.streamrender.prefix_cache import StreamRenderH3PrefixBank
from lightx2v.models.networks.minimax_h3.infer.module_io import MiniMaxH3PreInferOutput
from lightx2v.models.networks.minimax_h3.infer.pre_infer import MiniMaxH3PreInfer, timestep_embedding
from lightx2v.utils.envs import GET_DTYPE


@dataclass
class StreamRenderH3PreInferOutput(MiniMaxH3PreInferOutput):
    prefix_bank: StreamRenderH3PrefixBank | None = None
    prefix_length: int = 0
    capture_prefix: bool = False
    valid_sequence_length: int = 0
    prepared_key: torch.Tensor | None = None
    prepared_value: torch.Tensor | None = None
    prepared_sage: object | None = None


class StreamRenderH3PreInfer(MiniMaxH3PreInfer):
    def __init__(self, config):
        super().__init__(config)
        self.rope = None
        self._text_context = None
        self._text_weights_offloaded = False

    def clear_text_condition(self):
        self._text_context = None

    def _text_condition(self, weights, prompt_embeds):
        if self._text_context is None:
            modules = (weights.context_embedder, weights.refiner_blocks, weights.refiner_final_norm)
            try:
                if self._text_weights_offloaded:
                    for module in modules:
                        module.to_cuda()
                projected = weights.context_embedder.apply(prompt_embeds.to(GET_DTYPE()))
                self._text_context = self._refine_text(weights, projected)
            finally:
                # Conditioning does not depend on timestep or stream history.
                # Keep its small result; the next request can upload the same
                # read-only CPU weights again without retaining a full refiner.
                for module in modules:
                    module.to_cpu()
                self._text_weights_offloaded = True
        return self._text_context

    def prepare_inputs(self, weights, prompt_embeds):
        scheduler = self.scheduler
        layout = scheduler.layout
        source = layout.source_rows.to(scheduler.video_tokens.device)
        device = scheduler.video_tokens.device
        hidden = torch.zeros((layout.valid_sequence_length, self.hidden_size), device=device, dtype=GET_DTYPE())
        image_rows = torch.cat((layout.picture_indices, layout.reference_indices, layout.video_indices)).sort().values.to(device)
        audio_rows = layout.audio_indices.to(device)

        if image_rows.numel():
            image_source = source.index_select(0, image_rows)
            image = scheduler.video_tokens.index_select(0, image_source).float()
            hidden.index_copy_(0, image_rows, weights.proj_in.apply(image).to(hidden.dtype))
        if audio_rows.numel():
            audio_source = source.index_select(0, audio_rows)
            audio = scheduler.audio_tokens.index_select(0, audio_source).float()
            hidden.index_copy_(0, audio_rows, weights.audio_proj_in.apply(audio).to(hidden.dtype))
        if layout.text_indices.numel():
            hidden.index_copy_(0, layout.text_indices.to(device), self._text_condition(weights, prompt_embeds))

        # Preserve the source's BF16 rounding, then expose those same values
        # in the FP32 compute format required by the fused Norm/RoPE backend.
        rotary = tuple(tensor.float() for tensor in self.rope.embeddings(layout.position_ids.to(device), dtype=hidden.dtype))
        return hidden, rotary

    def infer(self, weights, prompt_embeds):
        scheduler = self.scheduler
        layout = scheduler.layout
        hidden, rotary = self.prepare_inputs(weights, prompt_embeds)
        cache_adaln = self.config.get("use_adaln_cache", False)
        time_device = "cpu" if cache_adaln else hidden.device
        repo_time = scheduler.row_timesteps.to(time_device, dtype=torch.float32)
        tags = layout.token_tags.to(time_device)
        clean_anchor = repo_time.new_tensor((1.0, 0.999, 1.0))
        h3_time = torch.where(repo_time == 0, clean_anchor[tags.clamp(min=0)], 1.0 - repo_time)
        unique_time, time_indices = torch.unique(h3_time, return_inverse=True)
        if cache_adaln:
            scheduler.unique_timesteps_cpu = unique_time
            time_indices = time_indices.to(hidden.device)
            tags = tags.to(hidden.device)
            temb = None
        else:
            temb = timestep_embedding(unique_time, self.freq_dim)
            temb = weights.time_linear_2.apply(F.silu(weights.time_linear_1.apply(temb.float())))
        return StreamRenderH3PreInferOutput(
            hidden_states=hidden,
            temb=temb,
            timestep_indices=time_indices,
            adaln_indices=time_indices * 3 + tags.clamp(min=0),
            rotary_emb=rotary,
            video_indices=layout.noisy_video_indices.to(hidden.device),
            audio_indices=layout.noisy_audio_indices.to(hidden.device),
            text_indices=layout.text_indices.to(hidden.device),
            prefix_bank=scheduler.prefix_bank,
            prefix_length=layout.prefix_length,
            capture_prefix=layout.capture_prefix,
            valid_sequence_length=layout.valid_sequence_length,
        )

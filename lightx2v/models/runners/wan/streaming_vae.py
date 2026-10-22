"""Causal Wan decoder selection and chunk decode shared by AR runners."""

import torch

from lightx2v.utils.async_vae import AsyncVAEChunkDecoder
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.profiler import no_sync_profiling
from lightx2v_platform.base.global_var import AI_DEVICE


class WanStreamingVAERunnerMixin:
    def load_vae(self):
        if self.config.get("lightvae_path"):
            self.vae_encoder = self.load_vae_encoder()
            return self.vae_encoder, self.load_vae_decoder()
        return super().load_vae()

    def load_vae_decoder(self):
        if not self.config.get("lightvae_path"):
            return super().load_vae_decoder()
        from lightx2v.models.video_encoders.hf.wan.lightvae import WanLightVAEDecoder

        family = "wan22" if self.config["vae_stride"][1] == 16 else "wan21"
        scale = self.vae_encoder.scale if family == "wan22" and self.vae_encoder is not None else None
        if family == "wan22" and scale is None:
            from lightx2v.models.video_encoders.hf.wan.vae_2_2 import get_wan22_vae_scale

            dtype = self._resolve_vae_dtype(self.config.get("vae_dtype"), GET_DTYPE())
            mean, _, inv_std = get_wan22_vae_scale(dtype, AI_DEVICE)
            scale = [mean, inv_std]
        return WanLightVAEDecoder(
            self.config["lightvae_path"],
            family=family,
            scale=scale,
            dtype=GET_DTYPE(),
            device=AI_DEVICE,
            cpu_offload=self.config.get("vae_cpu_offload", self.config.get("cpu_offload", False)),
        )

    def decode_segment_latents(self, segment_idx, segment_latents):
        return self.vae_decoder.cached_decode_withflag(segment_latents.to(GET_DTYPE()), segment_idx == 0, segment_idx == self.video_segment_num - 1)

    def run_vae_decoder(self, latents):
        chunk_size = self.config["ar_config"]["num_frame_per_chunk"]
        chunks = latents.split(chunk_size, dim=1)
        decoder = AsyncVAEChunkDecoder.from_config(self.config, device=AI_DEVICE, vae_decoder=self.vae_decoder, dispatch_in_worker=True)
        try:
            with no_sync_profiling(enabled=decoder.is_async):
                for index, chunk in enumerate(chunks):
                    decoder.submit(self.vae_decoder.cached_decode_withflag, chunk.to(GET_DTYPE()), index == 0, index == len(chunks) - 1)
                return torch.cat(decoder.finish(), dim=2)
        finally:
            decoder.finish()

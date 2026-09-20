import torch


class MiniMaxH3StreamingVideoDecoder:
    """Decode 7-latent windows at stride 5, retaining the five-frame overlap."""

    def __init__(self, vae, num_frames):
        self.vae = vae
        self.num_frames = num_frames
        self.latents = None
        self.overlap = None
        self.emitted_frames = 0

    def _decode_window(self, latents: torch.Tensor) -> torch.Tensor | None:
        """Decode seven normalized latents to 28 raw frames for streaming overlap.

        Preserve spatial tiling and the decoder dtype through overlap blending,
        matching the causal checkpoint's VAE path.
        """
        if latents.ndim != 5 or latents.shape[2] != 7:
            raise ValueError("H3 streaming decode requires exactly seven latent frames")
        vae = self.vae
        try:
            device = vae._activate()
            latents = vae.denormalize_latents(latents.to(device)).to(vae.infer_dtype)
            with torch.no_grad():
                layout = vae._spatial_tile_layout(latents)
                tiles = vae._get_all_tiles(latents, 1, layout)
                if vae.decode_parallel:
                    decoded = vae._decode_parallel(tiles)
                    if decoded is None:
                        return None
                else:
                    decoded = [vae.decoder(vae.post_quant_conv(tile)) for tile in tiles]
                return vae._stitch_clip(decoded, layout.height_overlaps, layout.width_overlaps)
        finally:
            if vae.cpu_offload:
                vae.offload()

    def push(self, latents):
        self.latents = latents if self.latents is None else torch.cat((self.latents, latents), dim=2)
        outputs = []
        while self.latents.shape[2] >= 7:
            decoded = self._decode_window(self.latents[:, :, :7])
            # Parallel decode assembles windows only on rank zero. Every rank
            # must still advance the latent window before the next collective.
            if decoded is not None:
                primary = decoded[:, :, 3:20]
                if self.overlap is not None:
                    primary = self.vae._blend(self.overlap, primary, 5, dim=-3)
                outputs.append(self.vae.postprocess(primary).float().cpu())
                self.overlap = decoded[:, :, 23:28].clone()
            self.latents = self.latents[:, :, 5:].clone()
            self.emitted_frames += 17
        return outputs

    def finish(self):
        if self.latents is None or self.latents.shape[2] != 2 or self.emitted_frames + 5 != self.num_frames:
            raise RuntimeError("H3 streaming decode ended with an incomplete latent window")
        output = self.vae.postprocess(self.overlap).float().cpu() if self.overlap is not None else None
        self.latents = self.overlap = None
        self.emitted_frames += 5
        return output

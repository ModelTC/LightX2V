import torch
import torch.distributed as dist


class MiniMaxH3StreamingVideoDecoder:
    """Decode overlapping windows to CPU RGB [B,3,F,H,W], float32 or uint8."""

    def __init__(self, vae, num_frames, *, output_uint8=False, pinned_memory=False, decode_graph=None, output_writer=None):
        self.vae = vae
        self.num_frames = num_frames
        self.output_uint8 = output_uint8
        self.pinned_memory = pinned_memory
        self.latents = None
        self.overlap = None
        self.emitted_frames = 0
        self.output_buffer = None
        self.output_offset = 0
        self.decode_graph = decode_graph
        self.pending_decode = None
        self.output_writer = output_writer

    def postprocess(self, video):
        if self.output_uint8 and video.is_cuda:
            from .rgb import to_uint8

            rgb = to_uint8(video, self.vae.pixel_mean, self.vae.pixel_std, self.vae.sensitive_layer_dtype)
            if self.output_writer is not None:
                self.output_writer.write(rgb)
                return None
            if not self.pinned_memory:
                return rgb.cpu()
            if self.output_buffer is None:
                b, c, _, h, w = rgb.shape
                self.output_buffer = torch.empty((b, c, self.num_frames, h, w), device="cpu", dtype=torch.uint8, memory_format=torch.channels_last_3d, pin_memory=True)
            end = self.output_offset + rgb.shape[2]
            output = self.output_buffer[:, :, self.output_offset : end]
            output.copy_(rgb, non_blocking=True)
            torch.cuda.current_stream(rgb.device).synchronize()
            self.output_offset = end
            return output
        video = self.vae.postprocess(video).float()
        if self.output_uint8:
            # Quantize after blending, before the host copy. Channels-last
            # lets the video encoder view FHWC frames without another copy.
            video = (video * 255.0).round().to(dtype=torch.uint8, memory_format=torch.channels_last_3d)
        if self.output_writer is not None:
            self.output_writer.write(video)
            return None
        return video.cpu()

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

    def _decode_local(self, latents):
        """Decode this rank's tiles without starting any distributed operation."""
        vae = self.vae
        device = vae._activate()
        latents = vae.denormalize_latents(latents.to(device)).to(vae.infer_dtype)
        with torch.no_grad():
            layout = vae._spatial_tile_layout(latents)
            tiles = vae._get_all_tiles(latents, 1, layout)
            counts = vae._balanced_task_counts(len(tiles), dist.get_world_size())
            rank = dist.get_rank()
            start = sum(counts[:rank])
            local = [self.decode_graph(vae.post_quant_conv(tile)) for tile in tiles[start : start + counts[rank]]]
        return local, counts, layout

    def _blend_window(self, decoded):
        primary = decoded[:, :, 3:20]
        if self.overlap is not None:
            primary = self.vae._blend(self.overlap, primary, 5, dim=-3)
        else:
            primary = primary.clone()
        self.overlap = decoded[:, :, 23:28].clone()
        return primary

    def _finish_window(self):
        if self.pending_decode is None:
            return None
        (local, counts, layout), ready, _source = self.pending_decode
        current = torch.cuda.current_stream(self.decode_graph.device)
        current.wait_event(ready)
        for tile in local:
            tile.record_stream(current)
        # Gather stays on the caller's stream, ordered with DiT collectives.
        tiles = self.vae._gather_tiles(local, counts)
        self.pending_decode = None
        if tiles is None:
            return None
        decoded = self.vae._stitch_clip(tiles, layout.height_overlaps, layout.width_overlaps)
        del tiles, local
        primary = self._blend_window(decoded)
        del decoded
        return self.postprocess(primary)

    def push(self, latents):
        outputs = []
        ready = self._finish_window()
        if ready is not None:
            outputs.append(ready)
        self.latents = latents if self.latents is None else torch.cat((self.latents, latents), dim=2)
        while self.latents.shape[2] >= 7:
            window = self.latents[:, :, :7]
            first = self.emitted_frames == 0
            if self.decode_graph is None:
                decoded = self._decode_window(window)
                if decoded is not None:
                    primary = self._blend_window(decoded)
                    del decoded
                    output = self.postprocess(primary)
                    if output is not None:
                        outputs.append(output)
            else:
                stream = self.decode_graph.stream
                stream.wait_stream(torch.cuda.current_stream(self.decode_graph.device))
                window.record_stream(stream)
                with torch.cuda.stream(stream):
                    job = self._decode_local(window)
                    ready_event = torch.cuda.Event()
                    ready_event.record()
                self.pending_decode = job, ready_event, window
            self.latents = self.latents[:, :, 5:].clone()
            # Logical frames drive actions on every rank, even while RGB waits.
            self.emitted_frames += 17
            if self.decode_graph is not None and (first or self.latents.shape[2] >= 7):
                ready = self._finish_window()
                if ready is not None:
                    outputs.append(ready)
        return outputs

    def finish(self):
        start = self.output_offset
        ready = self._finish_window()
        if self.latents is None or self.latents.shape[2] != 2 or self.emitted_frames + 5 != self.num_frames:
            raise RuntimeError("H3 streaming decode ended with an incomplete latent window")
        output = self.postprocess(self.overlap) if self.overlap is not None else None
        self.latents = self.overlap = None
        self.emitted_frames += 5
        if self.output_buffer is not None:
            return self.output_buffer[:, :, start : self.output_offset]
        return torch.cat((ready, output), dim=2) if ready is not None else output

    def close(self):
        """Drain asynchronous work before timing ends or request state is freed."""
        try:
            if self.decode_graph is not None:
                torch.cuda.synchronize(self.decode_graph.device)
        finally:
            self.pending_decode = None
            self.latents = self.overlap = None

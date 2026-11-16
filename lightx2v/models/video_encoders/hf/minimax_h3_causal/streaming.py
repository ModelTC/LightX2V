import torch
import torch.distributed as dist

from lightx2v.models.networks.minimax_h3.packing import FRAMES_PER_CHUNK, LATENTS_PER_CHUNK


class MiniMaxH3StreamingVideoDecoder:
    """Decode overlapping windows to CPU RGB [B,3,F,H,W], float32 or uint8.

    Each 7-latent window advances by 5 latents and emits 17 frames.
    Keep 2 latents between windows; finish() emits the final 5 frames.
    """

    def __init__(self, vae, num_frames, *, output_uint8=False, decode_graph=None, output_writer=None):
        self.vae = vae
        self.num_frames = num_frames
        self.output_uint8 = output_uint8
        self.latents = None
        self.overlap = None
        self.emitted_frames = 0
        self.decode_graph = decode_graph
        self.pending_decode = None
        self.output_writer = output_writer
        # Compact asynchronous tile outputs before cloning, gathering and stitching.
        # Synchronous decoding retains the original 28-frame layout.
        self.output_frame_indices = torch.tensor([*range(3, 20), *range(23, 28)], device=decode_graph.device) if decode_graph is not None else None

    def postprocess(self, video):
        if self.output_uint8 and video.is_cuda:
            from .rgb import to_uint8

            video = to_uint8(video, self.vae.pixel_mean, self.vae.pixel_std, self.vae.sensitive_layer_dtype)
        else:
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
            local = [self.decode_graph(vae.post_quant_conv(tile), frame_indices=self.output_frame_indices) for tile in tiles[start : start + counts[rank]]]
        return local, counts, layout

    def _blend_window(self, decoded):
        if self.output_frame_indices is None:
            primary, tail = decoded[:, :, 3:20], decoded[:, :, 23:28]
        else:
            primary, tail = decoded[:, :, :17], decoded[:, :, 17:22]
        if self.overlap is not None:
            primary = self.vae._blend(self.overlap, primary, 5, dim=-3)
        else:
            primary = primary.clone()
        self.overlap = tail.clone()
        return primary

    def flush(self):
        """Complete the pending decode and return its CPU RGB, if any."""
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
        ready = self.flush()
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
            self.latents = self.latents[:, :, LATENTS_PER_CHUNK:].clone()
            # Logical frames drive actions on every rank, even while RGB waits.
            self.emitted_frames += FRAMES_PER_CHUNK
            if self.decode_graph is not None and (first or self.latents.shape[2] >= 7):
                ready = self.flush()
                if ready is not None:
                    outputs.append(ready)
        return outputs

    def finish(self):
        ready = self.flush()
        if self.latents is None or self.latents.shape[2] != 2 or self.emitted_frames + 5 != self.num_frames:
            raise RuntimeError("H3 streaming decode ended with an incomplete latent window")
        output = self.postprocess(self.overlap) if self.overlap is not None else None
        self.latents = self.overlap = None
        self.emitted_frames += 5
        return torch.cat((ready, output), dim=2) if ready is not None else output

    def close(self):
        """Release request tensors after the runner has synchronized the device."""
        self.pending_decode = None
        self.latents = self.overlap = None

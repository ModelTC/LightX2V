import torch
import torch.distributed as dist

from lightx2v.models.networks.minimax_h3.packing import FRAMES_PER_CHUNK, LATENTS_PER_CHUNK


class MiniMaxH3VideoWindowDecoder:
    """Shared spatial VAE decoding and CPU RGB or writer output."""

    def __init__(self, vae, *, output_uint8=False, pinned_memory=False, decode_graph=None, output_writer=None, output_rank=0):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if not 0 <= output_rank < world_size:
            raise ValueError("Video decoder output_rank must identify a rank in the process group")
        self.vae = vae
        self.output_uint8 = output_uint8
        self.pinned_memory = pinned_memory
        self.decode_graph = decode_graph
        self.output_writer = output_writer
        self.output_rank = output_rank

    def postprocess(self, video):
        if self.output_uint8 and video.is_cuda:
            from .rgb import to_uint8

            video = to_uint8(video, self.vae.pixel_mean, self.vae.pixel_std, self.vae.sensitive_layer_dtype)
        else:
            video = self.vae.postprocess(video).float()
            if self.output_uint8:
                # Quantize before the host copy. Channels-last
                # lets the video encoder view FHWC frames without another copy.
                video = (video * 255.0).round().to(dtype=torch.uint8, memory_format=torch.channels_last_3d)
        if self.output_writer is not None:
            self.output_writer.write(video)
            return None
        if self.pinned_memory and video.is_cuda:
            # Live recorders retain frame views asynchronously. Each output owns
            # its storage until those views are released, rather than reusing it.
            output = torch.empty(video.shape, dtype=video.dtype, device="cpu", pin_memory=True, memory_format=torch.channels_last_3d)
            output.copy_(video, non_blocking=True)
            torch.cuda.current_stream(video.device).synchronize()
            return output
        return video.cpu()

    def _decode_window(self, latents: torch.Tensor, *, temporal_position_offset=0.0) -> torch.Tensor | None:
        """Decode a seven-latent history/current window with spatial tiling."""
        vae = self.vae
        if not vae.decode_parallel and dist.is_initialized() and dist.get_rank() != self.output_rank:
            return None
        try:
            device = vae._activate()
            latents = vae.denormalize_latents(latents.to(device)).to(vae.infer_dtype)
            with torch.no_grad():
                layout = vae._spatial_tile_layout(latents)
                tiles = vae._get_all_tiles(latents, 1, layout)
                if vae.decode_parallel:
                    decoded = vae._decode_parallel(tiles, output_rank=self.output_rank, temporal_position_offset=temporal_position_offset)
                    if decoded is None:
                        return None
                else:
                    decoded = [vae.decoder(vae.post_quant_conv(tile), temporal_position_offset=temporal_position_offset) for tile in tiles]
                return vae._stitch_clip(decoded, layout.height_overlaps, layout.width_overlaps)
        finally:
            if vae.cpu_offload:
                vae.offload()

    def _submit_graph_window(self, latents, *, temporal_position_offset=0.0):
        vae = self.vae
        stream = self.decode_graph.stream
        stream.wait_stream(torch.cuda.current_stream(self.decode_graph.device))
        latents.record_stream(stream)
        source = latents
        with torch.cuda.stream(stream), torch.no_grad():
            device = vae._activate()
            latents = vae.denormalize_latents(latents.to(device)).to(vae.infer_dtype)
            layout = vae._spatial_tile_layout(latents)
            tiles = vae._get_all_tiles(latents, 1, layout)
            counts = vae._balanced_task_counts(len(tiles), dist.get_world_size())
            rank = (dist.get_rank() - self.output_rank) % len(counts)
            start = sum(counts[:rank])
            local = [
                self.decode_graph(vae.post_quant_conv(tile), frame_indices=self.output_frame_indices, temporal_position_offset=temporal_position_offset) for tile in tiles[start : start + counts[rank]]
            ]
            ready = torch.cuda.Event()
            ready.record()
        return local, counts, layout, ready, source

    def _gather_graph_window(self, pending):
        local, counts, layout, ready, _source = pending
        current = torch.cuda.current_stream(self.decode_graph.device)
        current.wait_event(ready)
        for tile in local:
            tile.record_stream(current)
        # Gather stays on the caller's stream, ordered with DiT collectives.
        tiles = self.vae._gather_tiles(local, counts, output_rank=self.output_rank)
        if tiles is None:
            return None
        return self.vae._stitch_clip(tiles, layout.height_overlaps, layout.width_overlaps)


class MiniMaxH3StreamingVideoDecoder(MiniMaxH3VideoWindowDecoder):
    """Decode 5+2 lookahead windows with five-frame blending and a final tail.

    Each seven-latent window advances by five latents and emits 17 frames.
    Graph decoding can overlap the next DiT chunk; finish() emits the last five.
    """

    def __init__(self, vae, num_frames, *, output_uint8=False, pinned_memory=False, decode_graph=None, output_writer=None, output_rank=0):
        super().__init__(vae, output_uint8=output_uint8, pinned_memory=pinned_memory, decode_graph=decode_graph, output_writer=output_writer, output_rank=output_rank)
        self.num_frames = num_frames
        self.latents = None
        self.overlap = None
        self.emitted_frames = 0
        self.pending_decode = None
        self.output_frame_indices = torch.tensor([*range(3, 20), *range(23, 28)], device=decode_graph.device) if decode_graph is not None else None

    def _blend_window(self, decoded):
        if self.decode_graph is None:
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
        decoded = self._gather_graph_window(self.pending_decode)
        self.pending_decode = None
        if decoded is None:
            return None
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
                self.pending_decode = self._submit_graph_window(window)
            self.latents = self.latents[:, :, LATENTS_PER_CHUNK:].clone()
            # Logical frames advance on every rank, even while RGB waits.
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


class MiniMaxH3LiveVideoDecoder(MiniMaxH3VideoWindowDecoder):
    """Decode current five-latent chunks using two history latents.

    The second push emits both the first and current chunks; subsequent pushes
    emit the current chunk immediately without waiting for future latents.
    """

    def __init__(self, vae, *, output_uint8=False, pinned_memory=False, decode_graph=None, output_rank=0):
        super().__init__(vae, output_uint8=output_uint8, pinned_memory=pinned_memory, decode_graph=decode_graph, output_rank=output_rank)
        self.latents = None
        self.output_frame_indices = torch.arange(11, 28, device=decode_graph.device) if decode_graph is not None else None

    def push(self, latents):
        if self.latents is None:
            self.latents = latents.clone()
            return []
        windows = []
        if self.latents.shape[2] == LATENTS_PER_CHUNK:
            windows.append((torch.cat((self.latents, latents[:, :, :2]), dim=2), True))
        windows.append((torch.cat((self.latents[:, :, -2:], latents), dim=2), False))
        self.latents = latents[:, :, -2:].clone()
        outputs = []
        for window, is_startup in windows:
            # Current latents start two slots later than the trained 5+2 window.
            temporal_position_offset = 0.0 if is_startup else -4.0 / 7
            # Keep the one-off startup window eager so it cannot evict the shifted graph.
            if self.decode_graph is None or is_startup:
                decoded = self._decode_window(window, temporal_position_offset=temporal_position_offset)
                frame_start = 3 if is_startup else 11
                if decoded is not None:
                    decoded = decoded[:, :, frame_start : frame_start + FRAMES_PER_CHUNK]
            else:
                pending = self._submit_graph_window(window, temporal_position_offset=temporal_position_offset)
                decoded = self._gather_graph_window(pending)
            ready = self.postprocess(decoded) if decoded is not None else None
            if ready is not None:
                outputs.append(ready)
        return outputs

    def close(self):
        """Release request tensors after the runner has synchronized the device."""
        self.latents = None

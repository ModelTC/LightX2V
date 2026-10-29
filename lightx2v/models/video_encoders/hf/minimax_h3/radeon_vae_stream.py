"""Streaming MiniMax-H3 video VAE decode for the radeon runtime (``RADEON_CORESW_STREAM_SAVE``).

Tiles are dealt round-robin over the decode ranks, so temporal clips complete in order on every rank. Rank 0 stitches
each clip as soon as its tiles are in and yields the finished frames while later clips decode. The per-clip ops are
those of ``MiniMaxH3VideoVAE.decode`` (same tiles, stitching, blending, tail crop and post-processing), so the frames
are bitwise identical to a whole-video decode.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def _finish(vae, chunk: torch.Tensor) -> torch.Tensor:
    if vae.sensitive_layer_dtype != vae.infer_dtype:
        chunk = chunk.to(vae.sensitive_layer_dtype)
    return vae.postprocess(chunk).float()


def stream_decode(vae, latents: torch.Tensor, rank: int, world: int):
    """Decode normalized diffusion latents; rank 0 yields post-processed float32 [1, 3, f, H, W] chunks in order,
    other ranks yield nothing (they decode their tiles and send them to rank 0)."""
    if latents.ndim != 5 or latents.shape[1] != vae.post_quant_conv.in_channels:
        raise ValueError(f"video latents must have shape [batch, {vae.post_quant_conv.in_channels}, frames, height, width], got {tuple(latents.shape)}")
    device = vae._activate()
    sends = []
    try:
        latents = latents.to(device=device, dtype=vae.sensitive_layer_dtype)
        latents = vae.denormalize_latents(latents)
        if vae.sensitive_layer_dtype != vae.infer_dtype:
            latents = latents.to(vae.infer_dtype)
        with torch.no_grad():
            tokens_chunk_size = vae.tokens_chunk_size
            temporal_ratio = vae.temporal_compression_ratio
            chunk_num_frames = tokens_chunk_size * temporal_ratio
            num_tokens = latents.shape[2] + vae.token_drop
            pad_tokens = (-num_tokens) % tokens_chunk_size
            num_clips = (num_tokens + pad_tokens) // tokens_chunk_size - int(vae.token_drop > 0)
            if num_clips <= 0:
                raise ValueError(f"Video latent sequence is too short for clip_length={vae.clip_length}, token_drop={vae.token_drop}: got {latents.shape[2]} frames")
            pad_frames = 0
            if pad_tokens > 0:
                latents = torch.cat([latents, latents[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
                intra_tail = vae.clip_length % temporal_ratio
                before = latents.shape[2] - pad_tokens
                pad_frames = sum(intra_tail if intra_tail and (before + offset) % tokens_chunk_size == 0 else temporal_ratio for offset in range(pad_tokens))
            layout = vae._spatial_tile_layout(latents)
            per_clip = layout.num_tiles
            if per_clip < world:
                raise ValueError(f"streaming decode needs at least one tile per rank and clip ({per_clip} tiles, {world} ranks)")
            tiles = vae._get_all_tiles(latents, num_clips, layout)
            overlap, held = None, None

            def emit(chunk):
                # hold back the last pad_frames frames: the padded tail is cropped once the video is complete
                nonlocal held
                held = chunk if held is None else torch.cat([held, chunk], dim=2)
                ready = held.shape[2] - pad_frames
                if ready <= 0:
                    return None
                out, held = held[:, :, :ready], held[:, :, ready:]
                return _finish(vae, out)

            for clip_index in range(num_clips):
                first = clip_index * per_clip
                mine = [i for i in range(first, first + per_clip) if i % world == rank]
                decoded = [vae.decoder(vae.post_quant_conv(tiles[i])) for i in mine]
                if rank != 0:
                    batch = torch.stack(decoded).contiguous()
                    sends.append((dist.isend(batch, 0), batch))
                    while len(sends) > 2:
                        sends.pop(0)[0].wait()
                    continue
                clip_tiles = [None] * per_clip
                for index, tile in zip(mine, decoded):
                    clip_tiles[index - first] = tile
                for source in range(1, world):
                    theirs = [i for i in range(first, first + per_clip) if i % world == source]
                    batch = decoded[0].new_empty((len(theirs), *decoded[0].shape))
                    dist.recv(batch, source)
                    for index, tile in zip(theirs, batch):
                        clip_tiles[index - first] = tile
                clip = vae._stitch_clip(clip_tiles, layout.height_overlaps, layout.width_overlaps)
                del clip_tiles, decoded
                for overlap_index in range(int(vae.token_drop > 0) + 1):
                    frame_start = overlap_index * chunk_num_frames
                    chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
                    chunk = chunk[:, :, vae.frame_pre_padding :]
                    if overlap_index == 0:
                        if overlap is not None:
                            chunk = vae._blend(overlap, chunk, vae.frame_overlap, dim=-3)
                        out = emit(chunk)
                        if out is not None:
                            yield out
                    else:
                        overlap = chunk
            if rank == 0 and overlap is not None:
                out = emit(overlap)
                if out is not None:
                    yield out
            for work, _ in sends:
                work.wait()
    finally:
        if vae.cpu_offload:
            vae.offload()

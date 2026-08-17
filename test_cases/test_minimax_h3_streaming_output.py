import os
import tempfile
import unittest
from types import SimpleNamespace

os.environ.setdefault("SKIP_PLATFORM_CHECK", "1")

import av  # noqa: E402
import torch  # noqa: E402

from lightx2v.models.video_encoders.hf.ltx2.audio_vae.ops import Audio  # noqa: E402
from lightx2v.models.video_encoders.hf.minimax_h3.video_vae import MiniMaxH3VideoVAE  # noqa: E402
from lightx2v.utils.ltx2_media_io import AsyncAudioVideoWriter  # noqa: E402


class _FakeTemporalDecoder:
    tokens_chunk_size = 5
    temporal_compression_ratio = 4
    token_drop = 3
    token_overlap = 2
    frame_pre_padding = 3
    frame_overlap = 5
    clip_length = 17
    decode_parallel = False

    _blend = staticmethod(MiniMaxH3VideoVAE._blend)
    _decode_iter = MiniMaxH3VideoVAE._decode_iter

    @staticmethod
    def _spatial_tile_layout(latents):
        return SimpleNamespace(num_tiles=1, height_overlaps=[], width_overlaps=[])

    def _get_all_tiles(self, latents, num_clips, spatial_layout):
        del spatial_layout
        return [
            latents[
                :,
                :,
                index * self.tokens_chunk_size : index * self.tokens_chunk_size
                + self.tokens_chunk_size
                + self.token_overlap,
            ]
            for index in range(num_clips)
        ]

    @staticmethod
    def post_quant_conv(latents):
        return latents

    @staticmethod
    def decoder(latents):
        return torch.repeat_interleave(latents, repeats=4, dim=2)

    @staticmethod
    def _stitch_clip(tiles, height_overlaps, width_overlaps):
        del height_overlaps, width_overlaps
        return tiles[0]


def _reference_decode(decoder, latents):
    tokens_chunk_size = decoder.tokens_chunk_size
    temporal_ratio = decoder.temporal_compression_ratio
    chunk_num_frames = tokens_chunk_size * temporal_ratio
    num_tokens = latents.shape[2] + decoder.token_drop
    pad_tokens = (-num_tokens) % tokens_chunk_size
    num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(decoder.token_drop > 0)
    if pad_tokens > 0:
        latents = torch.cat([latents, latents[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

    decoded_chunks = []
    overlap = None
    for chunk_index in range(num_chunks):
        start = chunk_index * tokens_chunk_size
        clip = decoder.decoder(
            decoder.post_quant_conv(
                latents[:, :, start : start + tokens_chunk_size + decoder.token_overlap]
            )
        )
        for overlap_index in range(int(decoder.token_drop > 0) + 1):
            frame_start = overlap_index * chunk_num_frames
            chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
            chunk = chunk[:, :, decoder.frame_pre_padding :]
            if overlap_index == 0:
                if overlap is not None:
                    chunk = decoder._blend(overlap, chunk, decoder.frame_overlap, dim=-3)
                decoded_chunks.append(chunk)
            else:
                overlap = chunk
    if overlap is not None:
        decoded_chunks.append(overlap)

    decoded = torch.cat(decoded_chunks, dim=2)
    if pad_tokens > 0:
        intra_tail = decoder.clip_length % temporal_ratio
        num_tokens_before_pad = latents.shape[2] - pad_tokens
        pad_frames = sum(intra_tail if intra_tail and (num_tokens_before_pad + offset) % tokens_chunk_size == 0 else temporal_ratio for offset in range(pad_tokens))
        decoded = decoded[:, :, :-pad_frames]
    return decoded


class MiniMaxH3StreamingDecodeTest(unittest.TestCase):
    def test_temporal_iterator_matches_original_concat_recipe(self):
        decoder = _FakeTemporalDecoder()
        for latent_frames in (7, 8, 9, 10, 11, 12, 17, 22):
            with self.subTest(latent_frames=latent_frames):
                latents = torch.arange(latent_frames, dtype=torch.float32).reshape(1, 1, latent_frames, 1, 1)
                chunks = list(decoder._decode_iter(latents))
                actual = torch.cat(chunks, dim=2)
                expected = _reference_decode(decoder, latents)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_aligned_h3_sequence_yields_17_frame_chunks_and_5_frame_tail(self):
        decoder = _FakeTemporalDecoder()
        latents = torch.arange(17, dtype=torch.float32).reshape(1, 1, 17, 1, 1)
        chunks = list(decoder._decode_iter(latents))
        self.assertEqual([chunk.shape[2] for chunk in chunks], [17, 17, 17, 5])


class AsyncAudioVideoWriterTest(unittest.TestCase):
    def test_writes_video_chunks_and_stereo_audio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "result.mp4")
            writer = AsyncAudioVideoWriter(
                output_path=output_path,
                fps=4,
                audio_sample_rate=32000,
                queue_size=1,
                video_codec_options={"preset": "ultrafast"},
            )
            writer.submit_video(torch.zeros((2, 16, 16, 3), dtype=torch.uint8))
            writer.submit_video(torch.full((2, 16, 16, 3), 255, dtype=torch.uint8))
            writer.submit_audio(Audio(waveform=torch.zeros((2, 32000), dtype=torch.float32), sampling_rate=32000))
            writer.finish()

            self.assertTrue(os.path.isfile(output_path))
            with av.open(output_path) as container:
                video_stream = container.streams.video[0]
                audio_stream = container.streams.audio[0]
                self.assertEqual(video_stream.codec_context.name, "h264")
                self.assertEqual(video_stream.width, 16)
                self.assertEqual(video_stream.height, 16)
                self.assertEqual(audio_stream.codec_context.name, "aac")
                self.assertEqual(audio_stream.codec_context.sample_rate, 32000)
                self.assertEqual(audio_stream.codec_context.layout.name, "stereo")
                self.assertEqual(sum(1 for _ in container.decode(video=0)), 4)


if __name__ == "__main__":
    unittest.main()

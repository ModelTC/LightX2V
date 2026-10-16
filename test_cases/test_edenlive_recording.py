import av
import numpy as np
import pytest
import torch
from PIL import Image

from lightx2v.models.input_encoders.edenlive.recording import RecordingReader, prepare_recording


@pytest.mark.parametrize("fps,max_frames,pixel_frames,source_indices", [(9, None, 9, [0, 5, 11]), (120, 17, 17, [0, 0, 1, 1, 2])])
def test_recording_samples_only_latent_frames(tmp_path, monkeypatch, fps, max_frames, pixel_frames, source_indices):
    video = tmp_path / "recording.mp4"
    cover = tmp_path / "cover.png"
    Image.new("RGB", (80, 48), (255, 255, 255)).save(cover)
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=12)
        stream.width, stream.height, stream.pix_fmt = 80, 48, "yuv420p"
        for index in range(12):
            pixels = np.full((48, 80, 3), (index * 17, index * 11, index * 7), dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(video)) as container:
        colors = [frame.to_ndarray(format="rgb24")[24, 40].copy() for frame in container.decode(video=0)]
    original_resize = Image.Image.resize
    resized = []

    def resize(image, *args, **kwargs):
        resized.append(1)
        return original_resize(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", resize)
    proxy, image, frames = prepare_recording(str(video), str(cover), 32, 64, fps, max_frames)
    expected = torch.from_numpy(np.stack([colors[index] for index in source_indices])).T
    torch.testing.assert_close(proxy, expected[:, :, None, None].expand_as(proxy), rtol=0, atol=0)
    torch.testing.assert_close(image, torch.ones_like(image), rtol=0, atol=3e-7)
    assert frames == pixel_frames
    assert proxy.dtype == torch.uint8
    assert len(resized) == len(set(source_indices))


@pytest.fixture
def vfr_recording(tmp_path):
    from fractions import Fraction

    video, cover = tmp_path / "vfr.mp4", tmp_path / "cover.png"
    Image.new("RGB", (80, 48), "white").save(cover)
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 80, 48, "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for index, timestamp in enumerate([100, 200, 300, 400, 600, 650, 830, 1070, 1100, 1310, 1490, 1600]):
            pixels = np.full((48, 80, 3), (index * 17, index * 11, index * 7), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return str(video), str(cover)


@pytest.mark.parametrize("fps,max_frames", [(16, None), (120, 17), (120, None)])
def test_streaming_matches_full_timestamp_sampling(vfr_recording, fps, max_frames):
    video, cover = vfr_recording
    with av.open(video) as container:
        decoded = list(container.decode(video=0))
        times = np.array([float(frame.time) for frame in decoded])
    proxy, _, pixel_frames = prepare_recording(video, cover, 48, 80, fps, max_frames)
    times -= times[0]
    indices = np.searchsorted((times[:-1] + times[1:]) / 2, np.arange(0, pixel_frames, 4) / fps)
    expected = torch.from_numpy(np.stack([decoded[index].to_ndarray(format="rgb24") for index in indices])).permute(3, 0, 1, 2)
    torch.testing.assert_close(proxy, expected, rtol=0, atol=0)
    chunks = []
    with RecordingReader(video, 48, 80, fps, max_frames, chunk_size=3) as reader:
        while reader.remaining:
            chunks.append(reader.next_chunk())
    torch.testing.assert_close(torch.cat(chunks, dim=1), expected, rtol=0, atol=0)
    assert all(chunk.shape[1] == 3 for chunk in chunks[:-1])


def test_reader_prefetch_is_bounded_and_closes_on_early_stop(vfr_recording, monkeypatch):
    decoded, closed = [], []

    def frames(self):
        try:
            for index in range(len(self.targets)):
                decoded.append(index)
                yield np.zeros((48, 80, 3), dtype=np.uint8)
        finally:
            closed.append(True)

    monkeypatch.setattr(RecordingReader, "_decode", frames)
    with RecordingReader(vfr_recording[0], 48, 80, 120, None, chunk_size=3) as reader:
        assert reader.next_chunk().shape == (3, 3, 48, 80)
    assert 3 <= len(decoded) <= 6
    assert closed == [True]


def test_reader_propagates_decode_errors(vfr_recording, monkeypatch):
    def broken(self):
        yield np.zeros((48, 80, 3), dtype=np.uint8)
        raise RuntimeError("decode failed")

    monkeypatch.setattr(RecordingReader, "_decode", broken)
    with RecordingReader(vfr_recording[0], 48, 80, 16, None, chunk_size=3) as reader:
        with pytest.raises(RuntimeError, match="decode failed"):
            reader.next_chunk()


@pytest.mark.parametrize("fail", [False, True])
def test_runner_closes_recording_on_success_and_failure(fail):
    from types import SimpleNamespace

    from lightx2v.models.runners.wan.edenlive_runner import EdenLiveRunner

    class Pipeline:
        def run_pipeline(self, input_info):
            if fail:
                raise RuntimeError("encoder failed")
            return "video"

    class Runner(EdenLiveRunner, Pipeline):
        pass

    runner = Runner.__new__(Runner)
    closed = []
    runner.recording_reader = SimpleNamespace(close=lambda: closed.append(True))
    if fail:
        with pytest.raises(RuntimeError, match="encoder failed"):
            runner.run_pipeline(None)
    else:
        assert runner.run_pipeline(None) == "video"
    assert runner.recording_reader is None and closed == [True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_uint8_dino_preprocessing_matches_original_rounding():
    import torch.nn.functional as F

    from lightx2v.models.input_encoders.edenlive.proxy_dino import ProxyDINO

    pixels = torch.arange(256, dtype=torch.uint8).view(1, 16, 16, 1).expand(-1, -1, -1, 3).contiguous().permute(0, 3, 1, 2)
    original = (pixels.float() / 127.5 - 1).cuda().add_(1).mul_(0.5)
    expected = F.interpolate(original, size=(480, 832), mode="bilinear", align_corners=False, antialias=True)
    expected.sub_(expected.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]).div_(expected.new_tensor((0.229, 0.224, 0.225))[None, :, None, None])
    encoder = ProxyDINO.__new__(ProxyDINO)
    encoder.resize_mode = "bilinear"
    values = torch.arange(256, dtype=torch.float32).div_(127.5).sub_(1)
    encoder.pixel_values = values.add_(1).mul_(0.5).cuda()
    actual = encoder.preprocess(pixels.cuda())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

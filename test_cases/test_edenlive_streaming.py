import threading

import pytest
import torch

from lightx2v.models.video_encoders.hf.tae import TAEHV
from lightx2v.utils.async_vae import AsyncVAEChunkDecoder


def test_tae_chunk_history_and_first_frame_trimming():
    torch.manual_seed(3)
    model = TAEHV(checkpoint_path=None, model_type="wan22").eval()
    latents = torch.randn(1, 7, 48, 1, 2)
    with torch.no_grad():
        expected = model.decode_video(latents, parallel=False, show_progress_bar=False)
        for _ in range(2):
            memory = [None] * len(model.decoder)
            chunks = [model.decode_video(latents[:, start : start + 3], parallel=False, show_progress_bar=False, memory=memory, first_chunk=start == 0) for start in range(0, 7, 3)]
            assert [chunk.shape[1] for chunk in chunks] == [9, 12, 4]
            torch.testing.assert_close(torch.cat(chunks, dim=1), expected, rtol=0, atol=0)


@pytest.mark.parametrize("asynchronous,dispatch_in_worker", [(False, False), (True, False), (True, True)])
def test_chunk_callback_ready_order_and_error_propagation(asynchronous, dispatch_in_worker):
    if asynchronous and not torch.cuda.is_available():
        pytest.skip("CUDA required for asynchronous callbacks")
    device = "cuda" if asynchronous else "cpu"
    ready = threading.Event()
    received = []

    def callback(pixels, index):
        received.append((index, pixels.cpu().tolist()))
        ready.set()

    decoder = AsyncVAEChunkDecoder(enabled=asynchronous, device=device, on_chunk=callback, dispatch_in_worker=dispatch_in_worker)
    decoder.submit(torch.square, torch.tensor([2.0], device=device))
    assert ready.wait(timeout=5), "first chunk must arrive before finish()"
    decoder.submit(torch.square, torch.tensor([3.0], device=device))
    chunks = decoder.finish()
    assert received == [(0, [4.0]), (1, [9.0])]
    assert len(chunks) == 2 and decoder.finish() == []

    def fail(pixels, index):
        raise RuntimeError("consumer failed")

    decoder = AsyncVAEChunkDecoder(enabled=asynchronous, device=device, on_chunk=fail, dispatch_in_worker=dispatch_in_worker)
    with pytest.raises(RuntimeError, match="consumer failed"):
        decoder.submit(torch.square, torch.ones(1, device=device))
        decoder.finish()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_async_submit_does_not_wait_for_vae_dispatch():
    entered = threading.Event()
    release = threading.Event()

    def decode(latents):
        entered.set()
        assert release.wait(timeout=5)
        return latents.square()

    decoder = AsyncVAEChunkDecoder(True, device="cuda", dispatch_in_worker=True)
    decoder.submit(decode, torch.ones(1, device="cuda"))
    try:
        assert entered.wait(timeout=5)
        # The inference thread can continue while the decoder is still dispatching.
        assert not release.is_set()
    finally:
        release.set()
    assert len(decoder.finish()) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wait_for_chunk_preserves_history_and_caller_stream():
    decoder = AsyncVAEChunkDecoder.from_config({"ar_config": {"async_vae_decode": False, "overlap_kv_vae": True}}, device="cuda", dispatch_in_worker=True)
    assert decoder.is_async
    stream = torch.cuda.current_stream()
    history = []

    def decode(value):
        output = value if not history else value + history[-1]
        history.append(output)
        return output

    for value in [2.0, 3.0]:
        decoder.submit(decode, torch.tensor([value], device="cuda"))
        output = decoder.wait_for_chunk()
        assert torch.cuda.current_stream() == stream
        assert output.item() == sum([2.0] if value == 2.0 else [2.0, 3.0])
    assert [chunk.item() for chunk in decoder.finish()] == [2.0, 5.0]

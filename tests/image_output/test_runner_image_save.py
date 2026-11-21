from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from lightx2v.models.runners.minimax_h3_image.minimax_h3_image_runner import MiniMaxH3ImageRunner
from lightx2v.utils.image_io import save_image


@pytest.mark.parametrize("method", ["pil", "opencv"])
@pytest.mark.parametrize("suffix", [".jpg", ".webp", ".png"])
def test_runner_quality_reaches_saved_file(tmp_path, method, suffix):
    pixels = np.random.default_rng(42).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    decoded = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None, :, None].float() / 255
    runner = MiniMaxH3ImageRunner.__new__(MiniMaxH3ImageRunner)
    runner.config = {"save_image_method": method}
    runner.output_size = pixels.shape[:2]
    results = []
    for quality in (10, 95):
        path = tmp_path / f"{quality}{suffix}"
        runner.input_info = SimpleNamespace(return_result_tensor=False, save_result_path=str(path), output_compression=quality)
        runner.process_images_after_vae_decoder(decoded)
        with Image.open(path) as image:
            assert image.size == (96, 64)
        results.append(path.read_bytes())
    if suffix == ".png":
        assert results[0] == results[1]
    else:
        assert len(results[0]) < len(results[1])
        assert results[0] != results[1]


@pytest.mark.parametrize("method", ["pil", "opencv"])
def test_shared_writer_preserves_unspecified_quality(tmp_path, method):
    pixels = np.random.default_rng(3).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    image = Image.fromarray(pixels) if method == "pil" else pixels
    actual = tmp_path / "actual.jpg"
    expected = tmp_path / "expected.jpg"
    save_image(image, actual, method)
    if method == "pil":
        image.save(expected)
    else:
        assert cv2.imwrite(str(expected), image)
    assert actual.read_bytes() == expected.read_bytes()


@pytest.mark.parametrize("quality", [-1, 101, 50.5, True, "95"])
def test_writer_rejects_invalid_quality(tmp_path, quality):
    with pytest.raises(ValueError, match="output_compression"):
        save_image(Image.new("RGB", (32, 32)), tmp_path / "invalid.jpg", "pil", output_compression=quality)

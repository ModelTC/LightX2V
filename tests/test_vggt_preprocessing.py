import numpy as np
import pytest
import torch
from PIL import Image
from reconstruction_imports import cpu_imports
from torchvision.transforms.functional import to_tensor

with cpu_imports():
    from lightx2v.models.networks.vggt.utils.image_processor import load_images


def test_parallel_preprocessing_preserves_pixels_and_order(tmp_path):
    paths = []
    expected = []
    for index, shape in enumerate([(23, 37), (44, 18), (11, 11)]):
        generator = np.random.default_rng(index)
        image = Image.fromarray(generator.integers(0, 256, (*shape, 4), dtype=np.uint8))
        path = tmp_path / f"{index}.png"
        image.save(path)
        paths.append(path)
        rgb = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image).convert("RGB")
        width, height = rgb.size
        side = max(width, height)
        square = Image.new("RGB", (side, side))
        square.paste(rgb, ((side - width) // 2, (side - height) // 2))
        expected.append(to_tensor(square.resize((28, 28), Image.Resampling.BICUBIC)))
    serial, serial_coords = load_images(paths, target_size=28, workers=1)
    parallel, parallel_coords = load_images(paths, target_size=28, workers=8)
    torch.testing.assert_close(serial, torch.stack(expected), rtol=0, atol=0)
    torch.testing.assert_close(parallel, serial, rtol=0, atol=0)
    torch.testing.assert_close(parallel_coords, serial_coords, rtol=0, atol=0)


@pytest.mark.parametrize("paths,workers", [([], 1), (["unused"], 0)])
def test_invalid_preprocessing_request(paths, workers):
    with pytest.raises(ValueError):
        load_images(paths, workers=workers)

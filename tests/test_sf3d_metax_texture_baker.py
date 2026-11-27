"""The MetaX backend preserves the released CPU baking computation."""

import pytest
import torch

from lightx2v.models.networks.sf3d.utils.metax_texture_baker import MetaXTextureBaker


def test_texture_baking_preserves_cpu_results():
    texture_baker = pytest.importorskip("texture_baker")
    original = texture_baker.TextureBaker()
    adapter = MetaXTextureBaker(original)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    uv = torch.tensor([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]])
    faces = torch.tensor([[0, 1, 2]])
    attributes = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    reference_raster = original.rasterize(uv, faces, 16)
    reference = original.interpolate(attributes, reference_raster, faces)
    raster = adapter.rasterize(uv.to(device), faces.to(device), 16)
    result = adapter.interpolate(attributes.to(device), raster, faces.to(device))
    assert result.device.type == device
    assert torch.equal(reference_raster, raster.cpu())
    assert torch.equal(original.get_mask(reference_raster), adapter.get_mask(raster).cpu())
    assert torch.equal(reference, result.cpu())

"""Run the released texture-baking operators on CPU for MetaX tensors."""

from torch import nn


class MetaXTextureBaker(nn.Module):
    def __init__(self, baker):
        super().__init__()
        self.baker = baker

    def rasterize(self, uv, face_indices, bake_resolution):
        raster = self.baker.rasterize(uv.cpu(), face_indices.cpu(), bake_resolution)
        return raster.to(uv.device)

    def get_mask(self, raster):
        return self.baker.get_mask(raster)

    def interpolate(self, attributes, raster, face_indices):
        result = self.baker.interpolate(attributes.cpu(), raster.cpu(), face_indices.cpu())
        return result.to(attributes.device)

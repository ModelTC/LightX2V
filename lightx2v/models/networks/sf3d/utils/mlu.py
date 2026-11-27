"""Run upstream SF3D CPU UV and baking algorithms for MLU tensors.

Keep transfers inside request timing; neural material decoding stays on MLU.
"""

from torch import nn


class MLUTextureBaker(nn.Module):
    def __init__(self, baker):
        super().__init__()
        self.baker = baker

    def rasterize(self, uv, face_indices, bake_resolution):
        return self.baker.rasterize(uv.cpu(), face_indices.cpu(), bake_resolution).to(uv.device)

    def get_mask(self, raster):
        return self.baker.get_mask(raster)

    def interpolate(self, attributes, raster, face_indices):
        return self.baker.interpolate(attributes.cpu(), raster.cpu(), face_indices.cpu()).to(attributes.device)


class MLUUnwrapper:
    def __init__(self, unwrapper):
        self.unwrapper = unwrapper

    def __call__(self, vertices, normals, faces, island_padding):
        uv, indices = self.unwrapper(vertices.cpu(), normals.cpu(), faces.cpu(), island_padding)
        return uv.to(vertices.device), indices.to(faces.device)

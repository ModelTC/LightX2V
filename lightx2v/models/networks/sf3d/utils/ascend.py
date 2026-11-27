"""Ascend equivalents for SF3D, selected only for NPU tensors.

fold2d implements col2im overlap summation, including stride/padding/dilation.
No input, channel, patch, or overlap is discarded. It runs on the input device
and retains dtype/autograd. This is a correctness implementation, not a speed
claim. AscendTextureBaker uses the upstream extension's existing CPU backend
for geometry baking, retaining full UV/material outputs.
"""

import torch


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


def fold2d(columns, output_size, kernel_size, dilation=1, padding=0, stride=1):
    """Equivalent to F.fold for floating 2D/3D columns, without NPU col2im.

    Each kernel offset contributes a strided rectangular patch. Summing the
    patches implements fold's overlap sum, not an average or an overwrite.
    """
    if columns.ndim not in (2, 3):
        raise ValueError("fold2d expects (C*kH*kW,L) or (N,C*kH*kW,L)")
    unbatched = columns.ndim == 2
    x = columns.unsqueeze(0) if unbatched else columns
    h, w = _pair(output_size)
    kh, kw = _pair(kernel_size)
    dh, dw = _pair(dilation)
    ph, pw = _pair(padding)
    sh, sw = _pair(stride)
    if min(h, w, kh, kw, dh, dw, sh, sw) <= 0 or min(ph, pw) < 0:
        raise ValueError("Invalid fold dimensions")
    lh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    lw = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    if lh <= 0 or lw <= 0 or x.shape[-1] != lh * lw or x.shape[1] % (kh * kw):
        raise ValueError("Columns do not match the requested fold geometry")
    n, ck, _ = x.shape
    c = ck // (kh * kw)
    patches = x.reshape(n, c, kh, kw, lh, lw)
    result = x.new_zeros((n, c, h + 2 * ph, w + 2 * pw))
    for i in range(kh):
        for j in range(kw):
            y, z = i * dh, j * dw
            result[:, :, y:y + lh * sh:sh, z:z + lw * sw:sw] += patches[:, :, i, j]
    result = result[:, :, ph:ph + h, pw:pw + w]
    return result.squeeze(0) if unbatched else result


class AscendTextureBaker(torch.nn.Module):
    """Use SF3D's released CPU C++ rasterizer/interpolator, then return to NPU.

    The transfers and CPU work must remain inside request end-to-end timing.
    No geometry-only shortcut or alternate material model is used.
    """

    def __init__(self, baker):
        super().__init__()
        self.baker = baker

    def rasterize(self, uv, face_indices, bake_resolution):
        return self.baker.rasterize(uv.cpu(), face_indices.cpu(), bake_resolution).to(uv.device)

    def get_mask(self, raster):
        return self.baker.get_mask(raster)

    def interpolate(self, attributes, raster, face_indices):
        return self.baker.interpolate(attributes.cpu(), raster.cpu(), face_indices.cpu()).to(attributes.device)


class AscendUnwrapper:
    """Preserve the upstream CPU UV algorithm, face indices and island padding."""

    def __init__(self, unwrapper):
        self.unwrapper = unwrapper

    def __call__(self, vertices, normals, faces, island_padding):
        uv, indices = self.unwrapper(vertices.cpu(), normals.cpu(), faces.cpu(), island_padding)
        return uv.to(vertices.device), indices.to(faces.device)

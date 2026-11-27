from lightx2v.models.networks.native_components.ascend import preserve_native_precision

from .weights.loader import load_sf3d


class SF3DModel:
    def __init__(self, config):
        self.weights = load_sf3d(config)

    @preserve_native_precision
    def generate(self, image, bake_resolution=1024):
        return self.weights.run_image([image], bake_resolution=bake_resolution)

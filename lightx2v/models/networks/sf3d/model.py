from .weights.loader import load_sf3d


class SF3DModel:
    def __init__(self, config):
        self.weights = load_sf3d(config)

    def generate(self, image, bake_resolution=1024):
        return self.weights.run_image([image], bake_resolution=bake_resolution)

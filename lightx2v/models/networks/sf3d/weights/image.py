from dataclasses import dataclass

from ..utils.base import BaseModule


class DINOV2SingleImageTokenizer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = ""
        width: int = 512
        height: int = 512
        modulation_cond_dim: int = 768

    def configure(self):
        self.weights = None

    def forward(self, images, modulation_cond, **kwargs):
        from ..infer.vision import dinov2_image

        return dinov2_image(self.weights, images, modulation_cond)

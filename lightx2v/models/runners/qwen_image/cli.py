from lightx2v.models.runners.request_args import IMAGE
from lightx2v.utils.request_spec import Argument, RequestSpec


class QwenImageSpec(RequestSpec):
    def get_arguments(self, config, task):
        arguments = super().get_arguments(config, task)
        if task == "i2i" and config.get("layered", False):
            arguments.pop("i2i_denoise_strength", None)
        return arguments


SPECS = {
    "qwen_image": QwenImageSpec(
        {
            "t2i": IMAGE,
            "i2i": dict(
                IMAGE,
                i2i_denoise_strength=Argument(type=float, help="Image edit strength in [0, 1]; 0 preserves the input and 1 redraws it."),
                image_path=Argument(help="Input image path."),
            ),
        },
    ),
}

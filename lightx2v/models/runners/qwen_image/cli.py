from lightx2v.models.runners.request_args import IMAGE, fields
from lightx2v.utils.request_spec import RequestSpec


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
            "i2i": IMAGE | fields("i2i_denoise_strength", "image_path"),
        },
    ),
}

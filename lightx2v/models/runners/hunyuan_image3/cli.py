import argparse

from lightx2v.models.runners.request_args import COMMON
from lightx2v.utils.request_spec import Argument, RequestSpec


class HunyuanImage3Spec(RequestSpec):
    def get_arguments(self, config, task):
        arguments = super().get_arguments(config, task)
        if config.get("image_size", None):
            arguments.pop("size", None)
        return arguments


TEXT = COMMON | {
    "bot_task": Argument(help="HunyuanImage3 text generation mode."),
    "max_new_tokens": Argument(type=int, help="Maximum number of generated text tokens."),
    "prompt": Argument(help="Text prompt."),
    "stream_callback": Argument(cli=False),
    "system_prompt": Argument(help="System prompt for text generation."),
    "text_do_sample": Argument(action=argparse.BooleanOptionalAction, help="Enable sampling during text generation."),
    "text_temperature": Argument(type=float, help="Text sampling temperature."),
    "text_top_k": Argument(type=int, help="Top-k text sampling limit."),
    "text_top_p": Argument(type=float, help="Top-p text sampling threshold."),
}
REFERENCE_IMAGE = {
    "image_path": Argument(help="Input image paths, separated by commas."),
    "align_image_size": Argument(action=argparse.BooleanOptionalAction, help="Align HunyuanImage3 reference image sizes during inference."),
}
IMAGE_OUTPUT = {"prompt": Argument(help="Text prompt."), "size": Argument(type=int, nargs="+", help="Output size in pixels: HEIGHT WIDTH")}

SPECS = {
    "hunyuan_image3": HunyuanImage3Spec(
        {
            "t2t": TEXT,
            "ti2t": TEXT | REFERENCE_IMAGE,
            "t2i": COMMON | IMAGE_OUTPUT,
            "ti2i": COMMON | REFERENCE_IMAGE | IMAGE_OUTPUT,
            "i2i": COMMON | REFERENCE_IMAGE | IMAGE_OUTPUT,
        },
    ),
}

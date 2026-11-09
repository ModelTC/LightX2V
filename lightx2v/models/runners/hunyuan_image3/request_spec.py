import argparse

from lightx2v.models.runners.request_args import COMMON
from lightx2v.utils.request_spec import Argument, MultiTaskSpec, TaskSpec

REFERENCE_IMAGE = {
    "image_path": Argument(help="Input image paths, separated by commas."),
    "align_image_size": Argument(type=bool, action=argparse.BooleanOptionalAction, help="Align HunyuanImage3 reference image sizes during inference."),
}
IMAGE_OUTPUT = {"prompt": Argument(help="Text prompt."), "size": Argument(type=list[int], help="Output size in pixels: HEIGHT WIDTH")}

SPECS = {
    "hunyuan_image3": MultiTaskSpec(
        {
            "t2i": TaskSpec(COMMON | IMAGE_OUTPUT, outputs=("image",)),
            "ti2i": TaskSpec(COMMON | REFERENCE_IMAGE | IMAGE_OUTPUT, outputs=("image",)),
            "i2i": TaskSpec(COMMON | REFERENCE_IMAGE | IMAGE_OUTPUT, outputs=("image",)),
        },
    ),
}

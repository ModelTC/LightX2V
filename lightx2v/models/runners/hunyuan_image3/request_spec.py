from lightx2v.models.runners.request_args import COMMON, IMAGE
from lightx2v.utils.request_spec import Argument, MultiTaskSpec

REFERENCE_IMAGE = {
    "image_path": Argument(input_info_default=[], type=list[str], help="Input image paths."),
    "align_image_size": Argument(type=bool, help="Align HunyuanImage3 reference image sizes during inference."),
}
IMAGE_OUTPUT = {"prompt": Argument(input_info_default="", help="Text prompt."), "size": Argument(input_info_default=[], type=list[int], help="Output size in pixels: HEIGHT WIDTH")}

SPECS = {
    "hunyuan_image3": MultiTaskSpec(
        {
            "t2i": COMMON | IMAGE_OUTPUT,
            "ti2i": COMMON | REFERENCE_IMAGE | IMAGE_OUTPUT,
            "i2i": COMMON | REFERENCE_IMAGE | IMAGE_OUTPUT,
        },
        context=IMAGE,
    ),
}

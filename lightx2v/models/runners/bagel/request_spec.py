from lightx2v.models.runners.request_args import COMMON, IMAGE
from lightx2v.utils.request_spec import Argument, MultiTaskSpec

SPECS = {
    "bagel": MultiTaskSpec(
        {
            "t2i": dict(
                COMMON,
                aspect_ratio=Argument(input_info_default="", help="Output aspect ratio."),
                prompt=Argument(input_info_default="", help="Text prompt."),
                size=Argument(input_info_default=[], type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
            ),
            "i2i": dict(
                COMMON,
                image_path=Argument(input_info_default="", help="Input image path."),
                prompt=Argument(input_info_default="", help="Text prompt."),
                size=Argument(input_info_default=[], type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
            ),
        },
        context=IMAGE,
    ),
}

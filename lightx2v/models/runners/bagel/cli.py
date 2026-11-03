from lightx2v.models.runners.request_args import COMMON
from lightx2v.utils.request_spec import Argument, RequestSpec

SPECS = {
    "bagel": RequestSpec(
        {
            "t2i": COMMON
            | {"aspect_ratio": Argument(help="Output aspect ratio."), "prompt": Argument(help="Text prompt."), "size": Argument(type=int, nargs="+", help="Output size in pixels: HEIGHT WIDTH")},
            "i2i": COMMON
            | {
                "image_path": Argument(help="Input image path."),
                "prompt": Argument(help="Text prompt."),
                "size": Argument(type=int, nargs="+", help="Output size in pixels: HEIGHT WIDTH"),
            },
        },
    ),
}

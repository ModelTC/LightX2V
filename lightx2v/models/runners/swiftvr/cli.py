from lightx2v.models.runners.request_args import COMMON
from lightx2v.utils.request_spec import Argument, RequestSpec

SPECS = {
    "swiftvr": RequestSpec(
        {
            "sr": dict(
                COMMON,
                image_path=Argument(help="Input image path."),
                sr_ratio=Argument(type=float, help="super resolution ratio for sr task"),
                size=Argument(type=int, nargs="+", help="Output size in pixels: HEIGHT WIDTH"),
                video_path=Argument(help="Input video path."),
            )
        }
    ),
}

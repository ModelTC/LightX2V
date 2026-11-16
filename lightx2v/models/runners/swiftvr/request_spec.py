from lightx2v.models.runners.request_args import COMMON
from lightx2v.utils.request_spec import Argument, RequestSpec

SPECS = {
    "swiftvr": RequestSpec(
        {
            "sr": dict(
                COMMON,
                image_path=Argument(input_info_default="", help="Input image path."),
                sr_ratio=Argument(input_info_default=2.0, type=float, gt=0, help="super resolution ratio for sr task"),
                size=Argument(input_info_default=[], type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
                video_path=Argument(input_info_default="", help="Input video path."),
            )
        },
    ),
}

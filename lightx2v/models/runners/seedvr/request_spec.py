from lightx2v.models.runners.request_args import COMMON
from lightx2v.utils.request_spec import Argument, RequestSpec

SPECS = {
    "seedvr2": RequestSpec(
        {
            "sr": dict(
                COMMON,
                image_path=Argument(input_info_default="", help="Input image path."),
                match_target_size=Argument(input_info_default=True, type=bool, help="(SeedVR sr) Crop or resize decoded output to size; defaults to the startup config."),
                size=Argument(input_info_default=[], type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
                sr_ratio=Argument(input_info_default=2.0, type=float, gt=0, help="super resolution ratio for sr task"),
                video_path=Argument(input_info_default="", help="Input video path."),
            )
        },
    ),
}

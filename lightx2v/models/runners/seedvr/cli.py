import argparse

from lightx2v.models.runners.request_args import COMMON
from lightx2v.utils.request_spec import Argument, RequestSpec

SPECS = {
    "seedvr2": RequestSpec(
        {
            "sr": dict(
                COMMON,
                image_path=Argument(help="Input image path."),
                match_target_size=Argument(action=argparse.BooleanOptionalAction, help="(SeedVR sr) Crop or resize decoded output to size; defaults to the startup config."),
                size=Argument(type=int, nargs="+", help="Output size in pixels: HEIGHT WIDTH"),
                sr_ratio=Argument(type=float, help="super resolution ratio for sr task"),
                video_path=Argument(help="Input video path."),
            )
        }
    ),
}

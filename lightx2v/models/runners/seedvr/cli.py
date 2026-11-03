import argparse

from lightx2v.models.runners.request_args import COMMON
from lightx2v.utils.request_spec import Argument, RequestSpec


class SeedVRSpec(RequestSpec):
    def get_arguments(self, config, task):
        arguments = super().get_arguments(config, task)
        if config.get("seq_parallel", False) or (config.get("parallel", False) or {}).get("seq_p_size", 1) > 1:
            arguments.pop("image_path", None)
        return arguments


SPECS = {
    "seedvr2": SeedVRSpec(
        {
            "sr": COMMON
            | {
                "image_path": Argument(help="Input image path."),
                "match_target_size": Argument(action=argparse.BooleanOptionalAction, help="(SeedVR sr) Crop or resize decoded output to size; defaults to the startup config."),
                "size": Argument(type=int, nargs="+", help="Output size in pixels: HEIGHT WIDTH"),
                "sr_ratio": Argument(type=float, help="super resolution ratio for sr task"),
                "video_path": Argument(help="Input video path."),
            }
        }
    ),
}

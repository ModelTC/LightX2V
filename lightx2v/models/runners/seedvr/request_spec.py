from lightx2v.models.runners.request_args import COMMON, ImageVideoTaskSpec
from lightx2v.utils.request_spec import Argument, RequestSpec

SPECS = {
    "seedvr2": RequestSpec(
        {
            "sr": ImageVideoTaskSpec(
                dict(
                    COMMON,
                    image_path=Argument(help="Input image path."),
                    match_target_size=Argument(type=bool, help="(SeedVR sr) Crop or resize decoded output to size; defaults to the startup config."),
                    size=Argument(type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
                    sr_ratio=Argument(type=float, gt=0, help="super resolution ratio for sr task"),
                    video_path=Argument(help="Input video path."),
                )
            )
        }
    ),
}

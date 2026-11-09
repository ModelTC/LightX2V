from lightx2v.models.runners.request_args import COMMON, ImageVideoTaskSpec
from lightx2v.utils.request_spec import Argument, RequestSpec

SPECS = {
    "swiftvr": RequestSpec(
        {
            "sr": ImageVideoTaskSpec(
                dict(
                    COMMON,
                    image_path=Argument(help="Input image path."),
                    sr_ratio=Argument(type=float, gt=0, help="super resolution ratio for sr task"),
                    size=Argument(type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
                    video_path=Argument(help="Input video path."),
                )
            )
        }
    ),
}

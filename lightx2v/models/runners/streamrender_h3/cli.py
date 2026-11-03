from lightx2v.models.runners.request_args import COMMON, VIDEO_OUTPUT
from lightx2v.utils.request_spec import Argument, MultiTaskSpec

SPECS = {
    "streamrender_h3": MultiTaskSpec(
        {
            "refv2v": COMMON
            | VIDEO_OUTPUT
            | {
                "prompt": Argument(help="Text prompt."),
                "image_path": Argument(help="Input image path."),
                "video_path": Argument(help="Input video path."),
            }
        }
    ),
}

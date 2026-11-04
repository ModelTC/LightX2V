from lightx2v.models.runners.request_args import COMMON
from lightx2v.utils.request_spec import Argument, RequestSpec

RELAY_VSR = RequestSpec(
    {
        "sr": dict(
            COMMON,
            video_path=Argument(help="Input video path."),
            image_path=Argument(help="Input image path."),
            sr_ratio=Argument(type=float, help="super resolution ratio for sr task"),
        )
    }
)

SPECS = {
    "relayvsr_x2": RELAY_VSR,
    "relayvsr_x4": RELAY_VSR,
}

from lightx2v.models.runners.request_args import COMMON, VIDEO_OUTPUT
from lightx2v.utils.request_spec import Argument, RequestSpec

RELAY_VSR = RequestSpec(
    {
        "sr": dict(
            COMMON,
            video_path=Argument(input_info_default="", help="Input video path."),
            image_path=Argument(input_info_default="", help="Input image path."),
            sr_ratio=Argument(input_info_default=2.0, type=float, gt=0, help="super resolution ratio for sr task"),
            # An explicit HEIGHT WIDTH resizes each axis of the native result independently;
            # sr_ratio is then ignored (relayvsr_x2 keeps its exact 2x output and rejects it).
            size=VIDEO_OUTPUT["size"],
        )
    },
)

SPECS = {
    "relayvsr_x2": RELAY_VSR,
    "relayvsr_x4": RELAY_VSR,
}

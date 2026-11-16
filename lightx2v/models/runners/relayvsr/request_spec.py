from lightx2v.models.runners.request_args import COMMON, VIDEO_OUTPUT
from lightx2v.utils.request_spec import Argument, RequestSpec

RELAY_VSR = RequestSpec(
    {
        "sr": dict(
            COMMON,
            video_path=Argument(input_info_default="", help="Input video path."),
            image_path=Argument(input_info_default="", help="Input image path."),
            sr_ratio=Argument(input_info_default=2.0, type=float, gt=0, help="super resolution ratio for sr task"),
        )
    },
    context={"size": VIDEO_OUTPUT["size"]},
)

SPECS = {
    "relayvsr_x2": RELAY_VSR,
    "relayvsr_x4": RELAY_VSR,
}

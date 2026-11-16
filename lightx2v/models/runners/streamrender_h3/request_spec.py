from lightx2v.models.runners.request_args import COMMON, VIDEO_OUTPUT
from lightx2v.utils.request_spec import Argument, MultiTaskSpec

SPECS = {
    "streamrender_h3": MultiTaskSpec(
        {
            "refv2v": dict(
                COMMON | VIDEO_OUTPUT,
                prompt=Argument(input_info_default="", help="Text prompt."),
                image_path=Argument(input_info_default="", help="Input image path."),
                video_path=Argument(input_info_default="", help="Input video path."),
            )
        }
    ),
}

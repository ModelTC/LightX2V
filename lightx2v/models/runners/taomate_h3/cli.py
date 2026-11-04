from lightx2v.models.runners.request_args import COMMON, VIDEO_OUTPUT
from lightx2v.utils.request_spec import Argument, MultiTaskSpec

SPECS = {
    "taomate_h3": MultiTaskSpec(
        {
            "t2av": dict(
                COMMON,
                **VIDEO_OUTPUT,
                prompt=Argument(help="Text prompt."),
            )
        }
    ),
}

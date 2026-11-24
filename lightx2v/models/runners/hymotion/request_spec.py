from lightx2v.models.runners.request_args import COMMON
from lightx2v.utils.request_spec import Argument, RequestSpec

SPECS = {
    "hymotion": RequestSpec(
        {
            "t2m": dict(
                COMMON,
                prompt=Argument(input_info_default="", help="Motion description."),
                num_frames=Argument(type=int, input_info_default=120, ge=1, lt=5000, help="Requested frames; native output is clamped to 20..360."),
            )
        }
    )
}

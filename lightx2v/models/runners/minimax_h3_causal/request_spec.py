from lightx2v.models.runners.request_args import COMMON, VIDEO, VIDEO_OUTPUT
from lightx2v.utils.request_spec import Argument, RequestSpec

REQUEST = dict(
    COMMON | VIDEO_OUTPUT,
    image_path=Argument(input_info_default=[], type=list[str], help="Reference image paths."),
    audio_path=Argument(input_info_default="", help="Driving audio for refa2v, or voice reference audio for refa2av."),
    prompt=Argument(input_info_default="", help="Global text prompt."),
    motions=Argument(
        type=list[dict],
        help="Timed motion cues as JSON [{frame: 0, text: ...}, ...]; zero-based frames, strictly increasing from 0. Omit to use the global prompt.",
    ),
)

SPECS = {
    "minimax_h3_causal": RequestSpec({task: REQUEST for task in ("refa2v", "refa2av")}, context=VIDEO),
}

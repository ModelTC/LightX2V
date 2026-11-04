import json

from lightx2v.models.runners.request_args import COMMON, VIDEO_OUTPUT
from lightx2v.utils.request_spec import Argument, MultiTaskSpec

SPECS = {
    "minimax_h3_causal": MultiTaskSpec(
        {
            "refa2v": dict(
                COMMON,
                **VIDEO_OUTPUT,
                image_path=Argument(help="Input image path."),
                audio_path=Argument(help="Driving audio path."),
                prompt=Argument(help="Text prompt."),
                action_prompts=Argument(type=json.loads, help="MiniMax-H3 causal timed actions as a JSON object or list"),
            )
        }
    ),
}

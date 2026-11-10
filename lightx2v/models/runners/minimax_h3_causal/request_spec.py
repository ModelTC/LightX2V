from lightx2v.models.runners.request_args import COMMON, VIDEO_OUTPUT
from lightx2v.utils.request_spec import Argument, MultiTaskSpec, TaskSpec

SPECS = {
    "minimax_h3_causal": MultiTaskSpec(
        {
            "refa2v": TaskSpec(
                dict(
                    COMMON | VIDEO_OUTPUT,
                    image_path=Argument(type=list[str], help="Reference image paths."),
                    audio_path=Argument(help="Driving audio path."),
                    prompt=Argument(help="Text prompt."),
                    action_prompts=Argument(type=dict | list, help="MiniMax-H3 causal timed actions as a JSON object or list"),
                ),
                outputs=("video",),
            )
        }
    ),
}

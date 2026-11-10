from lightx2v.models.runners.request_args import COMMON, PROMPT, VIDEO
from lightx2v.utils.request_spec import Argument, MultiTaskSpec, TaskSpec

SPECS = {
    "lingbot_video": MultiTaskSpec(
        {
            "t2i": TaskSpec(
                dict(
                    COMMON | PROMPT,
                    size=Argument(type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
                ),
                outputs=("image",),
            ),
            "t2v": TaskSpec(VIDEO, outputs=("video",)),
            "i2v": TaskSpec(
                dict(
                    VIDEO,
                    image_path=Argument(help="Input image path."),
                ),
                outputs=("video",),
            ),
        },
    ),
}

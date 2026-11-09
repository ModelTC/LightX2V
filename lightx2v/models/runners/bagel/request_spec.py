from lightx2v.models.runners.request_args import COMMON
from lightx2v.utils.request_spec import Argument, MultiTaskSpec, TaskSpec

SPECS = {
    "bagel": MultiTaskSpec(
        {
            "t2i": TaskSpec(
                dict(
                    COMMON,
                    aspect_ratio=Argument(help="Output aspect ratio."),
                    prompt=Argument(help="Text prompt."),
                    size=Argument(type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
                ),
                outputs=("image",),
            ),
            "i2i": TaskSpec(
                dict(
                    COMMON,
                    image_path=Argument(help="Input image path."),
                    prompt=Argument(help="Text prompt."),
                    size=Argument(type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
                ),
                outputs=("image",),
            ),
        },
    ),
}

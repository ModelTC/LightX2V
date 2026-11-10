from lightx2v.models.runners.request_args import COMMON, IMAGE, PROMPT
from lightx2v.utils.request_spec import Argument, RequestSpec, TaskSpec

SPECS = {
    "longcat_image": RequestSpec(
        {
            "t2i": TaskSpec(IMAGE, outputs=("image",)),
            "i2i": TaskSpec(
                dict(
                    COMMON | PROMPT,
                    image_path=Argument(help="Input image path."),
                ),
                outputs=("image",),
            ),
        },
    ),
}

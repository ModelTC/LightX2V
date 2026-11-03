from lightx2v.models.runners.request_args import COMMON, IMAGE, PROMPT
from lightx2v.utils.request_spec import Argument, RequestSpec

SPECS = {
    "longcat_image": RequestSpec(
        {
            "t2i": IMAGE,
            "i2i": COMMON | PROMPT | {"image_path": Argument(help="Input image path.")},
        },
    ),
}

from lightx2v.models.runners.request_args import COMMON, PROMPT, VIDEO
from lightx2v.utils.request_spec import Argument, RequestSpec

SPECS = {
    "lingbot_video": RequestSpec(
        {
            "t2i": dict(
                COMMON,
                **PROMPT,
                size=Argument(type=int, nargs="+", help="Output size in pixels: HEIGHT WIDTH"),
            ),
            "t2v": VIDEO,
            "i2v": dict(
                VIDEO,
                image_path=Argument(help="Input image path."),
            ),
        },
    ),
}

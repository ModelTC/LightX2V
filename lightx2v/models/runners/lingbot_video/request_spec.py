from lightx2v.models.runners.request_args import COMMON, IMAGE, PROMPT, VIDEO
from lightx2v.utils.request_spec import Argument, MultiTaskSpec

SPECS = {
    "lingbot_video": MultiTaskSpec(
        {
            "t2i": dict(
                COMMON | PROMPT,
                size=Argument(input_info_default=[], type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
            ),
            "t2v": VIDEO,
            "i2v": dict(
                VIDEO,
                image_path=Argument(input_info_default="", help="Input image path."),
            ),
        },
        context=IMAGE | VIDEO,
    ),
}

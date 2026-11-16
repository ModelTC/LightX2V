from lightx2v.models.runners.request_args import COMMON, PROMPT, VIDEO
from lightx2v.utils.request_spec import Argument, RequestSpec

HUNYUAN_VIDEO15 = RequestSpec(
    {
        "t2v": dict(
            COMMON | PROMPT,
            num_frames=Argument(type=int, help="Requested output frame count. Model-specific length constraints apply."),
        ),
        "i2v": dict(
            COMMON | PROMPT,
            image_path=Argument(input_info_default="", help="Input image path."),
            num_frames=Argument(type=int, help="Requested output frame count. Model-specific length constraints apply."),
        ),
    },
    context=VIDEO,
)

SPECS = {
    "hunyuan_video_1.5": HUNYUAN_VIDEO15,
}

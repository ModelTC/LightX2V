from lightx2v.models.runners.request_args import COMMON, PROMPT, VIDEO, fields
from lightx2v.utils.request_spec import RequestSpec

SPECS = {
    "lingbot_video": RequestSpec(
        {
            "t2i": COMMON | PROMPT | fields("size"),
            "t2v": VIDEO,
            "i2v": VIDEO | fields("image_path"),
        },
    ),
}

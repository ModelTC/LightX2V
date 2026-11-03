from lightx2v.models.runners.request_args import COMMON, fields
from lightx2v.utils.request_spec import RequestSpec

SPECS = {
    "bagel": RequestSpec(
        {
            "t2i": COMMON | fields("aspect_ratio", "prompt", "size"),
            "i2i": COMMON | fields("image_path", "prompt", "size"),
        },
    ),
}

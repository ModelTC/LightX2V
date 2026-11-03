from lightx2v.models.runners.request_args import COMMON, IMAGE, PROMPT, fields
from lightx2v.utils.request_spec import RequestSpec

SPECS = {
    "longcat_image": RequestSpec(
        {
            "t2i": IMAGE,
            "i2i": COMMON | PROMPT | fields("image_path"),
        },
    ),
}

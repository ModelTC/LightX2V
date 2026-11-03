from lightx2v.models.runners.request_args import COMMON, fields
from lightx2v.utils.request_spec import RequestSpec

SPECS = {
    "hidream_o1_image": RequestSpec(
        {
            "t2i": COMMON | fields("prompt", "size"),
            "i2i": COMMON | fields("i2i_denoise_strength", "image_path", "keep_aspect_ratio", "layout_bboxes", "prompt", "size"),
        },
    ),
}

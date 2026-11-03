from lightx2v.models.runners.request_args import IMAGE, fields
from lightx2v.utils.request_spec import RequestSpec

SPECS = {
    "z_image": RequestSpec(
        {
            "t2i": IMAGE,
            "i2i": IMAGE | fields("i2i_denoise_strength", "image_path"),
        },
    ),
}

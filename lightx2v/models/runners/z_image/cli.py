from lightx2v.models.runners.request_args import IMAGE
from lightx2v.utils.request_spec import Argument, RequestSpec

SPECS = {
    "z_image": RequestSpec(
        {
            "t2i": IMAGE,
            "i2i": IMAGE
            | {
                "i2i_denoise_strength": Argument(type=float, help="Image edit strength in [0, 1]; 0 preserves the input and 1 redraws it."),
                "image_path": Argument(help="Input image path."),
            },
        },
    ),
}

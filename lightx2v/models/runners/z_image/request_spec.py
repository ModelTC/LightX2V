from lightx2v.models.runners.request_args import IMAGE
from lightx2v.utils.request_spec import Argument, MultiTaskSpec, TaskSpec

SPECS = {
    "z_image": MultiTaskSpec(
        {
            "t2i": TaskSpec(IMAGE, outputs=("image",)),
            "i2i": TaskSpec(
                dict(
                    IMAGE,
                    i2i_denoise_strength=Argument(type=float, help="Image edit strength in [0, 1]; 0 preserves the input and 1 redraws it."),
                    image_path=Argument(type=list[str], help="Input image paths."),
                ),
                outputs=("image",),
            ),
        },
    ),
}

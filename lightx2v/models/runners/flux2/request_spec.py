from lightx2v.models.runners.request_args import COMMON, IMAGE
from lightx2v.utils.request_spec import Argument, MultiTaskSpec

INPAINT = {
    "inpaint_blur_sigma": Argument(type=float, help="Flux2 inpainting mask blur sigma."),
    "inpaint_blur_size": Argument(type=int, help="Flux2 inpainting mask blur kernel size."),
}


class Flux2Spec(MultiTaskSpec):
    def get_arguments(self, config, task):
        arguments = super().get_arguments(config, task)
        if task == "i2i" and config.get("inpaint_mask_enabled", False):
            arguments.update(INPAINT)
        return arguments


SPECS = {
    "flux2": Flux2Spec(
        {
            "t2i": dict(
                COMMON,
                aspect_ratio=Argument(input_info_default="", help="Output aspect ratio."),
                prompt=Argument(input_info_default="", help="Text prompt."),
                size=Argument(input_info_default=[], type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
            ),
            "i2i": dict(
                COMMON,
                image_path=Argument(input_info_default=[], type=list[str], help="Input image paths or a directory containing reference images."),
                prompt=Argument(input_info_default="", help="Text prompt."),
            ),
        },
        context=IMAGE | INPAINT,
    ),
}

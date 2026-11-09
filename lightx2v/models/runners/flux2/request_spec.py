from lightx2v.models.runners.request_args import COMMON
from lightx2v.utils.request_spec import Argument, MultiTaskSpec, TaskSpec


class Flux2Spec(MultiTaskSpec):
    def get_arguments(self, config, task):
        arguments = super().get_arguments(config, task)
        if task == "i2i" and config.get("inpaint_mask_enabled", False):
            arguments.update(
                {"inpaint_blur_sigma": Argument(type=float, help="Flux2 inpainting mask blur sigma."), "inpaint_blur_size": Argument(type=int, help="Flux2 inpainting mask blur kernel size.")}
            )
        return arguments


SPECS = {
    "flux2": Flux2Spec(
        {
            "t2i": TaskSpec(
                dict(
                    COMMON,
                    aspect_ratio=Argument(help="Output aspect ratio."),
                    prompt=Argument(help="Text prompt."),
                    size=Argument(type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
                ),
                outputs=("image",),
            ),
            "i2i": TaskSpec(
                dict(
                    COMMON,
                    image_path=Argument(help="Input image path."),
                    prompt=Argument(help="Text prompt."),
                ),
                outputs=("image",),
            ),
        },
    ),
}

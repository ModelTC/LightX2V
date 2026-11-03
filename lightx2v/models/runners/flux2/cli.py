from lightx2v.models.runners.request_args import COMMON, fields
from lightx2v.utils.request_spec import RequestSpec


class Flux2Spec(RequestSpec):
    def get_arguments(self, config, task):
        arguments = super().get_arguments(config, task)
        if task == "i2i" and config.get("inpaint_mask_enabled", False):
            arguments.update(fields("inpaint_blur_sigma", "inpaint_blur_size"))
        return arguments


SPECS = {
    "flux2": Flux2Spec(
        {
            "t2i": COMMON | fields("aspect_ratio", "prompt", "size"),
            "i2i": COMMON | fields("image_path", "prompt"),
        },
    ),
}

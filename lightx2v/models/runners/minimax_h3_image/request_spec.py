from typing import Annotated

from pydantic import BeforeValidator

from lightx2v.models.runners.request_args import IMAGE
from lightx2v.utils.image_output import validate_output_compression
from lightx2v.utils.request_spec import Argument, RequestSpec


class MiniMaxH3ImageSpec(RequestSpec):
    def get_tasks(self, config):
        return super().get_tasks(config) if config.get("task") is not None else tuple(self.tasks)


IMAGE = dict(
    IMAGE, output_compression=Argument(type=Annotated[int, BeforeValidator(validate_output_compression)], input_info_default=95, help="JPEG/WebP quality, integer 0-100 (default 95); ignored for PNG")
)

SPECS = {
    "minimax_h3_image": MiniMaxH3ImageSpec(
        {
            "t2i": IMAGE,
            "i2i": dict(
                IMAGE,
                image_path=Argument(input_info_default=[], type=list[str], help="Reference image paths in prompt order, up to 16 images."),
            ),
        }
    ),
}

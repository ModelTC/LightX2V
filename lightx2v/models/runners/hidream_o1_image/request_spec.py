import argparse

from lightx2v.models.runners.request_args import COMMON
from lightx2v.utils.request_spec import Argument, RequestSpec

SPECS = {
    "hidream_o1_image": RequestSpec(
        {
            "t2i": dict(
                COMMON,
                prompt=Argument(help="Text prompt."),
                size=Argument(type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
            ),
            "i2i": dict(
                COMMON,
                i2i_denoise_strength=Argument(type=float, help="Image edit strength in [0, 1]; 0 preserves the input and 1 redraws it."),
                image_path=Argument(help="Input image path."),
                keep_aspect_ratio=Argument(type=bool, action=argparse.BooleanOptionalAction, help="(i2i) When exactly one reference image is provided, preserve its aspect ratio with max_size=2048."),
                layout_bboxes=Argument(help="(i2i) Layout boxes as a JSON string or JSON file path for HiDream layout-conditioned editing."),
                prompt=Argument(help="Text prompt."),
                size=Argument(type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
            ),
        },
    ),
}

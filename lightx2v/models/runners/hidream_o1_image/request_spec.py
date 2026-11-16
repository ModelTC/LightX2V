from lightx2v.models.runners.request_args import COMMON, IMAGE
from lightx2v.utils.request_spec import Argument, MultiTaskSpec

SPECS = {
    "hidream_o1_image": MultiTaskSpec(
        {
            "t2i": dict(
                COMMON,
                prompt=Argument(input_info_default="", help="Text prompt."),
                size=Argument(input_info_default=[], type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
            ),
            "i2i": dict(
                COMMON,
                i2i_denoise_strength=Argument(type=float, help="Image edit strength in [0, 1]; 0 preserves the input and 1 redraws it."),
                image_path=Argument(input_info_default=[], type=list[str], help="Input image paths."),
                keep_aspect_ratio=Argument(input_info_default=False, type=bool, help="(i2i) When exactly one reference image is provided, preserve its aspect ratio with max_size=2048."),
                layout_bboxes=Argument(type=str | list | dict, help="(i2i) Layout boxes as JSON or a JSON file path for HiDream layout-conditioned editing."),
                prompt=Argument(input_info_default="", help="Text prompt."),
                size=Argument(input_info_default=[], type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
            ),
        },
        context=IMAGE,
    ),
}

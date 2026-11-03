from lightx2v.models.runners.request_args import COMMON, PROMPT
from lightx2v.utils.request_spec import Argument, RequestSpec


class HunyuanVideoSpec(RequestSpec):
    def get_arguments(self, config, task):
        arguments = super().get_arguments(config, task)
        sr = config.get("video_super_resolution", {})
        if sr.get("sr_version", None) is not None and sr.get("enable_cfg", config.get("enable_cfg", False)):
            arguments.update({"negative_prompt": Argument(help="Negative text prompt.")})
        return arguments


HUNYUAN_VIDEO15 = HunyuanVideoSpec(
    {
        "t2v": COMMON | PROMPT | {"num_frames": Argument(type=int, help="Requested output frame count. Model-specific length constraints apply.")},
        "i2v": COMMON
        | PROMPT
        | {
            "image_path": Argument(help="Input image path."),
            "num_frames": Argument(type=int, help="Requested output frame count. Model-specific length constraints apply."),
        },
    }
)

SPECS = {
    "hunyuan_video_1.5": HUNYUAN_VIDEO15,
}

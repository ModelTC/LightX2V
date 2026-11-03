from lightx2v.models.runners.request_args import COMMON, PROMPT, fields
from lightx2v.utils.request_spec import RequestSpec


class HunyuanVideoSpec(RequestSpec):
    def get_arguments(self, config, task):
        arguments = super().get_arguments(config, task)
        sr = config.get("video_super_resolution", {})
        if sr.get("sr_version", None) is not None and sr.get("enable_cfg", config.get("enable_cfg", False)):
            arguments.update(fields("negative_prompt"))
        return arguments


HUNYUAN_VIDEO15 = HunyuanVideoSpec({"t2v": COMMON | PROMPT | fields("num_frames"), "i2v": COMMON | PROMPT | fields("image_path", "num_frames")})

SPECS = {
    "hunyuan_video_1.5": HUNYUAN_VIDEO15,
}

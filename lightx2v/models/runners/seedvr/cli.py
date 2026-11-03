from lightx2v.models.runners.request_args import COMMON, fields
from lightx2v.utils.request_spec import RequestSpec


class SeedVRSpec(RequestSpec):
    def get_arguments(self, config, task):
        arguments = super().get_arguments(config, task)
        if config.get("seq_parallel", False) or (config.get("parallel", False) or {}).get("seq_p_size", 1) > 1:
            arguments.pop("image_path", None)
        return arguments


SPECS = {
    "seedvr2": SeedVRSpec({"sr": COMMON | fields("image_path", "match_target_size", "size", "sr_ratio", "video_path")}),
}

from lightx2v.models.runners.request_args import COMMON, RESAMPLING, VIDEO_OUTPUT, fields
from lightx2v.utils.request_spec import RequestSpec


class MiniMaxH3Spec(RequestSpec):
    def get_tasks(self, config):
        if config.get("text_encoder_disk_streaming", False):
            return ("t2av",)
        if config["model_variant"] == "ref2av":
            return ("ref2av",)
        return tuple(self.tasks)


SPECS = {
    "minimax_h3": MiniMaxH3Spec(
        {
            "t2av": COMMON | VIDEO_OUTPUT | RESAMPLING | fields("prompt"),
            "i2av": COMMON | VIDEO_OUTPUT | RESAMPLING | fields("image_path", "prompt"),
            "l2av": COMMON | VIDEO_OUTPUT | RESAMPLING | fields("last_frame_path", "prompt"),
            "fl2av": COMMON | VIDEO_OUTPUT | RESAMPLING | fields("image_path", "last_frame_path", "prompt"),
            "ref2av": COMMON | VIDEO_OUTPUT | RESAMPLING | fields("audio_path", "image_path", "prompt", "video_path"),
        },
    ),
}

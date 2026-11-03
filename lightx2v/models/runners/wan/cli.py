from lightx2v.models.runners.request_args import COMMON, PROMPT, VIDEO, fields, without
from lightx2v.utils.request_spec import RequestSpec


class WanSpec(RequestSpec):
    FIXED_FRAME_ATTENTION_TYPES = {"svg_attn", "radial_attn", "nbhd_attn", "nbhd_attn_flashinfer", "general_sparse_attn"}

    def get_arguments(self, config, task):
        arguments = super().get_arguments(config, task)
        if config.get("self_attn_1_type", None) in self.FIXED_FRAME_ATTENTION_TYPES:
            arguments.pop("num_frames", None)
        return arguments


class WanAnimateSpec(WanSpec):
    def get_arguments(self, config, task):
        arguments = super().get_arguments(config, task)
        if config.get("replace_flag", False):
            arguments.update(fields("mask_path", "background_video_path"))
        return arguments


class WanDancerSpec(WanSpec):
    def get_arguments(self, config, task):
        arguments = super().get_arguments(config, task)
        if config.get("dancer_stage", None) == "local":
            arguments.update(fields("video_path"))
        return arguments


class WanAudioARSpec(WanSpec):
    def get_arguments(self, config, task):
        arguments = super().get_arguments(config, task)
        if config.get("prompt_travel", None):
            arguments.pop("prompt", None)
        return arguments


EDEN_LIVE = WanSpec({"v2v": VIDEO | fields("image_path", "video_path")})

FAST_WAM = RequestSpec({"i2va": COMMON | fields("image_path", "prompt", "save_action_path", "state_path")})

WAN_AUDIO = WanSpec({task: COMMON | PROMPT | fields("audio_path", "image_path", "num_frames", "video_duration", "resize_mode", "fixed_area", "size") for task in ("s2v", "rs2v")})

WAN = WanSpec({"t2v": VIDEO, "i2v": VIDEO | fields("image_path"), "flf2v": VIDEO | fields("image_path", "last_frame_path")})

LINGBOT = WanSpec({"i2v": WAN.tasks["i2v"] | fields("action_path", "pose")})

WAN_SF = WanSpec({"t2v": without(WAN.tasks["t2v"], "num_frames")})

WAN_VACE = WanSpec({"vace": VIDEO | fields("mask_path", "ref_image_paths", "video_path")})

SPECS = {
    "dreamzero": WanSpec({"i2va": COMMON | PROMPT | fields("image_path", "save_action_path", "state_path")}),
    "fastwam": FAST_WAM,
    "infinitetalk": WanSpec({"s2v": COMMON | PROMPT | fields("audio_path", "image_path", "num_frames", "video_duration", "video_path")}),
    "lingbot_va": WanSpec({"i2va": COMMON | PROMPT | fields("image_path")}),
    "lingbot_world": LINGBOT,
    "lingbot_world_fast": WanSpec({"i2v": without(LINGBOT.tasks["i2v"], "num_frames")}),
    "matrix_game4": WanSpec({"v2v": VIDEO | fields("video_path", "image_path")}),
    "realtimewam": FAST_WAM,
    "seko_talk": WAN_AUDIO,
    "seko_talk_ar": WanAudioARSpec({"rs2v": without(WAN_AUDIO.tasks["rs2v"], "num_frames", "resize_mode", "fixed_area")}),
    "wan2.1": WAN,
    "wan2.1_sf": WAN_SF,
    "wan2.1_realtime": WAN_SF,
    "wan2.1_vace": WAN_VACE,
    "wan2.2": WanSpec({task: WAN.tasks[task] for task in ("t2v", "i2v")}),
    "wan2.2_animate": WanAnimateSpec({"animate": COMMON | PROMPT | fields("face_video_path", "pose_video_path", "ref_image_paths", "num_frames")}),
    "wan2.2_animate2_distilled": WanSpec({"animate": VIDEO | fields("image_path", "ref_video_prompt", "pose_video_path", "ref_image_paths", "video_path")}),
    "wan2.2_moe": WAN,
    "wan2.2_moe_vace": WAN_VACE,
    "wan2.2_s2v": WanSpec({"s2v": COMMON | PROMPT | fields("audio_path", "image_path", "pose_video_path")}),
    "wan21_edenlive": EDEN_LIVE,
    "wan21_edenlive_ar": EDEN_LIVE,
    "wan22_edenlive": EDEN_LIVE,
    "wan22_edenlive_ar": EDEN_LIVE,
    "wan_dancer": WanDancerSpec({"s2v": COMMON | PROMPT | fields("audio_path", "image_path")}),
}

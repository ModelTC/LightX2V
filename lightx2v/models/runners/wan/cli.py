from lightx2v.models.runners.request_args import COMMON, PROMPT, VIDEO, without
from lightx2v.utils.request_spec import Argument, RequestSpec


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
            arguments.update({"mask_path": Argument(help="Input mask path."), "background_video_path": Argument(help="Background video path.")})
        return arguments


class WanDancerSpec(WanSpec):
    def get_arguments(self, config, task):
        arguments = super().get_arguments(config, task)
        if config.get("dancer_stage", None) == "local":
            arguments.update({"video_path": Argument(help="Input video path.")})
        return arguments


class WanAudioARSpec(WanSpec):
    def get_arguments(self, config, task):
        arguments = super().get_arguments(config, task)
        if config.get("prompt_travel", None):
            arguments.pop("prompt", None)
        return arguments


EDEN_LIVE = WanSpec(
    {
        "v2v": VIDEO
        | {
            "image_path": Argument(help="Input image path."),
            "video_path": Argument(help="Input video path."),
        }
    }
)

FAST_WAM = RequestSpec(
    {
        "i2va": COMMON
        | {
            "image_path": Argument(help="Input image path."),
            "prompt": Argument(help="Text prompt."),
            "save_action_path": Argument(help="Output action file path."),
            "state_path": Argument(help="Input robot state file."),
        }
    }
)

WAN_AUDIO = WanSpec(
    {
        task: COMMON
        | PROMPT
        | {
            "audio_path": Argument(help="Driving audio path."),
            "image_path": Argument(help="Input image path."),
            "num_frames": Argument(type=int, help="Requested output frame count. Model-specific length constraints apply."),
            "video_duration": Argument(type=float, help="Requested output duration in seconds for audio-driven video generation."),
            "resize_mode": Argument(help="seko_talk s2v/rs2v image and mask resize mode; defaults to the startup config."),
            "fixed_area": Argument(help="seko_talk area/min-side resize tier: 480p, 720p, or 1080p; defaults to the startup config."),
            "size": Argument(type=int, nargs="+", help="Output size in pixels: HEIGHT WIDTH"),
        }
        for task in ("s2v", "rs2v")
    }
)

WAN = WanSpec(
    {
        "t2v": VIDEO,
        "i2v": VIDEO | {"image_path": Argument(help="Input image path.")},
        "flf2v": VIDEO
        | {
            "image_path": Argument(help="Input image path."),
            "last_frame_path": Argument(help="Last-frame image path."),
        },
    }
)

LINGBOT = WanSpec(
    {
        "i2v": WAN.tasks["i2v"]
        | {
            "action_path": Argument(help="Directory path for lingbot camera/action control files (poses.npy, intrinsics.npy, optional action.npy)."),
            "pose": Argument(help="Camera pose specification or JSON file path."),
        }
    }
)

WAN_SF = WanSpec({"t2v": without(WAN.tasks["t2v"], "num_frames")})

WAN_VACE = WanSpec(
    {
        "vace": VIDEO
        | {
            "mask_path": Argument(help="Input mask path."),
            "ref_image_paths": Argument(help="Reference image paths, separated by commas."),
            "video_path": Argument(help="Input video path."),
        }
    }
)

SPECS = {
    "dreamzero": WanSpec(
        {
            "i2va": COMMON
            | PROMPT
            | {
                "image_path": Argument(help="Input image path."),
                "save_action_path": Argument(help="Output action file path."),
                "state_path": Argument(help="Input robot state file."),
            }
        }
    ),
    "fastwam": FAST_WAM,
    "infinitetalk": WanSpec(
        {
            "s2v": COMMON
            | PROMPT
            | {
                "audio_path": Argument(help="Driving audio path."),
                "image_path": Argument(help="Input image path."),
                "num_frames": Argument(type=int, help="Requested output frame count. Model-specific length constraints apply."),
                "video_duration": Argument(type=float, help="Requested output duration in seconds for audio-driven video generation."),
                "video_path": Argument(help="Input video path."),
            }
        }
    ),
    "lingbot_va": WanSpec({"i2va": COMMON | PROMPT | {"image_path": Argument(help="Input image path.")}}),
    "lingbot_world": LINGBOT,
    "lingbot_world_fast": WanSpec({"i2v": without(LINGBOT.tasks["i2v"], "num_frames")}),
    "matrix_game4": WanSpec(
        {
            "v2v": VIDEO
            | {
                "video_path": Argument(help="Input video path."),
                "image_path": Argument(help="Input image path."),
            }
        }
    ),
    "realtimewam": FAST_WAM,
    "seko_talk": WAN_AUDIO,
    "seko_talk_ar": WanAudioARSpec({"rs2v": without(WAN_AUDIO.tasks["rs2v"], "num_frames", "resize_mode", "fixed_area")}),
    "wan2.1": WAN,
    "wan2.1_sf": WAN_SF,
    "wan2.1_realtime": WAN_SF,
    "wan2.1_vace": WAN_VACE,
    "wan2.2": WanSpec({task: WAN.tasks[task] for task in ("t2v", "i2v")}),
    "wan2.2_animate": WanAnimateSpec(
        {
            "animate": COMMON
            | PROMPT
            | {
                "face_video_path": Argument(help="Face input path."),
                "pose_video_path": Argument(help="Pose driving video path."),
                "ref_image_paths": Argument(help="Reference image paths, separated by commas."),
                "num_frames": Argument(type=int, help="Requested output frame count. Model-specific length constraints apply."),
            }
        }
    ),
    "wan2.2_animate2_distilled": WanSpec(
        {
            "animate": VIDEO
            | {
                "image_path": Argument(help="Input image path."),
                "ref_video_prompt": Argument(help="Reference/driving-video prompt for Wan-Animate-2."),
                "pose_video_path": Argument(help="Pose driving video path."),
                "ref_image_paths": Argument(help="Reference image paths, separated by commas."),
                "video_path": Argument(help="Input video path."),
            }
        }
    ),
    "wan2.2_moe": WAN,
    "wan2.2_moe_vace": WAN_VACE,
    "wan2.2_s2v": WanSpec(
        {
            "s2v": COMMON
            | PROMPT
            | {
                "audio_path": Argument(help="Driving audio path."),
                "image_path": Argument(help="Input image path."),
                "pose_video_path": Argument(help="Pose driving video path."),
            }
        }
    ),
    "wan21_edenlive": EDEN_LIVE,
    "wan21_edenlive_ar": EDEN_LIVE,
    "wan22_edenlive": EDEN_LIVE,
    "wan22_edenlive_ar": EDEN_LIVE,
    "wan_dancer": WanDancerSpec(
        {
            "s2v": COMMON
            | PROMPT
            | {
                "audio_path": Argument(help="Driving audio path."),
                "image_path": Argument(help="Input image path."),
            }
        }
    ),
}

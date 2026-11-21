from lightx2v.models.runners.request_args import COMMON, PROMPT, VIDEO, VIDEO_OUTPUT, without
from lightx2v.utils.request_spec import Argument, MultiTaskSpec, RequestSpec

REPLACEMENT = {
    "mask_path": Argument(input_info_default="", help="Input mask path."),
    "background_video_path": Argument(input_info_default="", help="Background video path."),
}


class WanAnimateSpec(RequestSpec):
    def get_arguments(self, config, task):
        arguments = super().get_arguments(config, task)
        if config.get("replace_flag", False):
            arguments.update(REPLACEMENT)
        return arguments


class WanDancerSpec(RequestSpec):
    def get_arguments(self, config, task):
        arguments = super().get_arguments(config, task)
        if config.get("dancer_stage", None) == "local":
            arguments["video_path"] = self.context["video_path"]
        return arguments


POLICY_PATHS = {
    "save_action_path": Argument(input_info_default="", help="Output action file path."),
    "state_path": Argument(input_info_default="", help="Input robot state file."),
}
POLICY_CONTEXT = dict(
    VIDEO | POLICY_PATHS,
    video_path=Argument(input_info_default=""),
    action_path=Argument(input_info_default=""),
    action_mode=Argument(input_info_default=""),
    domain_name=Argument(input_info_default=""),
    view_point=Argument(input_info_default=""),
)


EDEN_LIVE = RequestSpec(
    {
        "v2v": dict(
            VIDEO,
            image_path=Argument(input_info_default="", help="Input image path."),
            video_path=Argument(input_info_default="", help="Input video path."),
        )
    }
)

FAST_WAM = RequestSpec(
    {
        "i2va": dict(
            COMMON | POLICY_PATHS,
            image_path=Argument(input_info_default=[], type=list[str], help="Camera image paths, or a directory containing the camera inputs."),
            prompt=Argument(input_info_default="", help="Text prompt."),
        )
    },
    context=POLICY_CONTEXT,
)

WAN_AUDIO = RequestSpec(
    {
        task: dict(
            COMMON | PROMPT,
            audio_path=Argument(input_info_default="", help="Driving audio path."),
            image_path=Argument(input_info_default="", help="Input image path."),
            num_frames=Argument(type=int, help="Requested output frame count. Model-specific length constraints apply."),
            video_duration=Argument(type=float, help="Requested output duration in seconds for audio-driven video generation."),
            resize_mode=Argument(input_info_default="adaptive", help="seko_talk s2v/rs2v image and mask resize mode; defaults to the startup config."),
            fixed_area=Argument(help="seko_talk area/min-side resize tier: 480p, 720p, or 1080p; defaults to the startup config."),
            size=Argument(input_info_default=[], type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
        )
        for task in ("s2v", "rs2v")
    },
    context=dict(VIDEO, video_path=Argument(input_info_default=""), pose_video_path=Argument(input_info_default="")),
)

WAN = RequestSpec(
    {
        "t2v": VIDEO,
        "i2v": dict(
            VIDEO,
            image_path=Argument(input_info_default="", help="Input image path."),
        ),
        "flf2v": dict(
            VIDEO,
            image_path=Argument(input_info_default="", help="Input image path."),
            last_frame_path=Argument(input_info_default="", help="Last-frame image path."),
        ),
    }
)

LINGBOT = RequestSpec(
    {
        "i2v": dict(
            WAN.tasks["i2v"],
            action_path=Argument(input_info_default="", help="Directory path for lingbot camera/action control files (poses.npy, intrinsics.npy, optional action.npy)."),
            pose=Argument(type=str | dict, help="Camera pose specification or JSON file path."),
        )
    }
)

WAN_SF = RequestSpec({"t2v": without(WAN.tasks["t2v"], "num_frames")}, context=VIDEO)

WAN_VACE = RequestSpec(
    {
        "vace": dict(
            VIDEO,
            mask_path=Argument(help="Input mask path."),
            ref_image_paths=Argument(input_info_default=[], type=list[str], help="Reference image paths."),
            video_path=Argument(help="Input video path."),
        )
    }
)

SPECS = {
    "dreamzero": RequestSpec(
        {
            "i2va": dict(
                COMMON | PROMPT | POLICY_PATHS,
                image_path=Argument(input_info_default=[], type=list[str], help="Camera image paths, or a directory containing the camera inputs."),
            )
        },
        context=POLICY_CONTEXT,
    ),
    "fastwam": FAST_WAM,
    "infinitetalk": RequestSpec(
        {
            "s2v": dict(
                COMMON | PROMPT,
                audio_path=Argument(input_info_default=[], type=list[str], help="Driving audio paths, or a directory containing audio and speaker metadata."),
                image_path=Argument(input_info_default="", help="Input image path."),
                num_frames=Argument(type=int, help="Requested output frame count. Model-specific length constraints apply."),
                video_duration=Argument(type=float, help="Requested output duration in seconds for audio-driven video generation."),
                video_path=Argument(input_info_default="", help="Input video path."),
            )
        },
        context=VIDEO,
    ),
    "lingbot_va": RequestSpec(
        {
            "i2va": dict(
                COMMON | PROMPT,
                image_path=Argument(input_info_default=[], type=list[str], help="Camera image paths, or a directory containing the camera inputs."),
            )
        },
        context=POLICY_CONTEXT,
    ),
    "lingbot_world": LINGBOT,
    "lingbot_world_fast": RequestSpec({"i2v": without(LINGBOT.tasks["i2v"], "num_frames")}, context=VIDEO),
    "matrix_game4": RequestSpec(
        {
            "v2v": dict(
                VIDEO,
                video_path=Argument(input_info_default="", help="Input video path."),
                image_path=Argument(input_info_default="", help="Input image path."),
            )
        }
    ),
    "realtimewam": FAST_WAM,
    "seko_talk": WAN_AUDIO,
    "seko_talk_ar": RequestSpec({"rs2v": without(WAN_AUDIO.tasks["rs2v"], "num_frames", "resize_mode", "fixed_area")}, context=WAN_AUDIO.get_input_arguments({}, "rs2v")),
    "tbdub": RequestSpec(
        {
            "v2v": dict(
                COMMON | VIDEO_OUTPUT,
                video_path=Argument(input_info_default="", help="Input cropped face video path."),
                audio_path=Argument(input_info_default="", help="Driving audio path."),
            )
        },
        context=VIDEO,
    ),
    "wan2.1": WAN,
    "wan2.1_sf": WAN_SF,
    "wan2.1_realtime": WAN_SF,
    "wan2.1_vace": WAN_VACE,
    "wan2.2": MultiTaskSpec({task: WAN.tasks[task] for task in ("t2v", "i2v")}),
    "wan2.2_animate": WanAnimateSpec(
        {
            "animate": dict(
                COMMON | PROMPT,
                face_video_path=Argument(input_info_default="", help="Face input path."),
                pose_video_path=Argument(input_info_default="", help="Pose driving video path."),
                ref_image_paths=Argument(input_info_default=[], type=list[str], help="Reference image paths."),
                num_frames=Argument(type=int, help="Requested output frame count. Model-specific length constraints apply."),
            )
        },
        context=VIDEO | REPLACEMENT,
    ),
    "wan2.2_animate2_distilled": RequestSpec(
        {
            "animate": dict(
                VIDEO,
                image_path=Argument(input_info_default="", help="Input image path."),
                ref_video_prompt=Argument(input_info_default="人物动作的参考视频", help="Reference/driving-video prompt for Wan-Animate-2."),
                pose_video_path=Argument(input_info_default="", help="Pose driving video path."),
                ref_image_paths=Argument(input_info_default=[], type=list[str], help="Reference image paths."),
                video_path=Argument(input_info_default="", help="Input video path."),
            )
        },
        context=VIDEO | REPLACEMENT,
    ),
    "wan2.2_moe": WAN,
    "wan2.2_moe_vace": WAN_VACE,
    "wan2.2_s2v": RequestSpec(
        {
            "s2v": dict(
                COMMON | PROMPT,
                audio_path=Argument(input_info_default="", help="Driving audio path."),
                image_path=Argument(input_info_default="", help="Input image path."),
                pose_video_path=Argument(input_info_default="", help="Pose driving video path."),
            )
        },
        context=dict(VIDEO, video_path=Argument(input_info_default=""), video_duration=Argument(type=float)),
    ),
    "wan21_edenlive": EDEN_LIVE,
    "wan21_edenlive_ar": EDEN_LIVE,
    "wan22_edenlive": EDEN_LIVE,
    "wan22_edenlive_ar": EDEN_LIVE,
    "wan_dancer": WanDancerSpec(
        {
            "s2v": dict(
                COMMON | PROMPT,
                audio_path=Argument(input_info_default="", help="Driving audio path."),
                image_path=Argument(input_info_default="", help="Input image path."),
            )
        },
        context=dict(VIDEO, video_path=Argument(input_info_default="", help="Input video path."), pose_video_path=Argument(input_info_default=""), video_duration=Argument(type=float)),
    ),
}

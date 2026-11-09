from lightx2v.models.runners.request_args import COMMON, PROMPT, VIDEO, without
from lightx2v.utils.request_spec import Argument, MultiTaskSpec, RequestSpec, TaskSpec


class WanAnimateSpec(RequestSpec):
    def get_arguments(self, config, task):
        arguments = super().get_arguments(config, task)
        if config.get("replace_flag", False):
            arguments.update({"mask_path": Argument(help="Input mask path."), "background_video_path": Argument(help="Background video path.")})
        return arguments


class WanDancerSpec(RequestSpec):
    def get_arguments(self, config, task):
        arguments = super().get_arguments(config, task)
        if config.get("dancer_stage", None) == "local":
            arguments.update({"video_path": Argument(help="Input video path.")})
        return arguments


EDEN_LIVE = RequestSpec(
    {
        "v2v": TaskSpec(
            dict(
                VIDEO,
                image_path=Argument(help="Input image path."),
                video_path=Argument(help="Input video path."),
            ),
            outputs=("video",),
        )
    }
)

FAST_WAM = RequestSpec(
    {
        "i2va": TaskSpec(
            dict(
                COMMON,
                image_path=Argument(help="Input image path."),
                prompt=Argument(help="Text prompt."),
                save_action_path=Argument(help="Output action file path."),
                state_path=Argument(help="Input robot state file."),
            ),
            outputs=("video",),
        )
    }
)

WAN_AUDIO = RequestSpec(
    {
        task: TaskSpec(
            dict(
                COMMON | PROMPT,
                audio_path=Argument(help="Driving audio path."),
                image_path=Argument(help="Input image path."),
                num_frames=Argument(type=int, help="Requested output frame count. Model-specific length constraints apply."),
                video_duration=Argument(type=float, help="Requested output duration in seconds for audio-driven video generation."),
                resize_mode=Argument(help="seko_talk s2v/rs2v image and mask resize mode; defaults to the startup config."),
                fixed_area=Argument(help="seko_talk area/min-side resize tier: 480p, 720p, or 1080p; defaults to the startup config."),
                size=Argument(type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
            ),
            outputs=("video",),
        )
        for task in ("s2v", "rs2v")
    }
)

WAN = RequestSpec(
    {
        "t2v": TaskSpec(VIDEO, outputs=("video",)),
        "i2v": TaskSpec(
            dict(
                VIDEO,
                image_path=Argument(help="Input image path."),
            ),
            outputs=("video",),
        ),
        "flf2v": TaskSpec(
            dict(
                VIDEO,
                image_path=Argument(help="Input image path."),
                last_frame_path=Argument(help="Last-frame image path."),
            ),
            outputs=("video",),
        ),
    }
)

LINGBOT = RequestSpec(
    {
        "i2v": TaskSpec(
            dict(
                WAN.tasks["i2v"].arguments,
                action_path=Argument(help="Directory path for lingbot camera/action control files (poses.npy, intrinsics.npy, optional action.npy)."),
                pose=Argument(type=str | dict, help="Camera pose specification or JSON file path."),
            ),
            outputs=("video",),
        )
    }
)

WAN_SF = RequestSpec({"t2v": TaskSpec(without(WAN.tasks["t2v"].arguments, "num_frames"), outputs=("video",))})

WAN_VACE = RequestSpec(
    {
        "vace": TaskSpec(
            dict(
                VIDEO,
                mask_path=Argument(help="Input mask path."),
                ref_image_paths=Argument(type=str | list[str], help="Reference image paths, separated by commas."),
                video_path=Argument(help="Input video path."),
            ),
            outputs=("video",),
        )
    }
)

SPECS = {
    "dreamzero": RequestSpec(
        {
            "i2va": TaskSpec(
                dict(
                    COMMON | PROMPT,
                    image_path=Argument(help="Input image path."),
                    save_action_path=Argument(help="Output action file path."),
                    state_path=Argument(help="Input robot state file."),
                ),
                outputs=("video",),
            )
        }
    ),
    "fastwam": FAST_WAM,
    "infinitetalk": RequestSpec(
        {
            "s2v": TaskSpec(
                dict(
                    COMMON | PROMPT,
                    audio_path=Argument(help="Driving audio path."),
                    image_path=Argument(help="Input image path."),
                    num_frames=Argument(type=int, help="Requested output frame count. Model-specific length constraints apply."),
                    video_duration=Argument(type=float, help="Requested output duration in seconds for audio-driven video generation."),
                    video_path=Argument(help="Input video path."),
                ),
                outputs=("video",),
            )
        }
    ),
    "lingbot_va": RequestSpec(
        {
            "i2va": TaskSpec(
                dict(
                    COMMON | PROMPT,
                    image_path=Argument(help="Input image path."),
                ),
                outputs=("video",),
            )
        }
    ),
    "lingbot_world": LINGBOT,
    "lingbot_world_fast": RequestSpec({"i2v": TaskSpec(without(LINGBOT.tasks["i2v"].arguments, "num_frames"), outputs=("video",))}),
    "matrix_game4": RequestSpec(
        {
            "v2v": TaskSpec(
                dict(
                    VIDEO,
                    video_path=Argument(help="Input video path."),
                    image_path=Argument(help="Input image path."),
                ),
                outputs=("video",),
            )
        }
    ),
    "realtimewam": FAST_WAM,
    "seko_talk": WAN_AUDIO,
    "seko_talk_ar": RequestSpec({"rs2v": TaskSpec(without(WAN_AUDIO.tasks["rs2v"].arguments, "num_frames", "resize_mode", "fixed_area"), outputs=("video",))}),
    "wan2.1": WAN,
    "wan2.1_sf": WAN_SF,
    "wan2.1_realtime": WAN_SF,
    "wan2.1_vace": WAN_VACE,
    "wan2.2": MultiTaskSpec({task: WAN.tasks[task] for task in ("t2v", "i2v")}),
    "wan2.2_animate": WanAnimateSpec(
        {
            "animate": TaskSpec(
                dict(
                    COMMON | PROMPT,
                    face_video_path=Argument(help="Face input path."),
                    pose_video_path=Argument(help="Pose driving video path."),
                    ref_image_paths=Argument(type=str | list[str], help="Reference image paths, separated by commas."),
                    num_frames=Argument(type=int, help="Requested output frame count. Model-specific length constraints apply."),
                ),
                outputs=("video",),
            )
        }
    ),
    "wan2.2_animate2_distilled": RequestSpec(
        {
            "animate": TaskSpec(
                dict(
                    VIDEO,
                    image_path=Argument(help="Input image path."),
                    ref_video_prompt=Argument(help="Reference/driving-video prompt for Wan-Animate-2."),
                    pose_video_path=Argument(help="Pose driving video path."),
                    ref_image_paths=Argument(type=str | list[str], help="Reference image paths, separated by commas."),
                    video_path=Argument(help="Input video path."),
                ),
                outputs=("video",),
            )
        }
    ),
    "wan2.2_moe": WAN,
    "wan2.2_moe_vace": WAN_VACE,
    "wan2.2_s2v": RequestSpec(
        {
            "s2v": TaskSpec(
                dict(
                    COMMON | PROMPT,
                    audio_path=Argument(help="Driving audio path."),
                    image_path=Argument(help="Input image path."),
                    pose_video_path=Argument(help="Pose driving video path."),
                ),
                outputs=("video",),
            )
        }
    ),
    "wan21_edenlive": EDEN_LIVE,
    "wan21_edenlive_ar": EDEN_LIVE,
    "wan22_edenlive": EDEN_LIVE,
    "wan22_edenlive_ar": EDEN_LIVE,
    "wan_dancer": WanDancerSpec(
        {
            "s2v": TaskSpec(
                dict(
                    COMMON | PROMPT,
                    audio_path=Argument(help="Driving audio path."),
                    image_path=Argument(help="Input image path."),
                ),
                outputs=("video",),
            )
        }
    ),
}

from lightx2v.models.runners.request_args import COMMON, VIDEO_OUTPUT
from lightx2v.utils.request_spec import Argument, MultiTaskSpec, TaskSpec

REGENERATION = {
    "draft_steps": Argument(
        type=int,
        gt=0,
        strict=True,
        help="MiniMax-H3: only a positive K smaller than configured infer_steps enables a draft prefix and final velocity projection. Omit or use K >= infer_steps for normal sampling.",
    ),
    "resample_source_video": Argument(help="MiniMax-H3 source video with audio; RGB frames are resized to the target canvas before VAE encoding and regeneration."),
    "resample_start_step": Argument(
        type=int,
        ge=0,
        strict=True,
        help="MiniMax-H3 zero-based regeneration start index in [0,infer_steps]. Defaults to the startup config or 1; 0 runs all steps, infer_steps only decodes the VAE-encoded source.",
    ),
}
REQUEST = COMMON | VIDEO_OUTPUT | REGENERATION


SPECS = {
    "minimax_h3": MultiTaskSpec(
        {
            "t2av": TaskSpec(
                dict(
                    REQUEST,
                    prompt=Argument(help="Text prompt."),
                ),
                outputs=("video",),
            ),
            "i2av": TaskSpec(
                dict(
                    REQUEST,
                    image_path=Argument(help="Input image path."),
                    prompt=Argument(help="Text prompt."),
                ),
                outputs=("video",),
            ),
            "l2av": TaskSpec(
                dict(
                    REQUEST,
                    last_frame_path=Argument(help="Last-frame image path."),
                    prompt=Argument(help="Text prompt."),
                ),
                outputs=("video",),
            ),
            "fl2av": TaskSpec(
                dict(
                    REQUEST,
                    image_path=Argument(help="Input image path."),
                    last_frame_path=Argument(help="Last-frame image path."),
                    prompt=Argument(help="Text prompt."),
                ),
                outputs=("video",),
            ),
            "ref2av": TaskSpec(
                dict(
                    REQUEST,
                    audio_path=Argument(type=list[str], help="Reference audio paths."),
                    image_path=Argument(type=list[str], help="Reference image paths."),
                    prompt=Argument(help="Text prompt."),
                    video_path=Argument(type=list[str], help="Reference video paths."),
                ),
                outputs=("video",),
            ),
        },
    ),
}

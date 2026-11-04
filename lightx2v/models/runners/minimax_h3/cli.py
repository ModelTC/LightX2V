from lightx2v.models.runners.request_args import COMMON, VIDEO_OUTPUT
from lightx2v.utils.request_spec import Argument, MultiTaskSpec

RESAMPLING = {
    "draft_steps": Argument(
        type=int, help="MiniMax-H3: only a positive K smaller than configured infer_steps enables a draft prefix and final velocity projection. Omit or use K >= infer_steps for normal sampling."
    ),
    "resample_source_video": Argument(help="MiniMax-H3 source video with audio; RGB frames are resized to the target canvas before VAE encoding and regeneration."),
    "resample_start_step": Argument(
        type=int, help="MiniMax-H3 zero-based regeneration start index in [0,infer_steps]. Defaults to the startup config or 1; 0 runs all steps, infer_steps only decodes the VAE-encoded source."
    ),
}
REQUEST = COMMON | VIDEO_OUTPUT | RESAMPLING


SPECS = {
    "minimax_h3": MultiTaskSpec(
        {
            "t2av": REQUEST | {"prompt": Argument(help="Text prompt.")},
            "i2av": REQUEST
            | {
                "image_path": Argument(help="Input image path; ref2av accepts comma-separated reference images."),
                "prompt": Argument(help="Text prompt."),
            },
            "l2av": REQUEST | {"last_frame_path": Argument(help="Last-frame image path."), "prompt": Argument(help="Text prompt.")},
            "fl2av": REQUEST
            | {
                "image_path": Argument(help="Input image path; ref2av accepts comma-separated reference images."),
                "last_frame_path": Argument(help="Last-frame image path."),
                "prompt": Argument(help="Text prompt."),
            },
            "ref2av": REQUEST
            | {
                "audio_path": Argument(help="Reference audio paths, separated by commas."),
                "image_path": Argument(help="Input image path; ref2av accepts comma-separated reference images."),
                "prompt": Argument(help="Text prompt."),
                "video_path": Argument(help="Reference video paths, separated by commas."),
            },
        },
    ),
}

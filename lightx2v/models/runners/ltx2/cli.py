from lightx2v.models.runners.request_args import COMMON, VIDEO, VIDEO_OUTPUT
from lightx2v.utils.request_spec import Argument, RequestSpec

IMAGE_CONDITION = {
    "image_frame_indices": Argument(help="Comma-separated pixel frame indices, one per image. Omit to space images evenly across the output frames."),
    "image_path": Argument(help="Input image paths, separated by commas."),
    "image_strength": Argument(help="Image conditioning strength: one float, or comma-separated floats for each image."),
}

SPECS = {
    "ltx2": RequestSpec(
        {
            "t2av": VIDEO,
            "i2av": VIDEO | IMAGE_CONDITION,
            "ltx2_s2v": VIDEO | {"audio_path": Argument(help="Driving audio path.")} | IMAGE_CONDITION,
            "v2av": VIDEO
            | IMAGE_CONDITION
            | {
                "mux_audio_video_path": Argument(help="(v2av, optional) After saving, mux audio from this file into the output mp4 (ffmpeg). "),
                "reference_video_frame_cap": Argument(type=int, help="(v2av) Maximum number of frames to read from the reference/control video. Defaults to the full clip."),
                "reference_video_strength": Argument(
                    type=float, help="(v2av) IC-LoRA reference-video conditioning strength in [0.0, 1.0]. 1.0 = full adherence to the control signal, 0.0 = ignore it."
                ),
                "video_path": Argument(help="Input video path."),
            },
        },
    ),
    "ltx2_5": RequestSpec(
        {
            "t2av": COMMON | VIDEO_OUTPUT | {"prompt": Argument(help="Text prompt.")},
            "i2av": COMMON | VIDEO_OUTPUT | IMAGE_CONDITION | {"prompt": Argument(help="Text prompt.")},
        },
    ),
}

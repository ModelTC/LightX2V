from lightx2v.models.runners.request_args import COMMON, VIDEO, VIDEO_OUTPUT, fields
from lightx2v.utils.request_spec import RequestSpec

SPECS = {
    "ltx2": RequestSpec(
        {
            "t2av": VIDEO,
            "i2av": VIDEO | fields("image_frame_indices", "image_path", "image_strength"),
            "ltx2_s2v": VIDEO | fields("audio_path", "image_frame_indices", "image_path", "image_strength"),
            "v2av": VIDEO | fields("image_frame_indices", "image_path", "image_strength", "mux_audio_video_path", "reference_video_frame_cap", "reference_video_strength", "video_path"),
        },
    ),
    "ltx2_5": RequestSpec(
        {
            "t2av": COMMON | VIDEO_OUTPUT | fields("prompt"),
            "i2av": COMMON | VIDEO_OUTPUT | fields("image_frame_indices", "image_path", "image_strength", "prompt"),
        },
    ),
}

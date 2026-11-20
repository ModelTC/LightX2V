from lightx2v.models.runners.request_args import VIDEO, without
from lightx2v.utils.request_spec import Argument, RequestSpec

SPECS = {
    "longcat_video_avatar": RequestSpec(
        {
            "s2v": dict(
                without(VIDEO, "negative_prompt"),
                image_path=Argument(help="Reference image. Optional for one speaker, required for two speakers."),
                resolution=Argument(
                    input_info_default="480p", choices=("480p", "720p"), help="Output resolution preset. A reference image selects the aspect ratio. An explicit size takes precedence."
                ),
                num_segments=Argument(input_info_default=1, type=int, ge=1, strict=True, help="Number of audio-driven segments. Consecutive segments overlap by 13 frames."),
                audio_path=Argument(type=str | list[str], help="Driving audio path. For two speakers, pass a JSON array in person1, person2 order (left to right by default)."),
                audio_type=Argument(input_info_default="para", choices=("para", "add"), help="Two-speaker audio timing: para starts both tracks together, add plays the first track then the second."),
                bbox=Argument(
                    type=dict[str, list],
                    help="Boxes use integer source-image pixels [top, left, bottom, right]. Set person1 and person2 together. Use others for a flat list of silent bystander boxes.",
                ),
            )
        }
    )
}

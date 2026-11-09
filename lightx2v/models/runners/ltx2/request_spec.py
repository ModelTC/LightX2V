from lightx2v.models.runners.request_args import COMMON, VIDEO, VIDEO_OUTPUT
from lightx2v.utils.request_spec import Argument, MultiTaskSpec, TaskSpec

IMAGE_CONDITION = {
    "image_frame_indices": Argument(type=str | list[int], help="Comma-separated pixel frame indices, one per image. Omit to space images evenly across the output frames."),
    "image_path": Argument(help="Input image paths, separated by commas."),
    "image_strength": Argument(type=str | float | list[float], help="Image conditioning strength: one float, or comma-separated floats for each image."),
}


class LTX2Spec(MultiTaskSpec):
    def get_tasks(self, config):
        task = config.get("task")
        if task in ("ltx2_s2v", "v2av"):
            return (task,)
        return ("t2av", "i2av")


SPECS = {
    "ltx2": LTX2Spec(
        {
            "t2av": TaskSpec(VIDEO, outputs=("video",)),
            "i2av": TaskSpec(VIDEO | IMAGE_CONDITION, outputs=("video",)),
            "ltx2_s2v": TaskSpec(dict(VIDEO, audio_path=Argument(help="Driving audio path.")) | IMAGE_CONDITION, outputs=("video",)),
            "v2av": TaskSpec(
                dict(
                    VIDEO | IMAGE_CONDITION,
                    mux_audio_video_path=Argument(help="(v2av, optional) After saving, mux audio from this file into the output mp4 (ffmpeg). "),
                    reference_video_frame_cap=Argument(type=int, help="(v2av) Maximum number of frames to read from the reference/control video. Defaults to the full clip."),
                    reference_video_strength=Argument(
                        type=float, help="(v2av) IC-LoRA reference-video conditioning strength in [0.0, 1.0]. 1.0 = full adherence to the control signal, 0.0 = ignore it."
                    ),
                    video_path=Argument(help="Input video path."),
                ),
                outputs=("video",),
            ),
        },
    ),
    "ltx2_5": MultiTaskSpec(
        {
            "t2av": TaskSpec(
                dict(
                    COMMON | VIDEO_OUTPUT,
                    prompt=Argument(help="Text prompt."),
                ),
                outputs=("video",),
            ),
            "i2av": TaskSpec(
                dict(
                    COMMON | VIDEO_OUTPUT | IMAGE_CONDITION,
                    prompt=Argument(help="Text prompt."),
                ),
                outputs=("video",),
            ),
        },
    ),
}

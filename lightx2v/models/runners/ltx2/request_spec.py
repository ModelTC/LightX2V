from lightx2v.models.runners.request_args import COMMON, VIDEO, VIDEO_OUTPUT
from lightx2v.utils.request_spec import Argument, MultiTaskSpec

IMAGE_CONDITION = {
    "image_frame_indices": Argument(input_info_default=[], type=list[int], help="Pixel frame indices, one per image. Omit to space images evenly across the output frames."),
    "image_path": Argument(input_info_default=[], type=list[str], help="Input image paths."),
    "image_strength": Argument(input_info_default=[1.0], type=list[float], help="Image conditioning strengths: one value for all images, or one per image."),
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
            "t2av": VIDEO,
            "i2av": VIDEO | IMAGE_CONDITION,
            "ltx2_s2v": dict(VIDEO, audio_path=Argument(input_info_default="", help="Driving audio path.")) | IMAGE_CONDITION,
            "v2av": dict(
                VIDEO | IMAGE_CONDITION,
                mux_audio_video_path=Argument(input_info_default="", help="(v2av, optional) After saving, mux audio from this file into the output mp4 (ffmpeg). "),
                reference_video_frame_cap=Argument(type=int, help="(v2av) Maximum number of frames to read from the reference/control video. Defaults to the full clip."),
                reference_video_strength=Argument(
                    input_info_default=1.0, type=float, help="(v2av) IC-LoRA reference-video conditioning strength in [0.0, 1.0]. 1.0 = full adherence to the control signal, 0.0 = ignore it."
                ),
                video_path=Argument(input_info_default="", help="Input video path."),
            ),
        }
    ),
    "ltx2_5": MultiTaskSpec(
        {
            "t2av": dict(
                COMMON | VIDEO_OUTPUT,
                prompt=Argument(input_info_default="", help="Text prompt."),
            ),
            "i2av": dict(
                COMMON | VIDEO_OUTPUT | IMAGE_CONDITION,
                prompt=Argument(input_info_default="", help="Text prompt."),
            ),
        },
        context=VIDEO,
    ),
}

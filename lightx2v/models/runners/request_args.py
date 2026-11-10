"""Common request definitions; model-specific fields belong in each model's request_spec.py."""

from lightx2v.utils.request_spec import Argument, TaskSpec


class ImageVideoTaskSpec(TaskSpec):
    def __init__(self, arguments):
        super().__init__(arguments, outputs=("image", "video"))

    def get_output_type(self, inputs):
        if inputs.get("video_path", ""):
            return "video"
        if inputs.get("image_path", ""):
            return "image"
        return None


COMMON = {
    "return_result_tensor": Argument(type=bool, server=False, action="store_true", help="Whether to return result tensor. (Useful for comfyui)"),
    "save_result_path": Argument(help="The path to save video path/file"),
    "seed": Argument(type=int, help="The seed for random generator"),
    "task": Argument(cli=False),
}

PROMPT = {"negative_prompt": Argument(help="Negative text prompt."), "prompt": Argument(help="Text prompt.")}

VIDEO_OUTPUT = {
    "size": Argument(type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
    "num_frames": Argument(type=int, help="Requested output frame count. Model-specific length constraints apply."),
}

IMAGE = dict(
    COMMON | PROMPT,
    aspect_ratio=Argument(help="Output aspect ratio."),
    size=Argument(type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
)
VIDEO = COMMON | PROMPT | VIDEO_OUTPUT


def without(arguments, *names):
    return {name: argument for name, argument in arguments.items() if name not in names}

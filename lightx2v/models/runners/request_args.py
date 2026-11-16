"""Common request definitions; model-specific fields belong in each model's request_spec.py."""

from lightx2v.utils.request_spec import Argument

COMMON = {
    "return_result_tensor": Argument(input_info_default=False, type=bool, server=False, action="store_true", help="Whether to return result tensor. (Useful for comfyui)"),
    "save_result_path": Argument(help="The path to save video path/file"),
    "seed": Argument(input_info_default=0, type=int, help="The seed for random generator"),
    "task": Argument(cli=False),
}

PROMPT = {"negative_prompt": Argument(input_info_default="", help="Negative text prompt."), "prompt": Argument(input_info_default="", help="Text prompt.")}

VIDEO_OUTPUT = {
    "size": Argument(input_info_default=[], type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
    "num_frames": Argument(type=int, help="Requested output frame count. Model-specific length constraints apply."),
}

IMAGE = dict(
    COMMON | PROMPT,
    aspect_ratio=Argument(input_info_default="", help="Output aspect ratio."),
    size=Argument(input_info_default=[], type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
)
VIDEO = COMMON | PROMPT | VIDEO_OUTPUT


def without(arguments, *names):
    return {name: argument for name, argument in arguments.items() if name not in names}

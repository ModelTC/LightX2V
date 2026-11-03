from lightx2v.models.runners.request_args import COMMON, VIDEO_OUTPUT, fields
from lightx2v.utils.request_spec import MultiTaskSpec

SPECS = {
    "minimax_h3_causal": MultiTaskSpec({"refa2v": COMMON | VIDEO_OUTPUT | fields("image_path", "audio_path", "prompt", "action_prompts")}),
}

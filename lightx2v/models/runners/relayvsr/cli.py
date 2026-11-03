from lightx2v.models.runners.request_args import COMMON, fields
from lightx2v.utils.request_spec import RequestSpec

RELAY_VSR = RequestSpec({"sr": COMMON | fields("video_path", "image_path", "sr_ratio")})

SPECS = {
    "relayvsr_x2": RELAY_VSR,
    "relayvsr_x4": RELAY_VSR,
}

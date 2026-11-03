from lightx2v.models.runners.request_args import COMMON, VIDEO_OUTPUT, fields
from lightx2v.utils.request_spec import MultiTaskSpec

SPECS = {
    "streamrender_h3": MultiTaskSpec({"refv2v": COMMON | VIDEO_OUTPUT | fields("prompt", "image_path", "video_path")}),
}

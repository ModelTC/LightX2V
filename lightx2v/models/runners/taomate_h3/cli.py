from lightx2v.models.runners.request_args import COMMON, VIDEO_OUTPUT, fields
from lightx2v.utils.request_spec import MultiTaskSpec

SPECS = {
    "taomate_h3": MultiTaskSpec({"t2av": COMMON | VIDEO_OUTPUT | fields("prompt")}),
}

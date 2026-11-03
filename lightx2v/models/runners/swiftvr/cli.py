from lightx2v.models.runners.request_args import COMMON, fields
from lightx2v.utils.request_spec import RequestSpec

SPECS = {
    "swiftvr": RequestSpec({"sr": COMMON | fields("image_path", "sr_ratio", "size", "video_path")}),
}

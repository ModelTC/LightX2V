from lightx2v.models.runners.request_args import COMMON, fields, without
from lightx2v.utils.request_spec import RequestSpec

SPECS = {
    "hunyuan3d": RequestSpec({"i23d": without(COMMON, "return_result_tensor") | fields("image_path")}),
}

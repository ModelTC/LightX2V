from lightx2v.models.runners.request_args import COMMON, without
from lightx2v.utils.request_spec import Argument, RequestSpec

SPECS = {
    "hunyuan3d": RequestSpec({"i23d": without(COMMON, "return_result_tensor") | {"image_path": Argument(help="Input image path.")}}),
}

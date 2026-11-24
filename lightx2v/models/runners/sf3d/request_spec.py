from lightx2v.models.runners.request_args import COMMON, without
from lightx2v.utils.request_spec import Argument, RequestSpec

SPECS = {"sf3d": RequestSpec({"i23d": dict(without(COMMON, "return_result_tensor"), image_path=Argument(input_info_default="", help="One input image path."))})}

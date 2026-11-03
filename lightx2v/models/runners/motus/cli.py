from lightx2v.models.runners.request_args import COMMON, fields, without
from lightx2v.models.runners.wan.cli import WanSpec

SPECS = {
    "motus": WanSpec({"i2v": without(COMMON, "return_result_tensor") | fields("image_path", "prompt", "save_action_path", "state_path")}),
}

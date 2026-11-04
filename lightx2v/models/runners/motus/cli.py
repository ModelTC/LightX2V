from lightx2v.models.runners.request_args import COMMON, without
from lightx2v.models.runners.wan.cli import WanSpec
from lightx2v.utils.request_spec import Argument

SPECS = {
    "motus": WanSpec(
        {
            "i2v": dict(
                without(COMMON, "return_result_tensor"),
                image_path=Argument(help="Input image path."),
                prompt=Argument(help="Text prompt."),
                save_action_path=Argument(help="Output action file path."),
                state_path=Argument(help="Input robot state file."),
            )
        }
    ),
}

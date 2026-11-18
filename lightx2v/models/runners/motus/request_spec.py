from lightx2v.models.runners.request_args import COMMON, VIDEO, without
from lightx2v.utils.request_spec import Argument, RequestSpec

SPECS = {
    "motus": RequestSpec(
        {
            "i2v": dict(
                without(COMMON, "return_result_tensor"),
                image_path=Argument(input_info_default="", help="Input image path."),
                prompt=Argument(input_info_default="", help="Text prompt."),
                save_action_path=Argument(input_info_default="", help="Output action file path."),
                state_path=Argument(input_info_default="", help="Input robot state file."),
            )
        },
        context=VIDEO,
    ),
}

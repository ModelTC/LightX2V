from lightx2v.models.runners.request_args import COMMON, PROMPT, VIDEO
from lightx2v.utils.request_spec import Argument, MultiTaskSpec


class Cosmos3Spec(MultiTaskSpec):
    def get_tasks(self, config):
        variant = config["model_variant"]
        if variant == "omni":
            return super().get_tasks(config)
        return ({"t2i": "t2i", "i2v": "i2v", "policy": "i2va"}[variant],)


SPECS = {
    "cosmos3": Cosmos3Spec(
        {
            "t2i": dict(
                COMMON | PROMPT,
                size=Argument(input_info_default=[], type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
            ),
            "t2v": VIDEO,
            "i2v": dict(
                VIDEO,
                image_path=Argument(input_info_default="", help="Input image path."),
            ),
            "t2av": VIDEO,
            "i2av": dict(
                VIDEO,
                image_path=Argument(input_info_default="", help="Input image path."),
            ),
            "i2va": dict(
                COMMON | PROMPT,
                action_mode=Argument(input_info_default="", choices=["forward_dynamics", "inverse_dynamics", "policy"], help="Cosmos3 action mode."),
                action_path=Argument(input_info_default="", help="Input action path."),
                domain_name=Argument(input_info_default="", help="Cosmos3 action embodiment domain name."),
                image_path=Argument(input_info_default="", help="Input image path."),
                policy_image=Argument(server=False, cli=False, repr=False),
                policy_state=Argument(server=False, cli=False, repr=False),
                save_action_path=Argument(input_info_default="", help="Output action file path."),
                state_path=Argument(input_info_default="", help="Input robot state file."),
                size=Argument(input_info_default=[], type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
                video_path=Argument(input_info_default="", help="Input video path."),
                view_point=Argument(input_info_default="", help="Cosmos3 action viewpoint label."),
            ),
            "v2av": dict(
                COMMON | PROMPT,
                action_mode=Argument(input_info_default="", choices=["forward_dynamics", "inverse_dynamics", "policy"], help="Cosmos3 action mode."),
                action_path=Argument(input_info_default="", help="Input action path."),
                domain_name=Argument(input_info_default="", help="Cosmos3 action embodiment domain name."),
                image_path=Argument(input_info_default="", help="Input image path."),
                save_action_path=Argument(input_info_default="", help="Output action file path."),
                size=Argument(input_info_default=[], type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
                video_path=Argument(input_info_default="", help="Input video path."),
                view_point=Argument(input_info_default="", help="Cosmos3 action viewpoint label."),
            ),
        }
    ),
}

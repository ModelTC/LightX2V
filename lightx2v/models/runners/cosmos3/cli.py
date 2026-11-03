from lightx2v.models.runners.request_args import COMMON, PROMPT, VIDEO
from lightx2v.utils.request_spec import Argument, RequestSpec

SPECS = {
    "cosmos3": RequestSpec(
        {
            "t2i": COMMON | PROMPT | {"size": Argument(type=int, nargs="+", help="Output size in pixels: HEIGHT WIDTH")},
            "t2v": VIDEO,
            "i2v": VIDEO | {"image_path": Argument(help="Input image path.")},
            "t2av": VIDEO,
            "i2av": VIDEO | {"image_path": Argument(help="Input image path.")},
            "i2va": COMMON
            | PROMPT
            | {
                "action_mode": Argument(choices=["forward_dynamics", "inverse_dynamics", "policy"], help="Cosmos3 action mode."),
                "action_path": Argument(help="Input action path."),
                "domain_name": Argument(help="Cosmos3 action embodiment domain name."),
                "image_path": Argument(help="Input image path."),
                "policy_image": Argument(cli=False),
                "policy_state": Argument(cli=False),
                "save_action_path": Argument(help="Output action file path."),
                "state_path": Argument(help="Input robot state file."),
                "size": Argument(type=int, nargs="+", help="Output size in pixels: HEIGHT WIDTH"),
                "video_path": Argument(help="Input video path."),
                "view_point": Argument(help="Cosmos3 action viewpoint label."),
            },
            "v2av": COMMON
            | PROMPT
            | {
                "action_mode": Argument(choices=["forward_dynamics", "inverse_dynamics", "policy"], help="Cosmos3 action mode."),
                "action_path": Argument(help="Input action path."),
                "domain_name": Argument(help="Cosmos3 action embodiment domain name."),
                "image_path": Argument(help="Input image path."),
                "save_action_path": Argument(help="Output action file path."),
                "size": Argument(type=int, nargs="+", help="Output size in pixels: HEIGHT WIDTH"),
                "video_path": Argument(help="Input video path."),
                "view_point": Argument(help="Cosmos3 action viewpoint label."),
            },
        },
    ),
}

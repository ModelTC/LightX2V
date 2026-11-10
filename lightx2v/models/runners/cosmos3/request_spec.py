from lightx2v.models.runners.request_args import COMMON, PROMPT, VIDEO
from lightx2v.utils.request_spec import Argument, MultiTaskSpec, TaskSpec


class Cosmos3Spec(MultiTaskSpec):
    def get_tasks(self, config):
        variant = config["model_variant"]
        if variant == "omni":
            return super().get_tasks(config)
        return ({"t2i": "t2i", "i2v": "i2v", "policy": "i2va"}[variant],)

    def get_tasks_for_output(self, config, output):
        if config["model_variant"] == "policy" and not config.get("decode_video", False):
            return ()
        return super().get_tasks_for_output(config, output)


SPECS = {
    "cosmos3": Cosmos3Spec(
        {
            "t2i": TaskSpec(
                dict(
                    COMMON | PROMPT,
                    size=Argument(type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
                ),
                outputs=("image",),
            ),
            "t2v": TaskSpec(VIDEO, outputs=("video",)),
            "i2v": TaskSpec(
                dict(
                    VIDEO,
                    image_path=Argument(help="Input image path."),
                ),
                outputs=("video",),
            ),
            "t2av": TaskSpec(VIDEO, outputs=("video",)),
            "i2av": TaskSpec(
                dict(
                    VIDEO,
                    image_path=Argument(help="Input image path."),
                ),
                outputs=("video",),
            ),
            "i2va": TaskSpec(
                dict(
                    COMMON | PROMPT,
                    action_mode=Argument(choices=["forward_dynamics", "inverse_dynamics", "policy"], help="Cosmos3 action mode."),
                    action_path=Argument(help="Input action path."),
                    domain_name=Argument(help="Cosmos3 action embodiment domain name."),
                    image_path=Argument(help="Input image path."),
                    policy_image=Argument(server=False, cli=False),
                    policy_state=Argument(server=False, cli=False),
                    save_action_path=Argument(help="Output action file path."),
                    state_path=Argument(help="Input robot state file."),
                    size=Argument(type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
                    video_path=Argument(help="Input video path."),
                    view_point=Argument(help="Cosmos3 action viewpoint label."),
                ),
                outputs=("video",),
            ),
            "v2av": TaskSpec(
                dict(
                    COMMON | PROMPT,
                    action_mode=Argument(choices=["forward_dynamics", "inverse_dynamics", "policy"], help="Cosmos3 action mode."),
                    action_path=Argument(help="Input action path."),
                    domain_name=Argument(help="Cosmos3 action embodiment domain name."),
                    image_path=Argument(help="Input image path."),
                    save_action_path=Argument(help="Output action file path."),
                    size=Argument(type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
                    video_path=Argument(help="Input video path."),
                    view_point=Argument(help="Cosmos3 action viewpoint label."),
                ),
                outputs=("video",),
            ),
        },
    ),
}

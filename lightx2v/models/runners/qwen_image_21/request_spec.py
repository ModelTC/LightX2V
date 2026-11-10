from lightx2v.models.runners.request_args import IMAGE
from lightx2v.utils.request_spec import Argument, RequestSpec, TaskSpec


class QwenImage21Spec(RequestSpec):
    def get_tasks(self, config):
        return super().get_tasks(config) if config.get("task", None) is not None else tuple(self.tasks)


SPECS = {
    "qwen_image_21": QwenImage21Spec(
        {
            "t2i": TaskSpec(
                dict(
                    IMAGE,
                    resolution=Argument(type=int, gt=0, help="Qwen-Image-2.1 reference/automatic output resolution; defaults to the startup config"),
                ),
                outputs=("image",),
            ),
            "i2i": TaskSpec(
                dict(
                    IMAGE,
                    image_path=Argument(type=list[str], help="Input image paths."),
                    resolution=Argument(type=int, gt=0, help="Qwen-Image-2.1 reference/automatic output resolution; defaults to the startup config"),
                ),
                outputs=("image",),
            ),
        },
    ),
}

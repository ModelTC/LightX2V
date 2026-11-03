from lightx2v.models.runners.request_args import IMAGE, fields
from lightx2v.utils.request_spec import RequestSpec


class QwenImage21Spec(RequestSpec):
    def get_tasks(self, config):
        return super().get_tasks(config) if config.get("task", None) is not None else tuple(self.tasks)


SPECS = {
    "qwen_image_21": QwenImage21Spec(
        {
            "t2i": IMAGE | fields("resolution"),
            "i2i": IMAGE | fields("image_path", "resolution"),
        },
    ),
}

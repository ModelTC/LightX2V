from lightx2v.models.runners.request_args import COMMON, fields
from lightx2v.utils.request_spec import RequestSpec


class HunyuanImage3Spec(RequestSpec):
    def get_arguments(self, config, task):
        arguments = super().get_arguments(config, task)
        if config.get("image_size", None):
            arguments.pop("size", None)
        return arguments


TEXT = COMMON | fields("bot_task", "max_new_tokens", "prompt", "stream_callback", "system_prompt", "text_do_sample", "text_temperature", "text_top_k", "text_top_p")

SPECS = {
    "hunyuan_image3": HunyuanImage3Spec(
        {
            "t2t": TEXT,
            "ti2t": TEXT | fields("image_path", "align_image_size"),
            "t2i": COMMON | fields("prompt", "size"),
            "ti2i": COMMON | fields("image_path", "align_image_size", "prompt", "size"),
            "i2i": COMMON | fields("image_path", "align_image_size", "prompt", "size"),
        },
    ),
}

from lightx2v.models.runners.hunyuan_video.cli import HUNYUAN_VIDEO15, HunyuanVideoSpec
from lightx2v.models.runners.request_args import fields

WORLDPLAY = HunyuanVideoSpec({task: request_fields | fields("pose") for task, request_fields in HUNYUAN_VIDEO15.tasks.items()})

SPECS = {
    "worldplay_ar": WORLDPLAY,
    "worldplay_bi": WORLDPLAY,
    "worldplay_distill": WORLDPLAY,
}

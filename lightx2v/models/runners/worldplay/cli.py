from lightx2v.models.runners.hunyuan_video.cli import HUNYUAN_VIDEO15, HunyuanVideoSpec
from lightx2v.utils.request_spec import Argument

WORLDPLAY = HunyuanVideoSpec(
    {
        task: dict(
            request_fields,
            pose=Argument(help="Pose string (e.g., 'w-3, right-0.5') or JSON file path for WorldPlay models."),
        )
        for task, request_fields in HUNYUAN_VIDEO15.tasks.items()
    }
)

SPECS = {
    "worldplay_ar": WORLDPLAY,
    "worldplay_bi": WORLDPLAY,
    "worldplay_distill": WORLDPLAY,
}

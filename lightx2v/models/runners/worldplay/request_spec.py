from lightx2v.models.runners.hunyuan_video.request_spec import HUNYUAN_VIDEO15
from lightx2v.utils.request_spec import Argument, RequestSpec

WORLDPLAY = RequestSpec(
    {
        task: dict(
            request_fields,
            pose=Argument(type=str | dict, help="Pose string (e.g., 'w-3, right-0.5') or JSON file path for WorldPlay models."),
        )
        for task, request_fields in HUNYUAN_VIDEO15.tasks.items()
    }
)

SPECS = {
    "worldplay_ar": WORLDPLAY,
    "worldplay_bi": WORLDPLAY,
    "worldplay_distill": WORLDPLAY,
}

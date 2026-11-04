from lightx2v.models.runners.request_args import COMMON, without
from lightx2v.utils.request_spec import Argument, RequestSpec

SPECS = {
    "depth_anything3": RequestSpec(
        {
            "depth": without(COMMON, "return_result_tensor")
            | {
                "image_path": Argument(help="Input image path."),
                "video_path": Argument(help="Input video path."),
                "save_depth_path": Argument(help="Optional float32 NPY for an image, or streaming NPZ for a video."),
                "save_sky_mask_path": Argument(help="Optional binary PNG for an image, or streaming NPZ for a video; may share the depth NPZ path."),
            }
        }
    ),
}

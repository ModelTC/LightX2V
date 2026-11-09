import argparse

from lightx2v.models.runners.request_args import COMMON
from lightx2v.utils.request_spec import Argument, RequestSpec, TaskSpec

SPECS = {
    "worldmirror": RequestSpec(
        {
            "recon": TaskSpec(
                dict(
                    COMMON,
                    input_path=Argument(help="(worldmirror/recon) Path to a directory of images, a video file, or a single image."),
                    prior_cam_path=Argument(help="(worldmirror/recon) Optional camera prior JSON (extrinsics + intrinsics)."),
                    prior_depth_path=Argument(help="(worldmirror/recon) Optional depth prior directory (one .npy/.png per image)."),
                    render_depth=Argument(type=bool, action=argparse.BooleanOptionalAction, help="(worldmirror/recon) Also render a depth video with --save_rendered."),
                    render_interp_per_pair=Argument(type=int, help="(worldmirror/recon) Interpolated frames per camera pair for --save_rendered."),
                    save_rendered=Argument(type=bool, action=argparse.BooleanOptionalAction, help="(worldmirror/recon) Render an interpolated fly-through video from Gaussian splats."),
                    strict_output_path=Argument(help="(worldmirror/recon) If set, write outputs directly here instead of under save_result_path/<subdir>/<timestamp>/."),
                ),
                outputs=("reconstruction",),
            )
        }
    ),
}

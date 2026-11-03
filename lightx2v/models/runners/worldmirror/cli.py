from lightx2v.models.runners.request_args import COMMON, fields
from lightx2v.utils.request_spec import RequestSpec

SPECS = {
    "worldmirror": RequestSpec({"recon": COMMON | fields("input_path", "prior_cam_path", "prior_depth_path", "render_depth", "render_interp_per_pair", "save_rendered", "strict_output_path")}),
}

"""Prepared audio conditions shared by CLI and HTTP."""

from lightx2v.models.runners.request_args import COMMON
from lightx2v.utils.request_spec import Argument, RequestSpec

SPECS = {
    "seedvc": RequestSpec(
        {
            "a2a": COMMON
            | {
                "sample_path": Argument(help="Prepared sample JSON; tensor paths are relative to this file."),
                "diffusion_steps": Argument(type=int, input_info_default=30, ge=1, le=200),
                "cfg_rate": Argument(type=float, input_info_default=0.7, ge=0, le=10),
                "length_adjust": Argument(type=float, input_info_default=1.0, gt=0, le=4),
                "semitone_shift": Argument(type=int, input_info_default=0, ge=-24, le=24),
                "auto_f0_adjust": Argument(type=bool, input_info_default=False),
            },
        }
    ),
}

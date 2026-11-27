"""Lightweight public contract: one end-to-end audio task."""

from lightx2v.models.runners.request_args import COMMON
from lightx2v.utils.request_spec import Argument, RequestSpec

SPECS = {
    "yue2": RequestSpec(
        {
            "t2a": COMMON
            | {
                "style": Argument(help="Music style, instruments and vocal description."),
                "lyrics": Argument(help="Lyrics with section markers; empty string requests instrumental music."),
                "abc": Argument(help="Optional prepared ABC text; omitted for automatic planning."),
                "cot": Argument(input_info_default="full", choices=("full", "melody", "off"), help="YuE2 score-conditioning mode."),
                "cfg_scale": Argument(type=float, ge=0, le=20, help="AR guidance; defaults to 1.01 for off and 1.0 otherwise."),
                "id": Argument(input_info_default="song", help="Sample identifier."),
                "abc_sampling": Argument(type=dict, help="Optional native ABC sampling overrides as JSON."),
                "semantic_sampling": Argument(type=dict, help="Optional native music sampling overrides as JSON."),
            },
        }
    ),
}

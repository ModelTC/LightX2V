from lightx2v.models.runners.request_args import COMMON, PROMPT
from lightx2v.utils.request_spec import Argument, RequestSpec

SPECS = {"sdxl": RequestSpec({"t2i": COMMON | PROMPT | {"size": Argument(type=list[int], input_info_default=[1024, 1024], help="Output HEIGHT WIDTH; multiples of 64, at least 128.")}})}

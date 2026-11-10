from lightx2v.models.runners.request_args import IMAGE
from lightx2v.utils.request_spec import RequestSpec, TaskSpec

SPECS = {
    "ernie_image": RequestSpec({"t2i": TaskSpec(IMAGE, outputs=("image",))}),
}

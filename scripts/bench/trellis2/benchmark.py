from pathlib import Path

from lightx2v.common.ops.sparse import SparseConv3d
from lightx2v.common.ops.sparse.neighbor_map import NEIGHBOR_MAPS
from scripts.bench.reconstruction.benchmark import run_benchmark


def set_variant(runner, variant):
    builder = NEIGHBOR_MAPS["cpu_hash" if variant == "baseline" else "sorted"]
    for model in runner.models.values():
        for layer in model.modules():
            if isinstance(layer, SparseConv3d):
                layer.neighbor_map = builder


def call_targets(runner):
    return [(stage, sampler, "sample_once") for stage, sampler in runner.samplers.items()]


def request_fields(request, input_root, output, variant):
    return {"image_path": str(input_root / request["image"]["path"]), "save_result_path": str(output / f"{request['id']}_{variant}.glb")}


if __name__ == "__main__":
    run_benchmark("trellis2", "i23d", Path(__file__).with_name("requests.json"), call_targets, set_variant, request_fields)

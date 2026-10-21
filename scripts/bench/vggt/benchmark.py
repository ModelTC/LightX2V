from pathlib import Path

from scripts.bench.reconstruction.benchmark import run_benchmark


def set_variant(runner, variant):
    runner.config["preprocess_workers"] = 1 if variant == "baseline" else 8


def call_targets(runner):
    return [("vggt_forward", runner.model, "forward")]


def request_fields(request, input_root, _output, _variant):
    return {"input_path": [str(input_root / image["path"]) for image in request["images"]]}


if __name__ == "__main__":
    run_benchmark("vggt", "recon", Path(__file__).with_name("requests.json"), call_targets, set_variant, request_fields)

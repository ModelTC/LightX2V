import pytest

from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.request_fields import COMMON_REQUEST_FIELDS
from lightx2v.pipeline import LightX2VPipeline


class SeedRunner(BaseRunner):
    supported_request_fields_by_task = {"t2i": COMMON_REQUEST_FIELDS}

    def run_pipeline(self, input_info):
        return input_info.seed


def test_request_preparation_does_not_apply_seed(monkeypatch):
    seeded = []
    monkeypatch.setattr("lightx2v.models.runners.base_runner.seed_all", seeded.append)
    runner = SeedRunner({"task": "t2i"})
    runner._gc_frozen = True

    input_info = runner.prepare_request({"task": "t2i", "seed": 7})
    assert seeded == []

    assert runner.run_request(input_info) == 7
    assert seeded == [7]


@pytest.mark.parametrize("config", [{}, {"seed": None}, {"seed": 73}, {"seed": 0}])
@pytest.mark.parametrize("request_data", [{}, {"seed": None}, {"seed": 0}, {"seed": 12}])
def test_python_and_direct_requests_resolve_the_same_seed(config, request_data):
    runner = SeedRunner({"task": "t2i", **config})
    pipeline = LightX2VPipeline(task="t2i", model_cls="test")
    pipeline.runner = runner
    expected = request_data.get("seed")
    if expected is None:
        expected = 42

    assert runner.prepare_request({"task": runner.config["task"], **request_data}).seed == expected
    runner.run_request = lambda input_info: input_info.seed
    assert pipeline.generate(**request_data) == expected
    assert runner.config == {"task": "t2i", **config}


def test_fixed_request_seeds_do_not_replace_json(monkeypatch):
    seeded = []
    monkeypatch.setattr("lightx2v.models.runners.base_runner.seed_all", seeded.append)
    runner = SeedRunner({"task": "t2i", "seed": 123})
    runner._gc_frozen = True

    first = runner.prepare_request({"task": runner.config["task"]})
    assert runner.run_request(first) == 42
    assert runner.run_request(first) == 42
    for request_data, expected in [({"seed": 456}, 456), ({"seed": 0}, 0), ({}, 42)]:
        input_info = runner.prepare_request({"task": runner.config["task"], **request_data})
        assert runner.run_request(input_info) == expected
    assert seeded == [42, 42, 456, 0, 42]
    assert runner.config["seed"] == 123

"""Projection caches must release request tensors and invalidate on reload."""

from types import SimpleNamespace

import pytest

from lightx2v.models.networks.wan.edenlive_model import EdenLiveModel
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.runners.wan import edenlive_runner as module


def make_runner(events):
    runner = module.Wan22EdenLiveARRunner.__new__(module.Wan22EdenLiveARRunner)
    runner.model = SimpleNamespace(
        transformer_infer=SimpleNamespace(clear_request_state=lambda: events.append("transformer")),
        pre_infer=SimpleNamespace(clear_request_state=lambda: events.append("pre")),
    )
    runner.scheduler = SimpleNamespace(clear=lambda: events.append("scheduler"))
    runner.recording_reader = SimpleNamespace(close=lambda: events.append("reader"))
    return runner


@pytest.mark.parametrize("fail", [False, True])
def test_pipeline_releases_pre_infer_on_success_and_failure(monkeypatch, fail):
    events = []
    runner = make_runner(events)

    def pipeline(self, info):
        if fail:
            raise RuntimeError("inference failed")
        return info

    monkeypatch.setattr(module.WanRunner, "run_pipeline", pipeline)
    if fail:
        with pytest.raises(RuntimeError, match="inference failed"):
            runner.run_pipeline("request")
    else:
        assert runner.run_pipeline("request") == "request"
    assert events == ["transformer", "pre", "reader"]
    assert runner.recording_reader is None


def test_warmup_releases_pre_infer():
    events = []
    runner = make_runner(events)
    runner.clear_warmup_state()
    assert events == ["scheduler", "transformer", "pre"]


def test_end_run_releases_pre_infer_before_base_cleanup(monkeypatch):
    events = []
    runner = make_runner(events)
    monkeypatch.setattr(module.WanRunner, "end_run", lambda self: events.append("base"))
    runner.end_run()
    assert events == ["transformer", "pre", "base"]
    assert runner.proxy_pixels is None and runner.prompt_embeddings == {} and runner.prompts == []


def test_reload_invalidates_pre_infer_before_loading_weights(monkeypatch):
    events = []
    model = EdenLiveModel.__new__(EdenLiveModel)
    model.pre_infer = SimpleNamespace(clear_request_state=lambda: events.append("pre"))
    model.transformer_infer = SimpleNamespace(invalidate_cuda_graph=lambda: events.append("graph"))
    model.transformer_weights = SimpleNamespace(blocks=[])
    model.config = {}
    monkeypatch.setattr(WanModel, "_apply_weights", lambda self, weights: events.append(("load", weights)))
    model._apply_weights("replacement")
    assert events == ["pre", "graph", ("load", "replacement")]

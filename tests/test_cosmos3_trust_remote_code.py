from __future__ import annotations

from unittest.mock import patch

from lightx2v.models.runners.cosmos3.cosmos3_runner import Cosmos3Runner


def _runner_with_config(config: dict) -> Cosmos3Runner:
    runner = Cosmos3Runner.__new__(Cosmos3Runner)
    runner.config = config
    return runner


def test_load_text_encoder_defaults_trust_remote_code_to_false(tmp_path):
    config = {"model_path": str(tmp_path)}
    runner = _runner_with_config(config)
    with patch("lightx2v.models.runners.cosmos3.cosmos3_runner.AutoTokenizer.from_pretrained") as from_pretrained:
        runner.load_text_encoder()
    assert from_pretrained.call_args.kwargs["trust_remote_code"] is False


def test_load_text_encoder_honors_explicit_trust_remote_code_opt_in(tmp_path):
    config = {"model_path": str(tmp_path), "trust_remote_code": True}
    runner = _runner_with_config(config)
    with patch("lightx2v.models.runners.cosmos3.cosmos3_runner.AutoTokenizer.from_pretrained") as from_pretrained:
        runner.load_text_encoder()
    assert from_pretrained.call_args.kwargs["trust_remote_code"] is True

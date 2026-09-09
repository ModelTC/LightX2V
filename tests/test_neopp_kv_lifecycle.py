import weakref
from types import SimpleNamespace

import pytest
import torch

from lightx2v.models.runners.neopp.neopp_runner import NeoppRunner


@pytest.mark.parametrize("failure_stage", [None, "encoder", "main"])
def test_pipeline_releases_injected_kv_on_success_and_failure(failure_stage):
    runner = NeoppRunner.__new__(NeoppRunner)
    runner._gc_frozen = True
    runner.model = SimpleNamespace(transformer_infer=SimpleNamespace(kv_cache={}))
    runner.past_key_values_cond = torch.ones(1)
    runner.past_key_values_uncond = torch.zeros(1)
    tensor_refs = [weakref.ref(runner.past_key_values_cond), weakref.ref(runner.past_key_values_uncond)]
    failure = RuntimeError(f"{failure_stage} failed")

    def encode():
        if failure_stage == "encoder":
            raise failure
        return {"past_key_values_cond": runner.past_key_values_cond, "past_key_values_uncond": runner.past_key_values_uncond}

    def run_main():
        assert runner.inputs["past_key_values_cond"] is runner.past_key_values_cond
        assert runner.inputs["past_key_values_uncond"] is runner.past_key_values_uncond
        runner.model.transformer_infer.kv_cache.update(runner.inputs)
        if failure_stage == "main":
            raise failure
        return b"encoded image"

    runner.run_input_encoder = encode
    runner.run_main = run_main
    if failure_stage is None:
        assert runner.run_pipeline(SimpleNamespace()) == b"encoded image"
    else:
        with pytest.raises(RuntimeError) as raised:
            runner.run_pipeline(SimpleNamespace())
        assert raised.value is failure

    assert runner.past_key_values_cond is None
    assert runner.past_key_values_uncond is None
    assert runner.inputs == {}
    assert runner.model.transformer_infer.kv_cache == {}
    assert all(tensor_ref() is None for tensor_ref in tensor_refs)

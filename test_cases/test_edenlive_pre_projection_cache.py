"""Request-scoped pre-infer projections preserve inputs, results, and refreshes."""

from types import SimpleNamespace

import pytest
import torch

from lightx2v.common.ops.mm.mm_weight import MMWeight
from lightx2v.models.networks.wan.infer.edenlive.pre_infer import EdenLivePreInfer
from lightx2v.models.schedulers.wan.edenlive.scheduler import EdenLiveARScheduler


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device(request.param)


def _setup(device="cpu", enabled=True):
    device = torch.device(device)
    torch.manual_seed(123)
    infer = EdenLivePreInfer.__new__(EdenLivePreInfer)
    infer.config = dict(cache_pre_infer_projections=enabled, use_image_encoder=False, proxy_unconditional_control="same", proxy_condition_rmsnorm=False)
    infer.is_wan21 = False
    infer.freq_dim = infer.dim = 8
    infer.infer_dtype = infer.sensitive_layer_dtype = torch.bfloat16
    infer.clear_projection_cache()
    scheduler = EdenLiveARScheduler.__new__(EdenLiveARScheduler)
    scheduler.generator = torch.Generator(device=device).manual_seed(123)
    scheduler.infer_condition = True
    scheduler.timesteps = torch.tensor([1000, 937.5, 833.3333333, 625], device=device, dtype=torch.float32)
    scheduler.latents_input = torch.randn(8, 3, 2, 2, device=device, dtype=torch.bfloat16)
    infer.scheduler = scheduler

    def linear(name, cols, rows):
        module = MMWeight(name + ".weight", name + ".bias")
        module.weight = torch.randn(cols, rows, device=device, dtype=torch.bfloat16) * 0.02
        module.bias = torch.randn(rows, device=device, dtype=torch.bfloat16) * 0.02
        module.calls = 0
        original = module.apply

        def apply(value):
            module.calls += 1
            return original(value)

        module.apply = apply
        return module

    weights = SimpleNamespace(
        patch_embedding=SimpleNamespace(apply=lambda value: value[:, :8].contiguous()),
        text_embedding_0=linear("text0", 6, 12),
        text_embedding_2=linear("text2", 12, 8),
        time_embedding_0=linear("time0", 8, 12),
        time_embedding_2=linear("time2", 12, 8),
        time_projection_1=linear("timeproj", 8, 48),
    )
    inputs = {
        "image_encoder_output": {},
        "proxy_features": torch.randn(1, 1024, 3, 2, 2, device=device, dtype=torch.bfloat16),
        "text_encoder_output": {key: torch.randn(1, 5, 6, device=device, dtype=torch.bfloat16) for key in ("context", "context_null")},
    }
    _step(infer, 0)
    return infer, weights, inputs


def _step(infer, step, start=3, update=False):
    scheduler = infer.scheduler
    scheduler.step_index, scheduler.chunk_start, scheduler.cache_update = step, start, update
    scheduler.timestep_input = torch.zeros(1, device=scheduler.timesteps.device) if update else scheduler.timesteps[step : step + 1].float()
    if start == 0 and not update:
        scheduler.timestep_input = scheduler.timestep_input.expand(3).clone()
        scheduler.timestep_input[0] = 0


def _reference(infer, weights, inputs):
    enabled = infer.config["cache_pre_infer_projections"]
    infer.config["cache_pre_infer_projections"] = False
    output = infer.infer(weights, inputs)
    infer.config["cache_pre_infer_projections"] = enabled
    return output


def _assert_equal(actual, expected):
    for name in ("x", "embed", "embed0", "context"):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name
    assert actual.grid_sizes.tuple == expected.grid_sizes.tuple


def test_request_reuses_scalar_projections_and_matches_full_pre_infer(device):
    infer, weights, inputs = _setup(device)
    reference, _, _ = _setup(device, False)
    originals = {key: value.clone() for key, value in inputs["text_encoder_output"].items()}
    original_t = infer.scheduler.timesteps.clone()
    previous = {}
    for start in (0, 3, 6, 9):
        for step in range(4):
            _step(infer, step, start)
            reference.scheduler = infer.scheduler
            actual = infer.infer(weights, inputs)
            expected = reference.infer(weights, inputs)
            _assert_equal(actual, expected)
            if start >= 6:
                assert actual.embed is previous[step].embed
                assert actual.embed0 is previous[step].embed0
            elif start == 3:
                previous[step] = actual
            if start == 0 and step == 0:
                first_context = actual.context
                context_id = actual.adapter_args["context_id"]
            assert actual.context is first_context
            assert actual.adapter_args["context_id"] == context_id
        _step(infer, 3, start, True)
        reference.scheduler = infer.scheduler
        actual = infer.infer(weights, inputs)
        _assert_equal(actual, reference.infer(weights, inputs))
    # Reference contributes 20 calls; cache needs five first-chunk projections
    # plus four scalar denoising steps and one scalar cache-commit projection.
    assert weights.time_embedding_0.calls == 30
    assert weights.text_embedding_0.calls == 21
    assert len(infer._time_projection_cache) == 5
    assert torch.equal(infer.scheduler.timesteps, original_t)
    for key, value in inputs["text_encoder_output"].items():
        assert torch.equal(value, originals[key])


def test_cfg_branches_and_prompt_mutations_refresh_context_identity(device):
    infer, weights, inputs = _setup(device)
    first = infer.infer(weights, inputs)
    infer.scheduler.infer_condition = False
    null = infer.infer(weights, inputs)
    assert null.context is not first.context
    infer.scheduler.infer_condition = True
    assert infer.infer(weights, inputs).context is first.context
    source = inputs["text_encoder_output"]["context"]
    source.add_(1)
    mutated = infer.infer(weights, inputs)
    assert mutated.adapter_args["context_id"] != first.adapter_args["context_id"]
    assert not torch.equal(first.context, mutated.context)
    inputs["text_encoder_output"]["context"] = source.clone()
    replaced = infer.infer(weights, inputs)
    assert replaced.adapter_args["context_id"] != mutated.adapter_args["context_id"]
    assert torch.equal(replaced.context, mutated.context)
    assert len(infer._text_projection_cache) == 2
    assert weights.text_embedding_0.calls == 4


def test_new_generator_and_explicit_clear_release_request_state(device):
    infer, weights, inputs = _setup(device)
    first = infer.infer(weights, inputs)
    old_generator = infer.scheduler.generator
    assert infer._projection_request is old_generator
    infer.scheduler.generator = torch.Generator(device=device).manual_seed(123)
    second = infer.infer(weights, inputs)
    _assert_equal(first, second)
    assert second.context is not first.context
    assert infer._projection_request is infer.scheduler.generator
    assert weights.time_embedding_0.calls == weights.text_embedding_0.calls == 2
    infer.clear_projection_cache()
    assert infer._projection_request is None
    assert infer._projection_weight_refs == ()
    assert not infer._time_projection_cache and not infer._text_projection_cache
    assert infer._projection_schedule is None


@pytest.mark.parametrize("mutation", ["replace_weight", "weight_inplace", "bias_inplace", "replace_module"])
def test_weight_reload_or_edit_invalidates_both_caches(device, mutation):
    infer, weights, inputs = _setup(device)
    first = infer.infer(weights, inputs)
    module = weights.text_embedding_0
    if mutation == "replace_weight":
        module.weight = module.weight.clone()
    elif mutation == "weight_inplace":
        module.weight.add_(0.1)
    elif mutation == "bias_inplace":
        module.bias.add_(0.1)
    else:
        _, replacement, _ = _setup(device)
        weights.text_embedding_0 = replacement.text_embedding_0
    second = infer.infer(weights, inputs)
    assert second.embed is not first.embed
    assert second.context is not first.context
    assert second.adapter_args["context_id"] != first.adapter_args["context_id"]
    _assert_equal(second, _reference(infer, weights, inputs))


def test_changed_schedule_invalidates_time_cache_and_scalar_clones_bypass(device):
    infer, weights, inputs = _setup(device)
    first = infer.infer(weights, inputs)
    infer.scheduler.timesteps.add_(1)
    _step(infer, 0)
    changed = infer.infer(weights, inputs)
    assert changed.embed is not first.embed
    assert changed.context is first.context
    assert not torch.equal(changed.embed, first.embed)
    infer.scheduler.timestep_input = infer.scheduler.timestep_input.clone()
    bypassed = infer.infer(weights, inputs)
    assert bypassed.embed is not changed.embed
    assert torch.equal(bypassed.embed, changed.embed)


def test_mutated_returned_projections_are_not_reused(device):
    infer, weights, inputs = _setup(device)
    original = infer.infer(weights, inputs)
    expected_embed = original.embed.clone()
    expected_context = original.context.clone()
    original.embed.add_(3)
    original.embed0.add_(4)
    original.context.add_(5)
    repaired = infer.infer(weights, inputs)
    assert torch.equal(repaired.embed, expected_embed)
    assert torch.equal(repaired.context, expected_context)
    assert repaired.adapter_args["context_id"] != original.adapter_args["context_id"]


@pytest.mark.parametrize(
    "override",
    [
        {"cache_pre_infer_projections": False},
        {"cpu_offload": True},
        {"lazy_load": True},
        {"seq_parallel": True},
        {"tensor_parallel": True},
        {"use_image_encoder": True},
        {"lora_configs": ["adapter"]},
        {"lora_dynamic_apply": True},
    ],
)
def test_config_guards_clear_previous_cache(override):
    infer, weights, inputs = _setup()
    infer.infer(weights, inputs)
    infer.config.update(override)
    assert not infer._prepare_projection_cache(weights, torch.device("cpu"))
    assert infer._projection_request is None
    assert not infer._text_projection_cache and not infer._time_projection_cache


@pytest.mark.parametrize(
    "name,value",
    [
        ("has_lora_branch", True),
        ("has_diff", True),
        ("weight_diff", None),
        ("bias_diff", None),
        ("lazy_load", True),
        ("create_cpu_buffer", True),
        ("create_cuda_buffer", True),
    ],
)
def test_operator_guards_reject_dynamic_or_offloaded_weights(name, value):
    infer, weights, _ = _setup()
    setattr(weights.time_embedding_0, name, value)
    assert not infer._prepare_projection_cache(weights, torch.device("cpu"))


def test_guards_reject_unknown_scheduler_missing_generator_or_nonresident_weights():
    infer, weights, _ = _setup()
    infer.scheduler = SimpleNamespace(generator=torch.Generator())
    assert not infer._prepare_projection_cache(weights, torch.device("cpu"))
    infer, weights, _ = _setup()
    infer.scheduler.generator = None
    assert not infer._prepare_projection_cache(weights, torch.device("cpu"))
    infer, weights, _ = _setup()
    weights.text_embedding_0.weight = torch.empty_like(weights.text_embedding_0.weight, device="meta")
    assert not infer._prepare_projection_cache(weights, torch.device("cpu"))


def test_inference_tensors_without_version_use_ordinary_path():
    infer, weights, inputs = _setup()
    with torch.inference_mode():
        source = inputs["text_encoder_output"]["context"].clone()
    inputs["text_encoder_output"]["context"] = source
    first = infer.infer(weights, inputs)
    second = infer.infer(weights, inputs)
    assert torch.equal(first.context, second.context)
    assert first.context is not second.context
    assert not infer._text_projection_cache
    assert first.adapter_args["context_id"] == id(source)
    with torch.inference_mode():
        weights.time_embedding_0.weight = weights.time_embedding_0.weight.clone()
    assert not infer._prepare_projection_cache(weights, torch.device("cpu"))


def test_timestep_guards_and_cache_path_never_read_cuda_scalars(monkeypatch):
    infer, weights, inputs = _setup()

    def forbidden_item(*args, **kwargs):
        raise AssertionError("Caching must not read tensor scalars")

    monkeypatch.setattr(torch.Tensor, "item", forbidden_item)
    for start in (0, 3, 6):
        for step in range(4):
            _step(infer, step, start)
            infer.infer(weights, inputs)
        _step(infer, 3, start, True)
        infer.infer(weights, inputs)
    infer.scheduler.step_index = -1
    assert infer._time_projection_key(infer.scheduler.timestep_input) is None
    infer.scheduler.step_index = 4
    assert infer._time_projection_key(infer.scheduler.timestep_input) is None
    _step(infer, 0, 3, True)
    infer.scheduler.timestep_input.add_(1)
    assert infer._time_projection_key(infer.scheduler.timestep_input) is None


def test_inference_mode_outputs_without_version_are_not_cached():
    infer, weights, inputs = _setup()
    with torch.inference_mode():
        first = infer.infer(weights, inputs)
        first.context.add_(1)
        second = infer.infer(weights, inputs)
        assert second.context is not first.context
        assert not infer._text_projection_cache and not infer._time_projection_cache


@pytest.mark.parametrize("enabled", [False, True])
def test_proxy_interpolation_runs_once_per_chunk_and_preserves_full_outputs(device, monkeypatch, enabled):
    import lightx2v.models.networks.wan.infer.edenlive.pre_infer as module

    infer, weights, inputs = _setup(device)
    infer.config["cache_proxy_interpolation"] = enabled
    original = module.F.interpolate
    calls = []

    def interpolate(*args, **kwargs):
        calls.append(kwargs["size"])
        return original(*args, **kwargs)

    monkeypatch.setattr(module.F, "interpolate", interpolate)
    weights.patch_embedding.apply = lambda value: value[:, -8:].contiguous()
    for start in (3, 6):
        for step in range(5):
            _step(infer, min(step, 3), start, step == 4)
            actual = infer.infer(weights, inputs)
            expected_control = original(inputs["proxy_features"].to(infer.scheduler.latents_input), size=infer.scheduler.latents_input.shape[1:], mode="trilinear", align_corners=False)[0]
            expected_x = expected_control[-8:].flatten(1).transpose(0, 1).contiguous()
            assert torch.equal(actual.x, expected_x)
    assert len(calls) == (2 if enabled else 10)
    assert (infer._proxy_interpolation_cache is not None) == enabled


def test_proxy_cache_key_tracks_source_edits_output_edits_shape_dtype_and_chunk(device):
    infer, weights, inputs = _setup(device)
    infer.config["cache_proxy_interpolation"] = True
    assert infer._prepare_projection_cache(weights, infer.scheduler.latents_input.device)
    source = inputs["proxy_features"]
    x = infer.scheduler.latents_input
    first = infer._interpolate_proxy(source, x, True)
    assert infer._interpolate_proxy(source, x, True) is first
    source.add_(1)
    changed = infer._interpolate_proxy(source, x, True)
    assert changed is not first
    assert not torch.equal(first, changed)
    original = changed.clone()
    changed.add_(3)
    repaired = infer._interpolate_proxy(source, x, True)
    assert repaired is not changed
    assert torch.equal(repaired, original)
    replaced = infer._interpolate_proxy(source.clone(), x, True)
    assert replaced is not repaired and torch.equal(replaced, repaired)
    new_shape = x.new_zeros(8, 1, 4, 5)
    reshaped = infer._interpolate_proxy(source, new_shape, True)
    assert reshaped.shape == (1024, 1, 4, 5)
    recast = infer._interpolate_proxy(source, new_shape.float(), True)
    assert recast.dtype == torch.float32
    assert infer._proxy_interpolation_cache[1] is source
    infer.scheduler.chunk_start += 3
    next_chunk = infer._interpolate_proxy(source, new_shape.float(), True)
    assert next_chunk is not recast and torch.equal(next_chunk, recast)


@pytest.mark.parametrize("zero", [False, True])
def test_proxy_cache_shares_branches_only_when_control_is_identical(device, zero):
    infer, weights, inputs = _setup(device)
    infer.config.update(cache_proxy_interpolation=True, proxy_unconditional_control="zero" if zero else "same")
    assert infer._prepare_projection_cache(weights, infer.scheduler.latents_input.device)
    source, x = inputs["proxy_features"], infer.scheduler.latents_input
    conditional = infer._interpolate_proxy(source, x, True)
    infer.scheduler.infer_condition = False
    unconditional = infer._interpolate_proxy(source, x, True)
    if zero:
        assert unconditional is not conditional
        assert torch.equal(unconditional, torch.zeros_like(unconditional))
        infer.scheduler.infer_condition = True
        restored = infer._interpolate_proxy(source, x, True)
        assert restored is not unconditional and torch.equal(restored, conditional)
    else:
        assert unconditional is conditional


def test_proxy_cache_requires_projection_guards_and_clears_between_requests(device):
    infer, weights, inputs = _setup(device)
    infer.config["cache_proxy_interpolation"] = True
    infer.infer(weights, inputs)
    first = infer._proxy_interpolation_cache[2]
    infer.scheduler.generator = torch.Generator(device=device).manual_seed(123)
    infer.infer(weights, inputs)
    assert infer._proxy_interpolation_cache[2] is not first
    infer.clear_request_state()
    assert infer._proxy_interpolation_cache is None
    infer.config["cache_pre_infer_projections"] = False
    infer.infer(weights, inputs)
    assert infer._proxy_interpolation_cache is None
    infer.config["cache_pre_infer_projections"] = True
    weights.time_embedding_0.has_diff = True
    infer.infer(weights, inputs)
    assert infer._proxy_interpolation_cache is None


def test_proxy_rms_norm_is_applied_after_cache_and_tracks_its_current_weight(device):
    infer, weights, inputs = _setup(device)
    infer.config.update(cache_proxy_interpolation=True, proxy_condition_rmsnorm=True)
    weights.condition_norm = SimpleNamespace(tensor=torch.ones(1024, device=device, dtype=torch.bfloat16))
    weights.patch_embedding.apply = lambda value: value[:, -8:].contiguous()
    first = infer.infer(weights, inputs)
    cached_control = infer._proxy_interpolation_cache[2]
    weights.condition_norm.tensor.mul_(2)
    second = infer.infer(weights, inputs)
    assert infer._proxy_interpolation_cache[2] is cached_control
    assert not torch.equal(first.x, second.x)
    infer.config["cache_proxy_interpolation"] = False
    reference = infer.infer(weights, inputs)
    _assert_equal(second, reference)


def test_proxy_inference_tensors_or_inference_mode_outputs_are_not_cached():
    infer, weights, inputs = _setup()
    infer.config["cache_proxy_interpolation"] = True
    with torch.inference_mode():
        inputs["proxy_features"] = inputs["proxy_features"].clone()
    infer.infer(weights, inputs)
    assert infer._proxy_interpolation_cache is None
    inputs["proxy_features"] = inputs["proxy_features"].clone()
    with torch.inference_mode():
        infer.infer(weights, inputs)
        assert infer._proxy_interpolation_cache is None

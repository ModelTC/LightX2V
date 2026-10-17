from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from lightx2v.common.kvcache.rolling import RollingKVCachePool
from lightx2v.models.networks.wan.infer.edenlive.cuda_graph import EdenLiveCUDAGraphStack, _copy_tree, _map_tensors, _tree_signature
from lightx2v.models.networks.wan.infer.edenlive.transformer_infer import EdenLiveARTransformerInfer

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda_required
@pytest.mark.parametrize("compiled", [False, True])
def test_graph_stack_preserves_history_and_matches_repeated_raw_kv(compiled):
    graph = EdenLiveCUDAGraphStack(warmup_iterations=3)
    history = [torch.full((8, 4), layer + 1.0, device="cuda") for layer in range(3)]
    reference_history = [value.clone() for value in history]
    cross = [(torch.full((4,), layer + 1.0, device="cuda"),) for layer in range(3)]

    def block(x, embed, text, cache):
        raw_k = x + embed + text
        raw_v = raw_k * 2
        result = raw_k + cache[-x.shape[0] :]
        cache[-x.shape[0] :].copy_(raw_v)
        return result, raw_k, raw_v

    compute = torch.compile(block, fullgraph=True, dynamic=False) if compiled else block

    def make_runner(caches):
        def runner(x, embed, text):
            raw = []
            for cache, kv in zip(caches, text):
                x, k, v = compute(x, embed, kv[0], cache)
                raw.append((k, v))
            return x, raw

        return runner

    runner, reference = make_runner(history), make_runner(reference_history)
    for step in range(4):
        x = torch.full((2, 4), step + 1.0, device="cuda")
        embed = torch.full((4,), step + 2.0, device="cuda")
        original_x = x.clone()
        expected = reference(x.clone(), embed, cross)
        actual = graph.run("steady", runner, x, embed, cross_kv=cross, bindings=history, mutated_buffers=[cache[-2:] for cache in history])
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        for (actual_k, actual_v), (expected_k, expected_v) in zip(actual[1], expected[1]):
            torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
            torch.testing.assert_close(actual_v, expected_v, rtol=0, atol=0)
        for actual_cache, expected_cache in zip(history, reference_history):
            torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)
        torch.testing.assert_close(x, original_x, rtol=0, atol=0)
    assert graph.capture_count == 1
    assert graph.replay_count == 4
    assert graph.cross_refresh_count == 1


@cuda_required
def test_cross_kv_refreshes_contents_without_recapture_or_history_duplicate():
    graph = EdenLiveCUDAGraphStack()
    history = torch.arange(24, device="cuda", dtype=torch.float32).view(6, 4)
    original_history_ptr = history.data_ptr()
    x = torch.ones(2, 4, device="cuda")
    embed = torch.ones(4, device="cuda")
    cross = {"layer": (torch.ones(4, device="cuda"), torch.ones(4, device="cuda"))}

    def runner(x, embed, text):
        return x + embed + text["layer"][0] + text["layer"][1] + history[:2]

    for version in range(3):
        # A refresh must work even when the source tensors keep their pointers.
        cross["layer"][0].fill_(version + 1)
        cross["layer"][1].fill_(version + 2)
        actual = graph.run("steady", runner, x, embed, cross_kv=cross, cross_version=version, bindings=(history,))
        torch.testing.assert_close(actual, runner(x, embed, cross), rtol=0, atol=0)
        assert graph._bindings[0] is history
        assert graph._bindings[0].data_ptr() == original_history_ptr
    assert graph.capture_count == 1
    assert graph.cross_refresh_count == 3


@cuda_required
def test_cross_pointer_refresh_and_binding_or_shape_change_recapture():
    graph = EdenLiveCUDAGraphStack()
    history = torch.ones(8, 4, device="cuda")

    def make_runner(cache):
        return lambda x, embed, cross: x + embed + cross[0] + cache[: x.shape[0]]

    embed = torch.ones(4, device="cuda")
    for step, rows in enumerate([2, 2, 2, 3]):
        if step == 2:
            history = torch.full_like(history, 5)
        cross = (torch.full((4,), step + 1.0, device="cuda"),)
        runner = make_runner(history)
        x = torch.ones(rows, 4, device="cuda")
        actual = graph.run("steady", runner, x, embed, cross_kv=cross, bindings=(history,))
        torch.testing.assert_close(actual, runner(x, embed, cross), rtol=0, atol=0)
    assert graph.capture_count == 3
    assert graph.replay_count == 4
    assert graph.cross_refresh_count == 4


@cuda_required
def test_capture_failure_restores_external_history_and_can_retry():
    graph = EdenLiveCUDAGraphStack()
    x = torch.ones(2, 4, device="cuda")
    embed = torch.ones(4, device="cuda")
    history = torch.ones_like(x)
    original = history.clone()

    def failing(x, embed, cross):
        history.add_(10)
        raise RuntimeError("test capture failure")

    with pytest.raises(RuntimeError, match="test capture failure"):
        graph.run("steady", failing, x, embed, bindings=(history,), mutated_buffers=(history,))
    torch.testing.assert_close(history, original, rtol=0, atol=0)
    assert graph.graph is None

    def runner(x, embed, cross):
        history.add_(1)
        return x + embed + history

    output = graph.run("steady", runner, x, embed, bindings=(history,), mutated_buffers=(history,))
    torch.testing.assert_close(history, original + 1, rtol=0, atol=0)
    torch.testing.assert_close(output, x + embed + original + 1, rtol=0, atol=0)


@cuda_required
def test_runtime_key_changes_and_clear_release_graph_storage():
    graph = EdenLiveCUDAGraphStack()
    x = torch.ones(2, 4, device="cuda")
    embed = torch.ones(4, device="cuda")

    def runner(x, embed, cross):
        return x + embed

    for key in ["fp8", "packed_fp8"]:
        torch.testing.assert_close(graph.run(key, runner, x, embed), x + embed, rtol=0, atol=0)
    assert graph.capture_count == 2
    graph.clear()
    assert graph.graph is None
    assert graph.outputs is None
    assert graph._bindings is None
    assert graph.static_x is None
    assert graph.static_cross_kv is None


@cuda_required
def test_rejects_cpu_or_mixed_device_inputs():
    graph = EdenLiveCUDAGraphStack()
    with pytest.raises(ValueError, match="same CUDA device"):
        graph.run("cpu", None, torch.ones(2, 4), torch.ones(4))
    x = torch.ones(2, 4, device="cuda")
    with pytest.raises(ValueError, match="same CUDA device"):
        graph.run("mixed", None, x, x[0], cross_kv=(torch.ones(4),))


def test_rejects_empty_warmup():
    with pytest.raises(ValueError, match="at least one warmup"):
        EdenLiveCUDAGraphStack(warmup_iterations=0)


@dataclass
class _Pre:
    x: torch.Tensor
    embed0: torch.Tensor
    grid_sizes: object
    adapter_args: dict


class _CPURecordingGraph:
    """Exercise stack integration while recording the real binding signature."""

    def __init__(self):
        self.capture_count = 0
        self.replay_count = 0
        self.cross_refresh_count = 0
        self.signature = None
        self.cross_version = None

    def clear(self):
        self.signature = None

    def run(self, key, runner, x, embed0, *, cross_kv, cross_version, bindings, mutated_buffers):
        signature = (key, _tree_signature(bindings, pointers=True), _tree_signature(mutated_buffers, pointers=True))
        if signature != self.signature:
            self.capture_count += 1
            self.signature = signature
            self.cross = _map_tensors(cross_kv, lambda tensor: tensor.clone())
            self.cross_refresh_count += 1
            self.cross_version = cross_version
        elif cross_version != self.cross_version:
            _copy_tree(self.cross, cross_kv)
            self.cross_refresh_count += 1
            self.cross_version = cross_version
        self.replay_count += 1
        return runner(x.clone(), embed0.clone(), self.cross)


def _cpu_infer(graph):
    infer = EdenLiveARTransformerInfer.__new__(EdenLiveARTransformerInfer)
    infer.config = {
        "ar_config": {"local_attn_size": 18, "num_frame_per_chunk": 3, "sink_size": 6},
        "size": [2, 2],
        "vae_stride": [1, 1, 1],
        "patch_size": [1, 2, 2],
        "dit_quant_scheme": "fp8-sgl",
        "self_attn_1_type": "sage_attn2_sm120_cached",
    }
    infer.cuda_graph_stack = _CPURecordingGraph() if graph else None
    infer._graph_weight_key = None
    infer._graph_cross_generation = 0
    infer.history_cache = {}
    infer.freqs_cache = {}
    infer.cross_kv_cache = {}
    infer.cache_pools = {}
    infer.reuse_history = True
    infer.reuse_quantized_cross_kv = True
    infer.reuse_cross_kv = True
    infer.transposed_v_cache = True
    infer.use_compile = True
    infer.compile_backend = "default"
    infer.seq_parallel = False
    infer.has_image_context = False
    infer.num_heads = 2
    infer.head_dim = 4
    infer.current_pre = None
    infer.scheduler = SimpleNamespace(chunk_start=15, infer_condition=True, cache_update=False)
    return infer


@pytest.mark.parametrize(
    "override",
    [
        {"cuda_graph_stack": None},
        {"use_compile": False},
        {"compile_backend": "custom"},
        {"reuse_history": False},
        {"reuse_quantized_cross_kv": False},
        {"seq_parallel": True},
        {"has_image_context": True},
        {"config": {"tensor_parallel": True}},
        {"config": {"dit_quant_scheme": "bf16"}},
        {"config": {"lazy_load": True}},
        {"config": {"lora_configs": ["adapter"]}},
        {"config": {"lora_dynamic_apply": True}},
        {"chunk_start": 12},
        {"frames": 1},
        {"grid": (3, 2, 1)},
        {"text_segments": [3]},
        {"is_cuda": False},
    ],
)
def test_transformer_graph_guards_reject_unsupported_paths(override):
    infer = _cpu_infer(True)
    pre = SimpleNamespace(x=SimpleNamespace(is_cuda=True), grid_sizes=SimpleNamespace(tuple=(3, 1, 1)), adapter_args={"text_segments": None})
    assert infer._can_graph_stack(pre)
    for name, value in override.items():
        if name == "config":
            infer.config.update(value)
        elif name == "chunk_start":
            infer.scheduler.chunk_start = value
        elif name == "frames":
            pre.grid_sizes.tuple = (value, 1, 1)
        elif name == "grid":
            pre.grid_sizes.tuple = value
        elif name == "text_segments":
            pre.adapter_args[name] = value
        elif name == "is_cuda":
            pre.x.is_cuda = value
        else:
            setattr(infer, name, value)
    assert not infer._can_graph_stack(pre)


def test_transformer_graph_accepts_standard_history_layout_for_sla():
    infer = _cpu_infer(True)
    infer.transposed_v_cache = False
    pre = SimpleNamespace(x=SimpleNamespace(is_cuda=True), grid_sizes=SimpleNamespace(tuple=(3, 1, 1)), adapter_args={"text_segments": None})
    assert infer._can_graph_stack(pre)


def test_transformer_graph_accepts_resident_int8_convrot():
    infer = _cpu_infer(True)
    infer.config["dit_quant_scheme"] = "int8-convrot"
    pre = SimpleNamespace(x=SimpleNamespace(is_cuda=True), grid_sizes=SimpleNamespace(tuple=(3, 1, 1)), adapter_args={"text_segments": None})
    assert infer._can_graph_stack(pre)


def test_graph_clear_preserves_storage_resets_cursors_and_invalidates_explicitly():
    infer = _cpu_infer(True)
    key_storage = torch.ones(18, 2, 4)
    value_storage = torch.ones(1, 4, 2, 128)
    freqs = torch.ones(18, 1, 2)
    infer.history_cache = {(True, 0): (42, key_storage, value_storage)}
    infer.freqs_cache = {(18, 1, 1, torch.device("cpu")): freqs}
    infer.cross_kv_cache = {True: (1, [torch.ones(4)])}
    infer.cache_pools = {True: object()}
    infer.current_pre = object()
    infer.key_freqs = infer.query_freqs = freqs
    graph = infer.cuda_graph_stack
    graph.signature = "captured"
    infer._graph_weight_key = (7,)
    infer._graph_cross_generation = 9
    infer.clear_request_state()
    assert infer.history_cache[True, 0][0] == -1
    assert infer.history_cache[True, 0][1] is key_storage
    assert infer.history_cache[True, 0][2] is value_storage
    assert next(iter(infer.freqs_cache.values())) is freqs
    assert infer.cache_pools == {} and infer.cross_kv_cache == {}
    assert infer.current_pre is None and infer.key_freqs is None and infer.query_freqs is None
    assert graph.signature == "captured"
    assert infer._graph_cross_generation == 9
    infer.invalidate_cuda_graph()
    assert graph.signature is None and infer._graph_weight_key is None


def test_disabled_graph_clear_releases_history_and_supports_legacy_infer():
    infer = _cpu_infer(False)
    infer.history_cache = {(True, 0): (1, torch.ones(4), torch.ones(4))}
    infer.freqs_cache = {1: torch.ones(4)}
    del infer.cuda_graph_stack
    infer.clear_request_state()
    assert infer.history_cache == {} and infer.freqs_cache == {}


@pytest.mark.parametrize("transposed", [False, True])
def test_graph_integration_matches_window_roll_across_requests_and_one_frame_tail(monkeypatch, transposed):
    from lightx2v.models.networks.wan.infer.edenlive import transformer_infer as module

    def permute_indices(length, start=0):
        position = torch.arange(start, start + length)
        return (position & ~15) | (position & 1) | ((position & 8) >> 2) | ((position & 2) << 1) | ((position & 4) << 1)

    def update_v(v, output, start):
        output[..., permute_indices(v.shape[1], start)] = v.permute(0, 3, 2, 1)

    def make_v(v, capacity):
        output = v.new_zeros(v.shape[0], v.shape[-1], v.shape[-2], (capacity + 127) // 128 * 128)
        update_v(v, output, 0)
        return output

    monkeypatch.setattr(module, "make_transposed_v", make_v)
    monkeypatch.setattr(module, "update_transposed_v", update_v)

    class Rope:
        @staticmethod
        def apply_single(x, freqs):
            return x + freqs

    class Block:
        compute_phases = (SimpleNamespace(rope=Rope()),)

        @staticmethod
        def named_weight_leaves():
            return []

    blocks = [Block(), Block()]
    reference, graphed = _cpu_infer(False), _cpu_infer(True)
    for infer in (reference, graphed):
        infer.transposed_v_cache = transposed
        if not transposed:
            infer.config["self_attn_1_type"] = "general_sparse_attn"
        # Exercise the production stack branch on CPU; GPU support is verified
        # independently by the guard tests and real CUDA helper tests.
        infer._can_graph_stack = lambda pre, infer=infer: infer.cuda_graph_stack is not None and pre.grid_sizes.tuple[0] == 3 and infer._visible_frames(3) == 18

        def make_compute(infer, layer):
            def compute(x, pre, cross, cache):
                keys, values = cache
                raw_k = x.view(-1, 2, 4).clone()
                raw_v = raw_k + layer + 1
                keys[-x.shape[0] :].copy_(raw_k + infer.query_freqs)
                if infer.transposed_v_cache:
                    update_v(raw_v.unsqueeze(0), values, keys.shape[0] - x.shape[0])
                    logical_v = values[..., permute_indices(keys.shape[0])].permute(0, 3, 2, 1).squeeze(0)
                else:
                    values[-x.shape[0] :].copy_(raw_v)
                    logical_v = values
                output = (keys.mean(0) + logical_v.mean(0)).flatten() + cross[0] + pre.embed0.flatten()[0]
                x.add_(output)
                return x, raw_k, raw_v

            return compute

        functions = [make_compute(infer, index) for index in range(len(blocks))]
        infer.get_compiled_block = lambda index, block, functions=functions: functions[index]

    history_pointers = None
    freq_pointer = None
    for request in range(2):
        for infer in (reference, graphed):
            pool = RollingKVCachePool(2, 18, 2, 4, torch.float32, torch.device("cpu"))
            pool._init_kv_buffer()
            infer.cache_pools[True] = pool
            # A request may reuse the same context-id integer. The monotonically
            # increasing refresh generation must still stage its new text KV.
            infer.cross_kv_cache[True] = ((1, True), [(torch.tensor(float(request + layer + 1)),) for layer in range(2)])
            infer._graph_cross_generation += 1
        start = 0
        for frames in [3] * 11 + [1]:
            for commit in [False, False, True]:
                x = torch.arange(frames * 8, dtype=torch.float32).view(frames, 8) / 32 + request
                outputs = []
                for infer in (reference, graphed):
                    infer.scheduler = SimpleNamespace(chunk_start=start, infer_condition=True, cache_update=commit)
                    visible = infer._visible_frames(frames)
                    freq_key = (visible, 1, 1, torch.device("cpu"))
                    if freq_key not in infer.freqs_cache:
                        infer.freqs_cache[freq_key] = torch.arange(visible, dtype=torch.float32).view(-1, 1, 1) / 100
                    infer.key_freqs = infer.freqs_cache[freq_key]
                    infer.query_freqs = infer.key_freqs[-frames:]
                    pre = _Pre(x.clone(), torch.tensor([0.25]), SimpleNamespace(tuple=(frames, 1, 1)), {"text_segments": None})
                    infer.current_pre = pre
                    outputs.append(infer.infer_without_offload(blocks, pre.x, pre))
                    assert infer.current_pre is pre
                torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
                if commit:
                    for layer in range(2):
                        a, b = reference.cache_pools[True], graphed.cache_pools[True]
                        end = a.get_local_end(layer)
                        assert end == b.get_local_end(layer)
                        torch.testing.assert_close(a.k_cache(layer, 0, end), b.k_cache(layer, 0, end), rtol=0, atol=0)
                        torch.testing.assert_close(a.v_cache(layer, 0, end), b.v_cache(layer, 0, end), rtol=0, atol=0)
            start += frames
        pointers = tuple((keys.data_ptr(), values.data_ptr()) for _, keys, values in graphed.history_cache.values())
        full_freqs = graphed.freqs_cache[18, 1, 1, torch.device("cpu")]
        if request == 0:
            history_pointers, freq_pointer = pointers, full_freqs.data_ptr()
        else:
            assert pointers == history_pointers
            assert full_freqs.data_ptr() == freq_pointer
        reference.clear_request_state()
        graphed.clear_request_state()
    assert graphed.cuda_graph_stack.capture_count == 1
    assert graphed.cuda_graph_stack.replay_count == 2 * 6 * 3
    assert graphed.cuda_graph_stack.cross_refresh_count == 2

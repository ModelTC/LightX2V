"""Frame-slot cache compatibility with the shared KV cache factory."""

import pytest
import torch

import lightx2v  # noqa: F401 - initialize the selected accelerator platform
from lightx2v.common.kvcache.manager import KVCacheManager


@pytest.mark.parametrize("num_heads", [1, 2])
def test_frame_ring_factory_preserves_physical_slots_and_head_sharding(num_heads):
    manager = KVCacheManager({"num_layers": 2, "num_heads": 2, "dim": 8}, device=torch.device("cpu"))
    cache = manager.create_self_attn_kv_cache("wan", 12, kv_cache_scheme="frame_ring", dtype=torch.float32, num_heads=num_heads)
    assert manager.get_self_attn_kv_cache("wan") is cache
    buffers = [(cache.k_cache(layer), cache.v_cache(layer)) for layer in range(2)]
    pointers = [(key.data_ptr(), value.data_ptr()) for key, value in buffers]
    expected = [{}, {}]
    for frame in range(9):
        for layer in range(2):
            key = torch.full((3, num_heads, 4), frame * 10.0 + layer)
            value = -key - 1
            actual_key, actual_value = cache.update(key, value, layer, frame)
            expected[layer][frame % 4] = (key, value)
            slots = [expected[layer][slot] for slot in range(min(frame + 1, 4))]
            torch.testing.assert_close(actual_key, torch.cat([pair[0] for pair in slots]), rtol=0, atol=0)
            torch.testing.assert_close(actual_value, torch.cat([pair[1] for pair in slots]), rtol=0, atol=0)
            assert (cache.k_cache(layer).data_ptr(), cache.v_cache(layer).data_ptr()) == pointers[layer]
    with pytest.raises(ValueError, match="multiple of the frame token count"):
        cache.update(torch.empty(5, num_heads, 4), torch.empty(5, num_heads, 4), 0, 9)

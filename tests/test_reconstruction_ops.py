import pytest
import torch

import lightx2v.common.ops  # noqa: F401
from lightx2v.common.ops.inference_module import InferenceAttention
from lightx2v.common.ops.sparse import SparseConv3d, SparseTensor, VarLenTensor
from lightx2v.common.ops.sparse.attention.packed import PackedAttention
from lightx2v.common.ops.sparse.neighbor_map import cpu_hash_neighbors, sorted_neighbors
from lightx2v.models.schedulers.trellis2.flow_euler import FlowEulerSampler
from lightx2v.models.video_encoders.hf.trellis2.mesh_extraction import flexible_dual_grid_to_mesh


@pytest.mark.parametrize("dilation", [(1, 1, 1), (2, 1, 3)])
@pytest.mark.parametrize("kernel", [(1, 1, 1), (3, 3, 3), (3, 1, 3)])
def test_neighbor_lookup_matches_hash(kernel, dilation):
    generator = torch.Generator().manual_seed(21)
    coords = torch.cartesian_prod(torch.arange(2), torch.arange(4), torch.arange(5), torch.arange(3))
    coords = coords[torch.randperm(len(coords), generator=generator)[:61]]
    reference = cpu_hash_neighbors(coords, kernel, dilation)
    actual = sorted_neighbors(coords, kernel, dilation)
    for (offset_a, indices_a), (offset_b, indices_b) in zip(reference, actual):
        assert offset_a == offset_b
        torch.testing.assert_close(indices_a, indices_b, rtol=0, atol=0)


def test_empty_neighbor_map():
    coords = torch.empty(0, 4, dtype=torch.int32)
    maps = sorted_neighbors(coords, (3, 3, 3), (1, 1, 1))
    assert len(maps) == 27
    assert all(indices.numel() == 0 for _, indices in maps)


def test_submanifold_convolution_matches_dense():
    torch.manual_seed(5)
    coords = torch.cartesian_prod(torch.arange(2), torch.arange(3), torch.arange(4), torch.arange(3)).int()
    values = torch.randn(len(coords), 4)
    layer = SparseConv3d(4, 6, 3, neighbor_map=sorted_neighbors)
    sparse = SparseTensor(values, coords)
    dense = values.reshape(2, 3, 4, 3, 4).permute(0, 4, 1, 2, 3)
    expected = torch.nn.functional.conv3d(dense, layer.weight.permute(0, 4, 1, 2, 3), layer.bias, padding=1)
    actual = layer(sparse).feats.reshape(2, 3, 4, 3, 6).permute(0, 4, 1, 2, 3)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("batch", [1, 2])
def test_attention_layout_and_cross_attention(batch):
    torch.manual_seed(3)
    q = torch.randn(batch, 3, 7, 8, dtype=torch.bfloat16)
    k = torch.randn(batch, 3, 11, 8, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    actual = InferenceAttention("torch_sdpa")(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_ragged_attention_does_not_mix_samples():
    torch.manual_seed(11)
    features = torch.randn(8, 3, 2, 8, dtype=torch.bfloat16)
    layout = [slice(0, 3), slice(3, 8)]
    packed = VarLenTensor(features, layout)
    actual = PackedAttention()(packed)
    for section in layout:
        q, k, v = features[section].unbind(1)
        expected = torch.nn.functional.scaled_dot_product_attention(*(x.transpose(0, 1).unsqueeze(0) for x in (q, k, v)))
        torch.testing.assert_close(actual.feats[section], expected.squeeze(0).transpose(0, 1), rtol=0, atol=0)


def test_flow_euler_guidance_interval():
    calls = []

    def model(x, timestep, cond):
        calls.append(cond)
        return torch.full_like(x, cond)

    sampler = FlowEulerSampler()
    x = torch.zeros(1, 2)
    common = dict(cond=2, neg_cond=-1, guidance_strength=3, guidance_interval=(0.6, 1.0))
    guided = sampler.sample_once(model, x, 0.8, 0.7, **common)
    torch.testing.assert_close(guided, torch.full_like(x, -0.8))
    assert calls == [2, -1]
    calls.clear()
    unguided = sampler.sample_once(model, x, 0.5, 0.4, **common)
    torch.testing.assert_close(unguided, torch.full_like(x, -0.2))
    assert calls == [2]


def test_dual_grid_mesh_extracts_one_quad():
    coords = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]])
    flags = torch.zeros(4, 3, dtype=torch.bool)
    flags[0, 0] = True
    vertices, faces = flexible_dual_grid_to_mesh(coords, torch.full((4, 3), 0.5), flags, torch.ones(4, 1), aabb=[[-0.5] * 3, [0.5] * 3], grid_size=2)
    torch.testing.assert_close(vertices, (coords.float() + 0.5) / 2 - 0.5)
    torch.testing.assert_close(faces, torch.tensor([[0, 1, 3], [3, 1, 2]], dtype=torch.int32))


def test_dual_grid_mesh_rejects_linear_key_aliases():
    coords = torch.tensor([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0]])
    flags = torch.zeros(4, 3, dtype=torch.bool)
    flags[0, 0] = True
    with pytest.raises(ValueError, match="connected dual-grid quads"):
        flexible_dual_grid_to_mesh(coords, torch.full((4, 3), 0.5), flags, torch.ones(4, 1), aabb=[[-0.5] * 3, [0.5] * 3], grid_size=2)

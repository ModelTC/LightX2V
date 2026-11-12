"""Teacher-positive pairing, official-grid rollout and resume checks."""
# ruff: noqa: E402

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from common import atomic_json, atomic_jsonl, fingerprint
from prepare_cache import cache_record, make_cache
from prepare_gan_real_cache import generate_teacher_latent, read_conditions, real_record, teacher_recipe, validate_latent


def condition_fixture(tmp_path):
    row = cache_record({"id": 12, "prompt": "PE-expanded caption"}, "text-recipe", 32, 32)
    source = tmp_path / "pe-cache"
    (source / "conditions").mkdir(parents=True)
    condition = {"prompt_embeds": torch.ones(1, 3, 5120, dtype=torch.bfloat16), "text_token_tags": torch.ones(3, dtype=torch.long)}
    torch.save(make_cache(row, condition), source / row["condition_path"])
    manifest = source / "metadata.jsonl"
    atomic_jsonl(manifest, [row])
    return row, manifest, condition


def test_real_record_retains_matching_pe_condition_and_portable_paths(tmp_path):
    row, manifest, _ = condition_fixture(tmp_path)
    records = read_conditions(manifest, 1)
    output = tmp_path / "gan-cache"
    record = real_record(records[0], manifest, output, "teacher-recipe", 42)
    assert record["prompt"] == "PE-expanded caption"
    assert record["cache_fingerprint"] == row["cache_fingerprint"]
    assert (output / record["condition_path"]).resolve() == (manifest.parent / row["condition_path"]).resolve()
    assert record["vae_normalized"] and record["source"] == "teacher_generated"
    assert record["teacher_steps"] == 6
    assert 0 <= record["teacher_seed"] < 2**63
    assert record == real_record(row, manifest, output, "teacher-recipe", 42)
    changed_seed = real_record(row, manifest, output, "different-recipe", 43)
    assert changed_seed["teacher_seed"] != record["teacher_seed"]
    assert changed_seed["real_cache_fingerprint"] != record["real_cache_fingerprint"]


def test_teacher_rollout_uses_six_unshifted_sigmas_fp32_and_paired_condition(tmp_path):
    row, manifest, condition = condition_fixture(tmp_path)
    record = real_record(row, manifest, tmp_path / "gan", "teacher-recipe", 42)
    sigmas = torch.tensor([1.0, 0.9675237, 0.8875485, 0.75, 0.5327739, 0.2320089, 0.0])

    class Teacher:
        device = torch.device("cpu")
        image_head_mode = "released"
        pdd_schedule = SimpleNamespace(physical_sigmas=lambda device: sigmas.to(device))

        def __init__(self):
            self.queries = []

        def prepare_text_condition(self, value):
            assert value is condition
            return value

        def predict_denoiser_output(self, state, sigma, value):
            assert not torch.is_grad_enabled()
            assert value is condition and state.dtype == torch.float32
            self.queries.append(sigma.clone())
            return torch.ones_like(state) * (len(self.queries) + 0.25)

    teacher = Teacher()
    result = generate_teacher_latent(teacher, condition, record)
    expected = torch.randn((1, *record["latent_shape"]), generator=torch.Generator().manual_seed(record["teacher_seed"]))
    for index in range(6):
        expected = expected + (sigmas[index + 1] - sigmas[index]) * (index + 1.25)
    torch.testing.assert_close(result, expected.squeeze(0), rtol=0, atol=0)
    torch.testing.assert_close(torch.stack(teacher.queries), sigmas[:-1], rtol=0, atol=0)
    teacher.image_head_mode = "base"
    with pytest.raises(ValueError, match="released six-head"):
        generate_teacher_latent(teacher, condition, record)


def test_positive_resume_rejects_wrong_identity_dtype_shape_and_nonfinite(tmp_path):
    row, manifest, _ = condition_fixture(tmp_path)
    record = real_record(row, manifest, tmp_path / "gan", "teacher-recipe", 42)
    payload = {"latents": torch.zeros(record["latent_shape"]), "real_sample_key": record["real_sample_key"], "real_cache_fingerprint": record["real_cache_fingerprint"]}
    validate_latent(payload, record)
    for replacement in (
        {"real_sample_key": "another-prompt"},
        {"real_cache_fingerprint": "another-teacher"},
        {"latents": payload["latents"].bfloat16()},
        {"latents": payload["latents"].unsqueeze(0)},
        {"latents": torch.full(record["latent_shape"], float("nan"))},
    ):
        with pytest.raises(ValueError, match="Invalid or stale"):
            validate_latent(dict(payload, **replacement), record)


def test_recipe_tracks_checkpoint_schedule_condition_and_seed(tmp_path):
    model = tmp_path / "model"
    (model / "transformer_diffusers").mkdir(parents=True)
    atomic_json(model / "transformer_diffusers/config.json", {"num_pdd_heads": 6})
    atomic_json(model / "transformer_diffusers/diffusion_pytorch_model.safetensors.index.json", {"weight_map": {"a": "one.safetensors"}})
    atomic_json(model / "pdd_config.json", {"num_inference_steps": 6, "sigmas": [1, 0.9, 0.8, 0.6, 0.4, 0.2, 0]})
    row, _, _ = condition_fixture(tmp_path)
    recipe = teacher_recipe(model, [row], 42, "torch_sdpa")
    assert recipe["condition_manifest_sha256"] == fingerprint([row])
    assert recipe["additional_shift"] is None and recipe["guidance_scale"] == 1
    assert recipe != teacher_recipe(model, [row], 43, "torch_sdpa")
    atomic_json(model / "transformer_diffusers/diffusion_pytorch_model.safetensors.index.json", {"weight_map": {"a": "two.safetensors"}})
    assert recipe != teacher_recipe(model, [row], 42, "torch_sdpa")

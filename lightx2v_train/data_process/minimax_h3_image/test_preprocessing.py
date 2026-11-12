"""CPU checks for cache contracts and PE's released NVFP4 representation."""
# ruff: noqa: E402

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))

from common import atomic_jsonl, establish_recipe, read_prompts
from expand_prompts import dequantize_nvfp4, messages_for, parse_answer
from lightx2v_train.data.minimax_h3_image_cache_dataset import MiniMaxH3ImageCacheDataset, _single_sample_collate
from prepare_cache import cache_record, encode_prompt, make_cache, validate_cache
from select_prompts import select_records


def test_stream_selection_deduplicates_and_stops_without_reading_rest():
    def stream():
        yield {"txt": b"  A red cat  ", "json": b'{"width":1024}', "__key__": "one"}
        yield {"txt": "a  RED cat"}
        yield {"txt": " "}
        yield {"txt": "A blue dog", "__key__": "two"}
        raise AssertionError("The stream should stop after the requested number of unique captions")

    rows = select_records(stream(), 2)
    assert [row["id"] for row in rows] == [0, 1]
    assert rows[0]["source_metadata"] == {"width": 1024}
    assert rows[1]["prompt"] == "A blue dog"


def test_nvfp4_decoding_uses_low_nibble_first_and_block_scales():
    codes = torch.arange(16, dtype=torch.uint8).repeat(2)
    packed = (codes[::2] | (codes[1::2] << 4)).reshape(1, 16)
    scales = torch.tensor([[2.0, 3.0]], dtype=torch.float8_e4m3fn)
    result = dequantize_nvfp4(packed, scales, torch.tensor(0.25), device="cpu", dtype=torch.float32)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    assert torch.equal(result, torch.cat((levels * 0.5, levels * 0.75)).reshape(1, 32))


def test_pe_prompt_and_json_preserve_official_contract():
    recipe = {"system": "system", "task": "Expand: {}", "parameters": {"response_format": {"json_schema": {"schema": {"properties": {"aspect_ratio": {"enum": ["1:1", "16:9"]}}}}}}}
    assert messages_for("cat", recipe)[1]["content"] == "Expand: cat"
    assert parse_answer('thinking</think>{"expanded_caption":"A cat", "aspect_ratio":"16:9"}', recipe)["expanded_caption"] == "A cat"
    with pytest.raises(ValueError):
        parse_answer('{"expanded_caption":"", "aspect_ratio":"1:1"}', recipe)
    with pytest.raises(ValueError):
        parse_answer('{"expanded_caption":"cat", "aspect_ratio":"8:1"}', recipe)


def test_condition_uses_verbatim_tokens_layer50_and_4096_limit():
    class Tokenizer:
        def __call__(self, prompt, add_special_tokens):
            assert prompt == "no chat template" and add_special_tokens is False
            return {"input_ids": list(range(5000))}

    class Processor:
        def create_mm_token_type_ids(self, ids):
            assert len(ids[0]) == 4096
            return [[0] * len(ids[0])]

    class Encoder:
        def model(self, **kwargs):
            assert kwargs["input_ids"].shape == (1, 4096)
            assert kwargs["mm_token_type_ids"].shape == (1, 4096)
            assert not kwargs["use_cache"] and kwargs["output_hidden_states"]
            hidden = torch.ones(1, 4096, 5120, dtype=torch.bfloat16)
            return SimpleNamespace(hidden_states=[None] * 50 + [hidden, hidden * 2])

    condition = encode_prompt("no chat template", Encoder(), Tokenizer(), Processor(), "cpu")
    assert condition["prompt_embeds"].shape == (1, 4096, 5120)
    assert bool((condition["prompt_embeds"] == 1).all())
    assert condition["text_token_tags"].shape == (4096,)


def test_cache_portability_loader_shape_and_stale_rejection(tmp_path):
    prompt = {"id": 7, "prompt": "A cat"}
    record = cache_record(prompt, "recipe", 2048, 2048)
    condition = {"prompt_embeds": torch.ones(1, 3, 5120, dtype=torch.bfloat16), "text_token_tags": torch.ones(3, dtype=torch.long)}
    path = tmp_path / record["condition_path"]
    path.parent.mkdir()
    torch.save(make_cache(record, condition), path)
    validate_cache(path, record)
    atomic_jsonl(tmp_path / "metadata.jsonl", [record])
    dataset = MiniMaxH3ImageCacheDataset(tmp_path)
    assert len(dataset) == 1
    assert json.loads(json.dumps(dataset.samples))[0]["latent_shape"] == [24, 1, 128, 128]
    sample = _single_sample_collate([dataset[0]])
    assert sample["conditioning"]["positive"]["prompt_embeds"].shape == (1, 3, 5120)
    assert sample["generation_shape"] == (2048, 2048)
    assert sample["inputs"] == {}
    dataset.samples[0]["cache_fingerprint"] = "wrong"
    with pytest.raises(ValueError, match="disagree"):
        dataset[0]


def test_recipe_and_count_prevent_stale_resume(tmp_path):
    recipe_path = tmp_path / "recipe.json"
    establish_recipe(recipe_path, {"count": 10000})
    establish_recipe(recipe_path, {"count": 10000})
    with pytest.raises(ValueError, match="differs"):
        establish_recipe(recipe_path, {"count": 5000})
    prompts = tmp_path / "prompts.jsonl"
    atomic_jsonl(prompts, [{"id": 0, "prompt": "A cat"}])
    assert read_prompts(prompts, 1)[0]["id"] == 0
    with pytest.raises(ValueError, match="Expected 2"):
        read_prompts(prompts, 2)

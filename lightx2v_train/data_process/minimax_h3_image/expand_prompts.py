"""Expand T2I prompts with the released MiniMax-H3-Image-PE checkpoint."""

import argparse
import json
import os
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen

from common import atomic_json, atomic_jsonl, establish_recipe, fingerprint, read_prompts


def load_prompt_recipe(model_path):
    folder = Path(model_path) / "prompt" / "t2i"
    return {
        "system": (folder / "system_prompt.txt").read_text().strip(),
        "task": (folder / "task_prompt.txt").read_text(),
        "parameters": json.loads((folder / "request_params.json").read_text()),
    }


def messages_for(prompt, recipe):
    return [{"role": "system", "content": recipe["system"]}, {"role": "user", "content": recipe["task"].format(prompt)}]


def parse_answer(content, recipe):
    answer = json.loads(content.split("</think>")[-1].strip())
    schema = recipe["parameters"]["response_format"]["json_schema"]["schema"]
    ratios = schema["properties"]["aspect_ratio"]["enum"]
    if set(answer) != {"expanded_caption", "aspect_ratio"} or not isinstance(answer["expanded_caption"], str) or not answer["expanded_caption"].strip() or answer["aspect_ratio"] not in ratios:
        raise ValueError(f"PE response does not match the released schema: {answer}")
    return answer


def dequantize_nvfp4(packed, scale, global_scale, *, device, dtype):
    """ModelOpt NVFP4: low nibble first, E2M1 * E4M3 block scale * FP32 scale.

    Matches NVIDIA Model-Optimizer's NVFP4QTensor.dequantize (block size 16):
    modelopt/torch/quantization/qtensor/nvfp4_tensor.py.
    Activation quantization is omitted: this backend runs the recovered weights
    with BF16 activations, rather than emulating native NVFP4 GEMMs.
    """
    import torch

    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("Expected a matrix of packed NVFP4 bytes")
    rows, packed_columns = packed.shape
    columns = packed_columns * 2
    if columns % 16 or tuple(scale.shape) != (rows, columns // 16):
        raise ValueError("Expected unswizzled ModelOpt block-size-16 scales")
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device=device, dtype=torch.float32)
    result = torch.empty((rows, columns), device=device, dtype=dtype)
    for start in range(0, rows, 256):
        block = packed[start : start + 256].to(device)
        codes = torch.stack((block & 15, block >> 4), dim=-1).reshape(-1, columns // 16, 16)
        factor = scale[start : start + 256].to(device=device, dtype=torch.float32)
        factor = factor * global_scale.to(device=device, dtype=torch.float32)
        result[start : start + 256] = (values[codes.long()] * factor.unsqueeze(-1)).reshape(-1, columns).to(dtype)
    return result


def load_pe_bf16(model_path, device):
    """Load the public NVFP4 checkpoint on Hopper without private remote blocks."""
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    root = Path(model_path)
    config = AutoConfig.from_pretrained(root, local_files_only=True)
    quantization = config.quantization_config
    if (quantization["quant_method"], quantization["quant_algo"], quantization["group_size"]) != ("modelopt", "NVFP4", 16):
        raise ValueError("This PE loader expects the released ModelOpt NVFP4 checkpoint")
    del config.quantization_config
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16, attn_implementation="sdpa")
    mapping = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    expected = model.state_dict()
    with ExitStack() as stack:
        handles = {name: stack.enter_context(safe_open(root / name, framework="pt", device="cpu")) for name in set(mapping.values())}

        def tensor(key):
            return handles[mapping[key]].get_tensor(key)

        for index, (key, prototype) in enumerate(expected.items()):
            value = tensor(key)
            prefix = key.removesuffix(".weight")
            if key.endswith(".weight") and prefix + ".weight_scale" in mapping:
                value = dequantize_nvfp4(value, tensor(prefix + ".weight_scale"), tensor(prefix + ".weight_scale_2"), device=device, dtype=torch.bfloat16)
            if value.shape != prototype.shape:
                raise ValueError(f"PE checkpoint shape mismatch at {key}: {value.shape} != {prototype.shape}")
            set_module_tensor_to_device(model, key, device, value=value, dtype=value.dtype)
            if index % 100 == 0:
                print(f"[pe-load] tensors={index + 1}/{len(expected)}", flush=True)
    model.to(device).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    return model, tokenizer


class TransformersExpander:
    def __init__(self, model_path, recipe, device):
        import xgrammar as xgr

        self.recipe, self.device = recipe, device
        started = perf_counter()
        self.model, self.tokenizer = load_pe_bf16(model_path, device)
        info = xgr.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=self.model.config.text_config.vocab_size)
        schema = recipe["parameters"]["response_format"]["json_schema"]["schema"]
        self.grammar = xgr.GrammarCompiler(info).compile_json_schema(json.dumps(schema), any_whitespace=False)
        self.end_think = self.tokenizer.convert_tokens_to_ids("</think>")
        if self.end_think is None:
            raise ValueError("The PE tokenizer has no </think> token")
        print(f"[pe] model_loaded device={device} seconds={perf_counter() - started:.1f}", flush=True)

    def __call__(self, prompt, seed):
        import torch
        from xgrammar.contrib.hf import LogitsProcessor

        parameters = self.recipe["parameters"]
        ids = self.tokenizer.apply_chat_template(messages_for(prompt, self.recipe), tokenize=True, add_generation_prompt=True, return_dict=False, **parameters["chat_template_kwargs"])
        inputs = torch.tensor([ids], device=self.device)
        grammar_processor = LogitsProcessor(self.grammar)
        end_think = self.end_think
        prompt_length = inputs.shape[1]

        class AnswerGrammar:
            def __init__(self):
                self.answer_started = False

            def __call__(self, input_ids, scores):
                if not self.answer_started:
                    self.answer_started = bool((input_ids[:, prompt_length:] == end_think).any())
                return grammar_processor(input_ids, scores) if self.answer_started else scores

        torch.manual_seed(seed)
        started = perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=parameters["max_tokens"],
                do_sample=True,
                temperature=parameters["temperature"],
                top_p=parameters["top_p"],
                top_k=parameters["top_k"],
                min_p=parameters["min_p"],
                repetition_penalty=parameters["repetition_penalty"],
                logits_processor=[AnswerGrammar()],
                use_cache=True,
            )
        answer_tokens = generated[0, prompt_length:]
        delimiters = (answer_tokens == self.end_think).nonzero().flatten()
        if not len(delimiters):
            raise ValueError("PE reached its token budget without finishing its reasoning")
        answer_tokens = answer_tokens[int(delimiters[-1]) + 1 :]
        text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        print(f"[pe-generate] device={self.device} tokens={generated.shape[1] - prompt_length} seconds={perf_counter() - started:.1f}", flush=True)
        return parse_answer(text, self.recipe)


class HTTPExpander:
    def __init__(self, endpoint, served_model, recipe):
        self.endpoint = endpoint.rstrip("/") + "/v1/chat/completions"
        self.served_model, self.recipe = served_model, recipe

    def __call__(self, prompt, seed):
        payload = {**self.recipe["parameters"], "model": self.served_model, "messages": messages_for(prompt, self.recipe), "stream": False, "seed": seed}
        request = Request(self.endpoint, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=1200) as response:
            result = json.load(response)
        return parse_answer(result["choices"][0]["message"]["content"], self.recipe)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--pe-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=10000)
    parser.add_argument("--backend", choices=("transformers", "http"), default="transformers")
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--served-model", default="MiniMax-H3-Image-PE")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    import torch
    import torch.distributed as dist

    rank, world, local_rank = (int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    torch.set_num_threads(4)
    if world > 1:
        dist.init_process_group("gloo", timeout=timedelta(days=3))
    rows = read_prompts(args.prompts, args.expected_samples)
    prompt_recipe = load_prompt_recipe(args.pe_model_path)
    recipe = {
        "schema_version": 1,
        "prompts_sha256": fingerprint(rows),
        "backend": args.backend,
        "pe_config_sha256": fingerprint(json.loads((args.pe_model_path / "config.json").read_text())),
        "pe_index_sha256": fingerprint(json.loads((args.pe_model_path / "model.safetensors.index.json").read_text())),
        "prompt_recipe": prompt_recipe,
        "seed": args.seed,
        "target_aspect_ratio": "1:1",
    }
    root = args.output_dir.resolve()
    if rank == 0:
        (root / "items").mkdir(parents=True, exist_ok=True)
        establish_recipe(root / "pe_recipe.json", recipe)
    if world > 1:
        dist.barrier()
    recipe_sha = fingerprint(recipe)
    pending = []
    for index, row in enumerate(rows):
        if index % world != rank:
            continue
        path = root / "items" / f"{row['id']:08d}.json"
        if path.exists():
            item = json.loads(path.read_text())
            if item["pe_fingerprint"] != fingerprint([recipe_sha, row]):
                raise ValueError(f"Stale PE result: {path}")
        else:
            pending.append(row)
    if pending:
        device = f"cuda:{local_rank}"
        if args.backend == "transformers":
            torch.cuda.set_device(local_rank)
            expander = TransformersExpander(args.pe_model_path, prompt_recipe, device)
        else:
            expander = HTTPExpander(args.endpoint, args.served_model, prompt_recipe)
        for index, row in enumerate(pending):
            answer = expander(row["prompt"], args.seed + row["id"])
            item = {
                **row,
                "original_prompt": row["prompt"],
                "prompt": answer["expanded_caption"],
                "pe_aspect_ratio": answer["aspect_ratio"],
                "target_aspect_ratio": "1:1",
                "pe_fingerprint": fingerprint([recipe_sha, row]),
            }
            atomic_json(root / "items" / f"{row['id']:08d}.json", item)
            print(f"[pe] rank={rank}/{world} completed={index + 1}/{len(pending)} id={row['id']}", flush=True)
        del expander
    if world > 1:
        dist.barrier()
    if rank == 0:
        expanded = []
        for row in rows:
            item = json.loads((root / "items" / f"{row['id']:08d}.json").read_text())
            if item["pe_fingerprint"] != fingerprint([recipe_sha, row]):
                raise ValueError("PE merge found an inconsistent item")
            expanded.append(item)
        atomic_jsonl(root / "prompts_expanded.jsonl", expanded)
        atomic_json(root / "pe.complete.json", {"count": len(expanded), "recipe_sha256": recipe_sha, "prompts_sha256": fingerprint(expanded)})
        print(f"[pe] complete={len(expanded)} path={root / 'prompts_expanded.jsonl'}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

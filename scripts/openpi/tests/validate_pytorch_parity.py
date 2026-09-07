"""Compare upstream OpenPI PyTorch and LightX2V with identical input and noise.

Run this with OpenPI's patched conversion environment. The deployed LightX2V
runtime itself does not import OpenPI or JAX.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import sys
import types
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
OPENPI_DATA_ROOT = WORKSPACE_ROOT / "openpi_data"
PRECISIONS = ("bfloat16", "float32")


def load_sample(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        demo = handle["data/demo_0"]
        image = np.asarray(demo["obs/agentview_rgb"][0], dtype=np.uint8)
        wrist = np.asarray(demo["obs/eye_in_hand_rgb"][0], dtype=np.uint8)
        state = np.concatenate([demo["obs/ee_pos"][0], demo["obs/ee_ori"][0], demo["obs/gripper_states"][0]])
    return image, wrist, state


def _install_lightx2v_package_stub() -> None:
    """Avoid LightX2V platform initialization in this model-only validator."""
    package = types.ModuleType("lightx2v")
    package.__path__ = [str(PROJECT_ROOT / "lightx2v")]
    sys.modules["lightx2v"] = package


def _create_upstream_policy(train_config, checkpoint: Path, device: str, precision: str):
    """Create the official policy while honoring its selectable torch precision.

    ``policy_config.create_trained_policy`` in OpenPI 15a9616 always applies
    BF16 after loading. This is the same construction with that one choice made
    explicit, so the validator can cover both official model modes.
    """
    from openpi import transforms
    from openpi.policies import policy as policy_module
    from openpi.training import checkpoints

    weight_path = checkpoint / "model.safetensors"
    model = train_config.model.load_pytorch(train_config, str(weight_path))
    model.paligemma_with_expert.to_bfloat16_for_selected_params(precision)

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("The upstream LIBERO data config has no normalization asset id")
    norm_stats = checkpoints.load_norm_stats(checkpoint / "assets", data_config.asset_id)

    return policy_module.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata=train_config.policy_metadata,
        is_pytorch=True,
        pytorch_device=device,
    )


def _parameter_dtype_counts(model: torch.nn.Module) -> dict[str, int]:
    counts = Counter(str(parameter.dtype).removeprefix("torch.") for parameter in model.parameters() if parameter.is_floating_point())
    return dict(sorted(counts.items()))


def _validate_selected_precision(counts: dict[str, int], precision: str, label: str) -> None:
    if precision == "float32":
        if set(counts) != {"float32"}:
            raise RuntimeError(f"{label} FP32 mode contains unexpected parameter dtypes: {counts}")
        return
    if "bfloat16" not in counts or "float32" not in counts:
        raise RuntimeError(f"{label} BF16 mode must retain the official selected FP32 parameters: {counts}")


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _array_comparison(reference, candidate) -> dict[str, object]:
    reference_array = _to_numpy(reference)
    candidate_array = _to_numpy(candidate)
    same_shape = reference_array.shape == candidate_array.shape
    if same_shape and reference_array.size:
        difference = reference_array.astype(np.float64) - candidate_array.astype(np.float64)
        max_abs_error = float(np.max(np.abs(difference)))
    else:
        max_abs_error = 0.0 if same_shape else None
    return {
        "upstream_shape": list(reference_array.shape),
        "lightx2v_shape": list(candidate_array.shape),
        "upstream_dtype": str(reference_array.dtype),
        "lightx2v_dtype": str(candidate_array.dtype),
        "exact": bool(same_shape and np.array_equal(reference_array, candidate_array)),
        "max_abs_error": max_abs_error,
    }


def _observation_comparison(upstream, local) -> dict[str, object]:
    components = {
        **{f"images/{key}": _array_comparison(upstream.images[key], local.images[key]) for key in upstream.images},
        **{f"image_masks/{key}": _array_comparison(upstream.image_masks[key], local.image_masks[key]) for key in upstream.image_masks},
        "state": _array_comparison(upstream.state, local.state),
        "tokenized_prompt": _array_comparison(upstream.tokenized_prompt, local.tokenized_prompt),
        "tokenized_prompt_mask": _array_comparison(
            upstream.tokenized_prompt_mask,
            local.tokenized_prompt_mask,
        ),
    }
    expected_image_keys = set(local.images)
    expected_mask_keys = set(local.image_masks)
    keys_match = set(upstream.images) == expected_image_keys and set(upstream.image_masks) == expected_mask_keys
    exact = keys_match and all(bool(component["exact"]) for component in components.values())
    return {
        "exact": exact,
        "image_keys_match": keys_match,
        "components": components,
    }


def _to_torch_batch(data, device: str):
    if isinstance(data, dict):
        return {key: _to_torch_batch(value, device) for key, value in data.items()}
    return torch.from_numpy(np.array(data)).to(device)[None, ...]


def run_self_check() -> None:
    """Exercise precision selection and the exact uint8 resize adapter."""
    from openpi_client import image_tools as upstream_image_tools

    _install_lightx2v_package_stub()
    from lightx2v.models.networks.openpi.config import Pi0Config
    from lightx2v.models.networks.openpi.infer.pre_infer import _resize_with_pad

    values = {
        "action_dim": 32,
        "action_horizon": 10,
        "max_token_len": 200,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "pi05": True,
        "discrete_state_input": False,
        "pytorch_compile_mode": None,
    }
    for precision in PRECISIONS:
        config = Pi0Config.from_mapping({**values, "dtype": precision})
        config.validate_pi05_libero()
        if config.dtype != precision:
            raise AssertionError(f"precision selection changed {precision!r} to {config.dtype!r}")

    image = np.arange(17 * 29 * 3, dtype=np.uint8).reshape(17, 29, 3)
    upstream = upstream_image_tools.resize_with_pad(image, 24, 24)
    local = _resize_with_pad(image, 24)
    if not np.array_equal(local, upstream):
        difference = np.abs(local.astype(np.int16) - upstream.astype(np.int16))
        raise AssertionError(f"resize-with-pad mismatch: max_abs={difference.max()}")

    print(json.dumps({"precision_selection": list(PRECISIONS), "uint8_resize_exact": True}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=OPENPI_DATA_ROOT / "openpi-assets/checkpoints/pi05_libero_pytorch_fp32",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/openpi/pi05_libero.json",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=OPENPI_DATA_ROOT / "raw/huggingface/yifengzhu-hf/LIBERO-datasets/libero_spatial" / "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OPENPI_DATA_ROOT / "results/pi05_libero_pytorch_parity.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=PRECISIONS, help="Override the dtype selected by the model JSON")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--self-check", action="store_true", help="Run lightweight checks without loading a checkpoint")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_check:
        run_self_check()
        return
    if args.atol < 0:
        raise ValueError("--atol must be non-negative")

    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    sample_path = args.sample.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    required = (
        checkpoint / "model.safetensors",
        checkpoint / "assets/paligemma_tokenizer.model",
        checkpoint / "assets/physical-intelligence/libero/norm_stats.json",
        config_path,
        sample_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Parity inputs are missing: {missing}")

    with config_path.open("r", encoding="utf-8") as handle:
        local_config = json.load(handle)
    precision = args.precision or local_config["dtype"]
    if precision not in PRECISIONS:
        raise ValueError(f"Unsupported OpenPI precision {precision!r}; expected one of {PRECISIONS}")
    local_config["dtype"] = precision

    image, wrist, state = load_sample(sample_path)
    # Match the official LIBERO evaluation client: renderer frames are resized
    # with PIL before they enter either policy. Both model adapters therefore
    # receive the exact same 224x224 uint8 arrays.
    from openpi_client import image_tools as client_image_tools

    image = client_image_tools.resize_with_pad(image, 224, 224)
    wrist = client_image_tools.resize_with_pad(wrist, 224, 224)
    prompt = "pick up the black bowl between the plate and the ramekin and place it on the plate"
    noise = np.random.default_rng(0).standard_normal((10, 32)).astype(np.float32)

    from openpi.training import config as training_config

    upstream_config = training_config.get_config("pi05_libero")
    upstream_config = dataclasses.replace(
        upstream_config,
        model=dataclasses.replace(
            upstream_config.model,
            dtype=precision,
            pytorch_compile_mode=None,
        ),
    )
    upstream_policy = _create_upstream_policy(upstream_config, checkpoint, args.device, precision)
    upstream_dtype_counts = _parameter_dtype_counts(upstream_policy._model)  # noqa: SLF001
    _validate_selected_precision(upstream_dtype_counts, precision, "upstream")
    raw_input = {
        "observation/image": image.copy(),
        "observation/wrist_image": wrist.copy(),
        "observation/state": state.copy(),
        "prompt": prompt,
    }
    upstream_inputs = upstream_policy._input_transform(raw_input)  # noqa: SLF001
    upstream_torch_inputs = _to_torch_batch(upstream_inputs, args.device)
    from openpi.models import model as upstream_model_module

    upstream_observation = upstream_model_module.Observation.from_dict(upstream_torch_inputs)
    upstream_noise = torch.from_numpy(noise).to(args.device)[None, ...]
    upstream_normalized = upstream_policy._sample_actions(  # noqa: SLF001
        args.device,
        upstream_observation,
        noise=upstream_noise,
    )
    upstream_outputs = {
        "state": _to_numpy(upstream_torch_inputs["state"][0]),
        "actions": _to_numpy(upstream_normalized[0]),
    }
    upstream_actions = np.asarray(upstream_policy._output_transform(upstream_outputs)["actions"])  # noqa: SLF001
    upstream_normalized_array = _to_numpy(upstream_normalized)
    del upstream_policy, upstream_normalized, upstream_noise, upstream_torch_inputs
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    _install_lightx2v_package_stub()
    from lightx2v.models.networks.openpi import OpenPIModel

    local_config["model_path"] = str(checkpoint)
    local_config["device"] = args.device
    local_config["seed"] = 0
    model = OpenPIModel.from_config(local_config)
    local_dtype_counts = _parameter_dtype_counts(model.core_model)
    _validate_selected_precision(local_dtype_counts, precision, "LightX2V")
    local_observation = model.pre_infer.infer(
        images={"agentview": image, "wrist": wrist},
        state=state,
        task_description=prompt,
    )
    preprocessing = _observation_comparison(upstream_observation, local_observation)
    local_normalized = model.transformer_infer.infer(
        model.core_model,
        local_observation,
        model.device,
        noise=torch.from_numpy(noise).to(model.device)[None, ...],
    )
    local_actions = model.post_infer.infer(local_normalized)

    normalized_difference = _to_numpy(local_normalized).astype(np.float64) - upstream_normalized_array.astype(np.float64)
    normalized_max_abs_error = float(np.max(np.abs(normalized_difference)))
    normalized_passed = bool(np.allclose(_to_numpy(local_normalized), upstream_normalized_array, rtol=0.0, atol=args.atol))
    physical_difference = np.asarray(local_actions, dtype=np.float64) - np.asarray(upstream_actions, dtype=np.float64)
    physical_max_abs_error = float(np.max(np.abs(physical_difference)))
    physical_passed = bool(np.allclose(local_actions, upstream_actions, rtol=0.0, atol=args.atol))
    passed = bool(preprocessing["exact"] and normalized_passed and physical_passed)
    report = {
        "precision": precision,
        "atol": args.atol,
        "upstream_parameter_dtypes": upstream_dtype_counts,
        "lightx2v_parameter_dtypes": local_dtype_counts,
        "preprocessing": preprocessing,
        "normalized_actions": {
            "upstream_shape": list(upstream_normalized_array.shape),
            "lightx2v_shape": list(local_normalized.shape),
            "max_abs_error": normalized_max_abs_error,
            "mean_abs_error": float(np.mean(np.abs(normalized_difference))),
            "allclose": normalized_passed,
        },
        "physical_actions": {
            "upstream_shape": list(upstream_actions.shape),
            "lightx2v_shape": list(local_actions.shape),
            "max_abs_error": physical_max_abs_error,
            "mean_abs_error": float(np.mean(np.abs(physical_difference))),
            "allclose": physical_passed,
        },
        "allclose": passed,
        "upstream_normalized_actions": upstream_normalized_array[0].tolist(),
        "lightx2v_normalized_actions": _to_numpy(local_normalized[0]).tolist(),
        "upstream_actions": upstream_actions.tolist(),
        "lightx2v_actions": local_actions.tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    raw_action_keys = {
        "upstream_normalized_actions",
        "lightx2v_normalized_actions",
        "upstream_actions",
        "lightx2v_actions",
    }
    print(json.dumps({key: value for key, value in report.items() if key not in raw_action_keys}, indent=2))
    del model
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    if not passed:
        raise SystemExit(
            "OpenPI PyTorch parity check failed: "
            f"preprocessing_exact={preprocessing['exact']}, "
            f"normalized_max_abs_error={normalized_max_abs_error}, "
            f"physical_max_abs_error={physical_max_abs_error}, atol={args.atol}"
        )


if __name__ == "__main__":
    main()

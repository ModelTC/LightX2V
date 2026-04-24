import argparse
import base64
from pathlib import Path
from typing import Any

import requests

try:
    from openai import OpenAI  # pyright: ignore[reportMissingImports]
except ImportError:
    OpenAI = None  # type: ignore[assignment]


def _extract_data_item(response: Any) -> dict[str, Any]:
    if not hasattr(response, "data") or not response.data:
        raise RuntimeError(f"Invalid OpenAI images response: {response}")
    item = response.data[0]
    if hasattr(item, "model_dump"):
        return item.model_dump()  # openai pydantic object
    if isinstance(item, dict):
        return item
    raise RuntimeError(f"Unsupported data item type: {type(item)!r}")


def _save_image_from_item(item: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "b64_json" in item and item["b64_json"]:
        image_bytes = base64.b64decode(item["b64_json"])
        output_path.write_bytes(image_bytes)
        return output_path

    if "url" in item and item["url"]:
        resp = requests.get(item["url"], timeout=120)
        resp.raise_for_status()
        output_path.write_bytes(resp.content)
        return output_path

    raise RuntimeError(f"Response item has neither b64_json nor url: {item}")


def run_generate(client: Any, args: argparse.Namespace) -> Path:
    response = client.images.generate(
        model=args.model,
        prompt=args.prompt,
        size=args.size,
        response_format=args.response_format,
    )
    item = _extract_data_item(response)
    print(f"[generate] response item: {item}")
    return _save_image_from_item(item, Path(args.output_dir) / "generate.png")


def run_edit(client: Any, args: argparse.Namespace) -> Path:
    if not args.image:
        raise ValueError("--image is required for edit mode")

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with image_path.open("rb") as image_file:
        kwargs = {
            "model": args.model,
            "image": image_file,
            "prompt": args.edit_prompt or args.prompt,
            "size": args.size,
            "response_format": args.response_format,
        }
        if args.mask:
            mask_path = Path(args.mask)
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            with mask_path.open("rb") as mask_file:
                response = client.images.edit(mask=mask_file, **kwargs)
        else:
            response = client.images.edit(**kwargs)

    item = _extract_data_item(response)
    print(f"[edit] response item: {item}")
    return _save_image_from_item(item, Path(args.output_dir) / "edit.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test OpenAI-compatible image APIs on LightX2V server.")
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--api_key", type=str, default="dummy-key", help="OpenAI API key placeholder")
    parser.add_argument("--model", type=str, default="gpt-image-1", help="Model name (for compatibility only)")
    parser.add_argument("--mode", choices=["generate", "edit", "all"], default="all", help="Test mode")
    parser.add_argument("--prompt", type=str, default="a futuristic city at sunset", help="Prompt for generation")
    parser.add_argument("--edit_prompt", type=str, default="", help="Prompt for edit (defaults to --prompt)")
    parser.add_argument("--size", type=str, default="1024x1024", help="Image size, e.g. 1024x1024")
    parser.add_argument("--response_format", choices=["url", "b64_json"], default="url", help="OpenAI response format")
    parser.add_argument("--image", type=str, default="", help="Input image path for edit mode")
    parser.add_argument("--mask", type=str, default="", help="Optional mask image path for edit mode")
    parser.add_argument("--output_dir", type=str, default="outputs/openai_images_test", help="Directory to save outputs")
    args = parser.parse_args()

    if OpenAI is None:
        raise RuntimeError("Missing dependency: openai. Please install it with `pip install openai`.")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    output_paths: list[Path] = []
    if args.mode in ("generate", "all"):
        output_paths.append(run_generate(client, args))
    if args.mode in ("edit", "all"):
        output_paths.append(run_edit(client, args))

    for path in output_paths:
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()

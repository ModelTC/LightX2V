import argparse
import base64
import json
import os
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

try:
    from openai import OpenAI  # pyright: ignore[reportMissingImports]
except ImportError:
    OpenAI = None  # type: ignore[assignment]


@dataclass
class SaveImageResult:
    path: Path
    source: str
    bytes_written: int
    total_seconds: float
    decode_seconds: float = 0.0
    download_seconds: float = 0.0
    write_seconds: float = 0.0


@dataclass
class ImageRequestResult:
    path: Path
    http_sdk_parse_seconds: float
    save: SaveImageResult


def _extract_data_item(response: Any) -> dict[str, Any]:
    if not hasattr(response, "data") or not response.data:
        raise RuntimeError(f"Invalid OpenAI images response: {response}")
    item = response.data[0]
    if hasattr(item, "model_dump"):
        return item.model_dump()  # openai pydantic object
    if isinstance(item, dict):
        return item
    raise RuntimeError(f"Unsupported data item type: {type(item)!r}")


def _save_image_from_item(item: dict[str, Any], output_path: Path) -> SaveImageResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    if "b64_json" in item and item["b64_json"]:
        decode_start = time.perf_counter()
        image_bytes = base64.b64decode(item["b64_json"])
        decode_seconds = time.perf_counter() - decode_start

        write_start = time.perf_counter()
        output_path.write_bytes(image_bytes)
        write_seconds = time.perf_counter() - write_start

        return SaveImageResult(
            path=output_path,
            source="b64_json",
            bytes_written=len(image_bytes),
            total_seconds=time.perf_counter() - total_start,
            decode_seconds=decode_seconds,
            write_seconds=write_seconds,
        )

    if "url" in item and item["url"]:
        download_start = time.perf_counter()
        resp = requests.get(item["url"], timeout=120)
        resp.raise_for_status()
        download_seconds = time.perf_counter() - download_start

        write_start = time.perf_counter()
        output_path.write_bytes(resp.content)
        write_seconds = time.perf_counter() - write_start

        return SaveImageResult(
            path=output_path,
            source="url",
            bytes_written=len(resp.content),
            total_seconds=time.perf_counter() - total_start,
            download_seconds=download_seconds,
            write_seconds=write_seconds,
        )

    raise RuntimeError(f"Response item has neither b64_json nor url: {item}")


def _summarize_response_item(item: dict[str, Any]) -> dict[str, Any]:
    summary = dict(item)
    if "b64_json" in summary and summary["b64_json"]:
        summary["b64_json"] = f"<base64 {len(summary['b64_json'])} chars>"
    return summary


def _format_duration(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    total_seconds, milliseconds = divmod(total_ms, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _format_request_timing(result: ImageRequestResult) -> str:
    save = result.save
    parts = [
        f"http_sdk_parse={result.http_sdk_parse_seconds:.3f}s",
        f"save_total={save.total_seconds:.3f}s",
    ]
    if save.decode_seconds:
        parts.append(f"base64_decode={save.decode_seconds:.3f}s")
    if save.download_seconds:
        parts.append(f"download={save.download_seconds:.3f}s")
    parts.extend(
        [
            f"disk_write={save.write_seconds:.3f}s",
            f"bytes={save.bytes_written}",
            f"source={save.source}",
        ]
    )
    return ", ".join(parts)


def _ensure_local_no_proxy(base_url: str) -> None:
    hostname = urlparse(base_url).hostname
    if hostname not in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}:
        return

    local_hosts = ["127.0.0.1", "localhost", "0.0.0.0", "::1"]
    for env_name in ("NO_PROXY", "no_proxy"):
        existing = [item.strip() for item in os.environ.get(env_name, "").split(",") if item.strip()]
        merged = local_hosts + [item for item in existing if item not in local_hosts]
        os.environ[env_name] = ",".join(merged)


def _load_prompts(prompt_json: str) -> list[str]:
    path = Path(prompt_json)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")

    prompts = []
    for index, item in enumerate(data):
        if isinstance(item, str):
            prompt = item
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            prompt = item["prompt"]
        else:
            raise ValueError(f"{path}[{index}] must be a string or an object with a string prompt")
        prompts.append(prompt)
    return prompts


def _write_summary_line(summary_file: Path, line: str) -> None:
    with summary_file.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")


def _extra_body_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.seed is None:
        return None
    return {"seed": args.seed}


def _image_paths_from_args(args: argparse.Namespace) -> list[Path]:
    image_paths = [Path(path.strip()) for path in args.image.split(",") if path.strip()]
    if not image_paths:
        raise ValueError("--image is required for edit mode")

    for image_path in image_paths:
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
    return image_paths


def run_generate(client: Any, args: argparse.Namespace, prompt: str | None = None, output_path: Path | None = None) -> ImageRequestResult:
    prompt = args.prompt if prompt is None else prompt
    output_path = Path(args.output_dir) / "generate.png" if output_path is None else output_path

    request_start = time.perf_counter()
    response = client.images.generate(
        model=args.model,
        prompt=prompt,
        size=args.size,
        response_format=args.response_format,
        extra_body=_extra_body_from_args(args),
    )
    http_sdk_parse_seconds = time.perf_counter() - request_start

    item = _extract_data_item(response)
    print(f"[generate] response item: {_summarize_response_item(item)}")
    save_result = _save_image_from_item(item, output_path)
    result = ImageRequestResult(path=save_result.path, http_sdk_parse_seconds=http_sdk_parse_seconds, save=save_result)
    print(f"[generate] timing: {_format_request_timing(result)}")
    return result


def run_edit(client: Any, args: argparse.Namespace) -> ImageRequestResult:
    image_paths = _image_paths_from_args(args)

    with ExitStack() as stack:
        image_files = [stack.enter_context(image_path.open("rb")) for image_path in image_paths]
        kwargs = {
            "model": args.model,
            "image": image_files[0] if len(image_files) == 1 else image_files,
            "prompt": args.edit_prompt or args.prompt,
            "size": args.size,
            "response_format": args.response_format,
            "extra_body": _extra_body_from_args(args),
        }
        request_start = time.perf_counter()
        if args.mask:
            mask_path = Path(args.mask)
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            with mask_path.open("rb") as mask_file:
                response = client.images.edit(mask=mask_file, **kwargs)
        else:
            response = client.images.edit(**kwargs)
        http_sdk_parse_seconds = time.perf_counter() - request_start

    item = _extract_data_item(response)
    print(f"[edit] response item: {_summarize_response_item(item)}")
    save_result = _save_image_from_item(item, Path(args.output_dir) / "edit.png")
    result = ImageRequestResult(path=save_result.path, http_sdk_parse_seconds=http_sdk_parse_seconds, save=save_result)
    print(f"[edit] timing: {_format_request_timing(result)}")
    return result


def run_generate_batch(client: Any, args: argparse.Namespace) -> int:
    prompts = _load_prompts(args.prompt_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = Path(args.summary_file) if args.summary_file else output_dir / f"{args.output_prefix}_summary_{run_stamp}.log"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    total = len(prompts)
    completed = 0
    failed = 0
    batch_start = time.perf_counter()
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary_file.write_text(
        "\n".join(
            [
                f"Run started at: {started_at}",
                f"Prompt JSON: {args.prompt_json}",
                f"Base URL: {args.base_url}",
                f"Model: {args.model}",
                f"Seed: {args.seed}",
                f"Size: {args.size}",
                f"Response format: {args.response_format}",
                f"Output directory: {output_dir}",
                f"Total prompts: {total}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Posting {total} OpenAI-format image prompts from {args.prompt_json} to {args.base_url}")
    print(f"Output directory: {output_dir}")
    print(f"Summary file: {summary_file}")

    for index, prompt in enumerate(prompts, 1):
        number = f"{index:03d}"
        output_path = output_dir / f"{args.output_prefix}_{number}.png"
        case_start = time.perf_counter()
        case_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"[{index}/{total}] submitting {output_path.name}")
        _write_summary_line(summary_file, f"Case {number} started at: {case_started_at}")

        try:
            result = run_generate(client, args, prompt=prompt, output_path=output_path)
        except Exception as e:
            failed += 1
            elapsed = time.perf_counter() - case_start
            print(f"[{index}/{total}] failed: {output_path.name}: {e}")
            _write_summary_line(summary_file, f"Case {number} status: failed, elapsed: {_format_duration(elapsed)} ({elapsed:.3f}s), error: {e}")
            if args.stop_on_error:
                break
        else:
            completed += 1
            elapsed = time.perf_counter() - case_start
            print(f"[{index}/{total}] saved {result.path}")
            print(f"[{index}/{total}] elapsed: {_format_duration(elapsed)} ({elapsed:.3f}s)")
            _write_summary_line(
                summary_file,
                f"Case {number} status: success, elapsed: {_format_duration(elapsed)} ({elapsed:.3f}s), {_format_request_timing(result)}, output: {result.path}",
            )

    total_elapsed = time.perf_counter() - batch_start
    ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_summary_line(summary_file, "")
    _write_summary_line(summary_file, f"Run ended at: {ended_at}")
    _write_summary_line(summary_file, f"Elapsed seconds: {total_elapsed:.3f}")
    _write_summary_line(summary_file, f"Elapsed time: {_format_duration(total_elapsed)}")
    _write_summary_line(summary_file, f"Completed prompts: {completed}")
    _write_summary_line(summary_file, f"Failed prompts: {failed}")

    print(f"Finished: completed={completed}/{total}, failed={failed}/{total}")
    print(f"Total elapsed: {_format_duration(total_elapsed)} ({total_elapsed:.3f}s)")
    print(f"Summary written to: {summary_file}")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Test OpenAI-compatible image APIs on LightX2V server.")
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--api_key", type=str, default="dummy-key", help="OpenAI API key placeholder")
    parser.add_argument("--model", type=str, default="gpt-image-1", help="Model name (for compatibility only)")
    parser.add_argument("--mode", choices=["generate", "edit", "all"], default="generate", help="Test mode")
    parser.add_argument("--prompt", type=str, default="a futuristic city at sunset", help="Prompt for generation")
    parser.add_argument("--prompt_json", "--json", dest="prompt_json", type=str, default="", help="JSON file containing prompts for batch generation")
    parser.add_argument("--edit_prompt", type=str, default="", help="Prompt for edit (defaults to --prompt)")
    parser.add_argument("--seed", type=int, default=None, help="Optional generation seed")
    parser.add_argument("--size", type=str, default="1024x1024", help="Image size, e.g. 1024x1024")
    parser.add_argument("--response_format", choices=["url", "b64_json"], default="b64_json", help="OpenAI response format")
    parser.add_argument("--image", type=str, default="", help="Input image path for edit mode. Use comma-separated paths for multiple images.")
    parser.add_argument("--mask", type=str, default="", help="Optional mask image path for edit mode")
    parser.add_argument("--output_dir", type=str, default="outputs/openai_images_test", help="Directory to save outputs")
    parser.add_argument("--output_prefix", type=str, default="openai_generate", help="Batch output filename prefix")
    parser.add_argument("--summary_file", type=str, default="", help="Batch timing summary file")
    parser.add_argument("--stop_on_error", action="store_true", help="Stop batch mode after the first failed prompt")
    args = parser.parse_args()

    if OpenAI is None:
        raise RuntimeError("Missing dependency: openai. Please install it with `pip install openai`.")

    _ensure_local_no_proxy(args.base_url)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    if args.prompt_json:
        raise SystemExit(run_generate_batch(client, args))

    output_results: list[ImageRequestResult] = []
    if args.mode in ("generate", "all"):
        output_results.append(run_generate(client, args))
    if args.mode in ("edit", "all"):
        output_results.append(run_edit(client, args))

    for result in output_results:
        print(f"[saved] {result.path}")


if __name__ == "__main__":
    main()

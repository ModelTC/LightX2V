import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

try:
    import boto3  # pyright: ignore[reportMissingImports]
    from botocore.config import Config as BotoConfig  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - runtime dependency check
    boto3 = None
    BotoConfig = None


def build_presigned_url(args: argparse.Namespace) -> tuple[str, Optional[str]]:
    if args.presigned_url:
        return args.presigned_url, (args.download_url or None)

    if boto3 is None:
        raise RuntimeError("boto3 is required to auto-generate presigned URL, please install boto3/aioboto3.")
    if BotoConfig is None:
        raise RuntimeError("botocore is required to configure S3 client.")

    bucket = args.s3_bucket or os.getenv("S3_BUCKET", "")
    if not bucket:
        raise ValueError("Missing S3 bucket. Set --s3_bucket or env S3_BUCKET.")

    object_key = args.s3_object_key
    if not object_key:
        key_prefix = (args.s3_key_prefix or os.getenv("S3_BASE_PATH", "lightx2v/sync")).strip("/")
        object_name = f"{uuid.uuid4().hex}.png"
        object_key = f"{key_prefix}/{object_name}" if key_prefix else object_name

    region = args.s3_region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    access_key = args.s3_access_key or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = args.s3_secret_key or os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = args.s3_session_token or os.getenv("AWS_SESSION_TOKEN")
    addressing_style = (args.s3_addressing_style or os.getenv("S3_ADDRESSING_STYLE", "auto")).strip().lower()
    if addressing_style not in {"auto", "path", "virtual"}:
        raise ValueError("--s3_addressing_style must be one of: auto, path, virtual")
    signature_version = (args.s3_signature_version or os.getenv("S3_SIGNATURE_VERSION", "s3v4")).strip()

    client_kwargs: Dict[str, Any] = {
        "service_name": "s3",
        "region_name": region,
        "config": BotoConfig(
            signature_version=signature_version,
            s3={"addressing_style": addressing_style},
        ),
    }
    if args.s3_endpoint_url:
        client_kwargs["endpoint_url"] = args.s3_endpoint_url
    if access_key and secret_key:
        client_kwargs["aws_access_key_id"] = access_key
        client_kwargs["aws_secret_access_key"] = secret_key
    if session_token:
        client_kwargs["aws_session_token"] = session_token

    s3_client = boto3.client(**client_kwargs)
    put_presigned_url = s3_client.generate_presigned_url(
        ClientMethod="put_object",
        Params={"Bucket": bucket, "Key": object_key},
        ExpiresIn=args.presign_expires,
        HttpMethod="PUT",
    )
    get_presigned_url = s3_client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": object_key},
        ExpiresIn=args.presign_expires,
        HttpMethod="GET",
    )
    print(f"Generated presigned URL for s3://{bucket}/{object_key}")
    print(f"Upload presigned URL: {put_presigned_url}")
    return put_presigned_url, get_presigned_url


def build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "infer_steps": args.infer_steps,
        "seed": args.seed,
        "aspect_ratio": args.aspect_ratio,
        "save_result_path": args.save_result_path,
        "use_prompt_enhancer": args.use_prompt_enhancer,
        "presigned_url": args.presigned_url,
    }
    if args.target_shape:
        payload["target_shape"] = args.target_shape
    return payload


def call_sync_api(args: argparse.Namespace, payload: Dict[str, Any]) -> requests.Response:
    endpoint = f"{args.url.rstrip('/')}/v1/tasks/image/sync?timeout_seconds={args.timeout_seconds}&poll_interval_seconds={args.poll_interval_seconds}"
    return requests.post(endpoint, json=payload, timeout=args.timeout_seconds + 30)


def download_uploaded_image(download_url: str, output_path: str, timeout_seconds: int) -> Path:
    response = requests.get(download_url, timeout=timeout_seconds)
    if response.status_code != 200:
        raise RuntimeError(f"Download failed ({response.status_code}): {response.text}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(response.content)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Call /v1/tasks/image/sync for T2I with presigned_url upload.")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="Server base url")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt text")
    parser.add_argument("--infer_steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--aspect_ratio", type=str, default="16:9", help="Aspect ratio for image task")
    parser.add_argument(
        "--target_shape",
        type=int,
        nargs="+",
        default=None,
        help="Target output shape, e.g. --target_shape 1536 2752",
    )
    parser.add_argument("--save_result_path", type=str, default="", help="Server-side save_result_path")
    parser.add_argument("--use_prompt_enhancer", action="store_true", help="Enable prompt enhancer")
    parser.add_argument("--timeout_seconds", type=int, default=600, help="Sync API timeout_seconds")
    parser.add_argument("--poll_interval_seconds", type=float, default=0.5, help="Sync API poll_interval_seconds")
    parser.add_argument("--presigned_url", type=str, default="", help="Presigned URL used by server to upload final PNG")
    parser.add_argument("--s3_endpoint_url", type=str, default="", help="S3 compatible endpoint, e.g. https://s3.amazonaws.com")
    parser.add_argument("--s3_region", type=str, default="", help="S3 region, defaults to AWS_DEFAULT_REGION or us-east-1")
    parser.add_argument("--s3_bucket", type=str, default="", help="S3 bucket name used for auto-generating presigned URL")
    parser.add_argument(
        "--s3_key_prefix",
        type=str,
        default="",
        help="S3 object key prefix when object key is omitted, defaults to env S3_BASE_PATH or lightx2v/sync",
    )
    parser.add_argument("--s3_object_key", type=str, default="", help="Full S3 object key for uploaded result, e.g. lightx2v/sync/a.png")
    parser.add_argument("--s3_access_key", type=str, default="", help="S3 access key, defaults to AWS_ACCESS_KEY_ID")
    parser.add_argument("--s3_secret_key", type=str, default="", help="S3 secret key, defaults to AWS_SECRET_ACCESS_KEY")
    parser.add_argument("--s3_session_token", type=str, default="", help="S3 session token, defaults to AWS_SESSION_TOKEN")
    parser.add_argument(
        "--s3_addressing_style",
        type=str,
        default="",
        help="S3 addressing style: auto/path/virtual, defaults to env S3_ADDRESSING_STYLE or auto",
    )
    parser.add_argument(
        "--s3_signature_version",
        type=str,
        default="",
        help="S3 signature version, defaults to env S3_SIGNATURE_VERSION or s3v4",
    )
    parser.add_argument("--presign_expires", type=int, default=3600, help="Presigned URL expiry seconds")
    parser.add_argument(
        "--save_response_json",
        type=str,
        default="",
        help="Optional local path to save response JSON",
    )
    parser.add_argument("--download_url", type=str, default="", help="Optional URL to download result image for verification")
    parser.add_argument(
        "--download_output",
        type=str,
        default="save_results/t2i_sync_presigned_result.png",
        help="Local output path for downloaded result image",
    )
    parser.add_argument("--download_timeout_seconds", type=int, default=120, help="Timeout for download verification")
    args = parser.parse_args()

    target_shape: Optional[List[int]] = args.target_shape
    if target_shape is not None and len(target_shape) < 2:
        raise ValueError("--target_shape must provide at least 2 integers, e.g. --target_shape 1536 2752")
    if args.presign_expires <= 0:
        raise ValueError("--presign_expires must be > 0")
    if args.download_timeout_seconds <= 0:
        raise ValueError("--download_timeout_seconds must be > 0")

    args.presigned_url, resolved_download_url = build_presigned_url(args)

    payload = build_payload(args)
    response = call_sync_api(args, payload)

    if response.status_code != 200:
        raise RuntimeError(f"Request failed ({response.status_code}): {response.text}")

    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type:
        raise RuntimeError(f"Unexpected response type. presigned_url mode should return JSON, but got content-type={content_type!r}.")

    result = response.json()
    print("Sync t2i request succeeded:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.save_response_json:
        output = Path(args.save_response_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved response JSON to: {output}")

    download_url = args.download_url or resolved_download_url or result.get("presigned_url")
    if download_url:
        downloaded_image = download_uploaded_image(
            download_url=download_url,
            output_path=args.download_output,
            timeout_seconds=args.download_timeout_seconds,
        )
        print(f"Downloaded uploaded image to: {downloaded_image}")
    else:
        print("Skip download verification: no download URL available.")


if __name__ == "__main__":
    main()

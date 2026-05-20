import argparse
import json
import os
import statistics
import time
import uuid
from typing import Any, Dict, List

import requests

try:
    import boto3  # pyright: ignore[reportMissingImports]
    from botocore.config import Config as BotoConfig  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - runtime dependency check
    boto3 = None
    BotoConfig = None


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * p
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def build_s3_client(args: argparse.Namespace):
    if boto3 is None or BotoConfig is None:
        raise RuntimeError("boto3/botocore is required. Please install boto3 or aioboto3.")

    region = args.s3_region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    access_key = args.s3_access_key or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = args.s3_secret_key or os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = args.s3_session_token or os.getenv("AWS_SESSION_TOKEN")
    addressing_style = (args.s3_addressing_style or os.getenv("S3_ADDRESSING_STYLE", "auto")).strip().lower()
    signature_version = (args.s3_signature_version or os.getenv("S3_SIGNATURE_VERSION", "s3v4")).strip()

    if addressing_style not in {"auto", "path", "virtual"}:
        raise ValueError("--s3_addressing_style must be one of: auto, path, virtual")

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
    return boto3.client(**client_kwargs)


def generate_presigned_pair(s3_client, bucket: str, object_key: str, expires_in: int) -> tuple[str, str]:
    put_url = s3_client.generate_presigned_url(
        ClientMethod="put_object",
        Params={"Bucket": bucket, "Key": object_key},
        ExpiresIn=expires_in,
        HttpMethod="PUT",
    )
    get_url = s3_client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": object_key},
        ExpiresIn=expires_in,
        HttpMethod="GET",
    )
    return put_url, get_url


def build_sync_payload(args: argparse.Namespace, presigned_url: str = "") -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "infer_steps": args.infer_steps,
        "seed": args.seed,
        "aspect_ratio": args.aspect_ratio,
        "save_result_path": args.save_result_path,
        "use_prompt_enhancer": args.use_prompt_enhancer,
    }
    if args.target_shape:
        payload["target_shape"] = args.target_shape
    if presigned_url:
        payload["presigned_url"] = presigned_url
    return payload


def call_sync(base_url: str, payload: Dict[str, Any], timeout_seconds: int, poll_interval_seconds: float) -> requests.Response:
    endpoint = f"{base_url.rstrip('/')}/v1/tasks/image/sync?timeout_seconds={timeout_seconds}&poll_interval_seconds={poll_interval_seconds}"
    return requests.post(endpoint, json=payload, timeout=timeout_seconds + 30)


def run_client_upload_flow(
    args: argparse.Namespace,
    s3_client,
    bucket: str,
    object_key: str,
) -> Dict[str, float]:
    put_url, get_url = generate_presigned_pair(s3_client, bucket, object_key, args.presign_expires)

    payload = build_sync_payload(args, presigned_url="")
    t0 = time.perf_counter()
    response = call_sync(args.url, payload, args.timeout_seconds, args.poll_interval_seconds)
    t1 = time.perf_counter()
    if response.status_code != 200:
        raise RuntimeError(f"[client_upload] sync failed ({response.status_code}): {response.text}")
    image_bytes = response.content
    t2 = time.perf_counter()
    upload_resp = requests.put(put_url, data=image_bytes, timeout=args.upload_timeout_seconds)
    t3 = time.perf_counter()
    if upload_resp.status_code not in (200, 201, 204):
        raise RuntimeError(f"[client_upload] upload failed ({upload_resp.status_code}): {upload_resp.text}")

    if args.verify_download:
        check_resp = requests.get(get_url, timeout=args.download_timeout_seconds)
        if check_resp.status_code != 200:
            raise RuntimeError(f"[client_upload] download verify failed ({check_resp.status_code}): {check_resp.text}")

    return {
        "sync_ms": (t1 - t0) * 1000.0,
        "upload_ms": (t3 - t2) * 1000.0,
        "total_ms": (t3 - t0) * 1000.0,
        "bytes": float(len(image_bytes)),
    }


def run_server_upload_flow(
    args: argparse.Namespace,
    s3_client,
    bucket: str,
    object_key: str,
) -> Dict[str, float]:
    put_url, get_url = generate_presigned_pair(s3_client, bucket, object_key, args.presign_expires)
    payload = build_sync_payload(args, presigned_url=put_url)

    t0 = time.perf_counter()
    response = call_sync(args.url, payload, args.timeout_seconds, args.poll_interval_seconds)
    t1 = time.perf_counter()
    if response.status_code != 200:
        raise RuntimeError(f"[server_upload] sync failed ({response.status_code}): {response.text}")

    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type:
        raise RuntimeError(f"[server_upload] expected JSON but got content-type={content_type!r}")
    body = response.json()
    if not body.get("uploaded_to_presigned_url"):
        raise RuntimeError(f"[server_upload] upload flag is false, response={json.dumps(body, ensure_ascii=False)}")

    if args.verify_download:
        check_resp = requests.get(get_url, timeout=args.download_timeout_seconds)
        if check_resp.status_code != 200:
            raise RuntimeError(f"[server_upload] download verify failed ({check_resp.status_code}): {check_resp.text}")

    return {
        "sync_total_ms": (t1 - t0) * 1000.0,
    }


def print_summary(label: str, values: List[float]) -> None:
    print(f"{label}: avg={statistics.mean(values):.2f} ms, p50={percentile(values, 0.5):.2f} ms, p90={percentile(values, 0.9):.2f} ms, min={min(values):.2f} ms, max={max(values):.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sync latency: client upload to S3 vs x2v server upload to S3.")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="x2v server base url")
    parser.add_argument("--runs", type=int, default=5, help="Benchmark rounds")
    parser.add_argument("--warmup_runs", type=int, default=0, help="Warmup rounds before measurement")
    parser.add_argument("--order", type=str, default="alternate", choices=["alternate", "client_first", "server_first"])

    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt text")
    parser.add_argument("--infer_steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--aspect_ratio", type=str, default="16:9", help="Aspect ratio")
    parser.add_argument("--target_shape", type=int, nargs="+", default=None, help="Target shape, e.g. 1536 2752")
    parser.add_argument("--save_result_path", type=str, default="", help="Server-side save_result_path")
    parser.add_argument("--use_prompt_enhancer", action="store_true")
    parser.add_argument("--timeout_seconds", type=int, default=600)
    parser.add_argument("--poll_interval_seconds", type=float, default=0.5)

    parser.add_argument("--s3_endpoint_url", type=str, default="")
    parser.add_argument("--s3_region", type=str, default="")
    parser.add_argument("--s3_bucket", type=str, default="")
    parser.add_argument("--s3_access_key", type=str, default="")
    parser.add_argument("--s3_secret_key", type=str, default="")
    parser.add_argument("--s3_session_token", type=str, default="")
    parser.add_argument("--s3_addressing_style", type=str, default="")
    parser.add_argument("--s3_signature_version", type=str, default="")
    parser.add_argument("--presign_expires", type=int, default=3600)
    parser.add_argument(
        "--client_key_prefix",
        type=str,
        default="",
        help="S3 key prefix for client-upload flow, defaults to $S3_BASE_PATH/benchmark/client",
    )
    parser.add_argument(
        "--server_key_prefix",
        type=str,
        default="",
        help="S3 key prefix for server-upload flow, defaults to $S3_BASE_PATH/benchmark/server",
    )
    parser.add_argument("--upload_timeout_seconds", type=int, default=120)
    parser.add_argument("--download_timeout_seconds", type=int, default=120)
    parser.add_argument("--verify_download", action="store_true", help="Verify uploaded object with GET presigned URL")
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be > 0")
    if args.warmup_runs < 0:
        raise ValueError("--warmup_runs must be >= 0")
    if args.presign_expires <= 0:
        raise ValueError("--presign_expires must be > 0")
    if args.target_shape is not None and len(args.target_shape) < 2:
        raise ValueError("--target_shape must provide at least 2 integers")

    bucket = args.s3_bucket or os.getenv("S3_BUCKET", "")
    if not bucket:
        raise ValueError("Missing S3 bucket. Set --s3_bucket or env S3_BUCKET.")
    base_path = os.getenv("S3_BASE_PATH", "lightx2v/sync").strip("/")
    client_prefix = (args.client_key_prefix or f"{base_path}/benchmark/client").strip("/")
    server_prefix = (args.server_key_prefix or f"{base_path}/benchmark/server").strip("/")

    s3_client = build_s3_client(args)

    total_rounds = args.warmup_runs + args.runs
    client_results: List[Dict[str, float]] = []
    server_results: List[Dict[str, float]] = []

    print(f"Start benchmark: warmup={args.warmup_runs}, runs={args.runs}, order={args.order}")
    for idx in range(total_rounds):
        is_warmup = idx < args.warmup_runs
        round_no = idx + 1
        tag = "warmup" if is_warmup else "measure"
        print(f"\nRound {round_no}/{total_rounds} [{tag}]")

        client_key = f"{client_prefix}/{uuid.uuid4().hex}.png"
        server_key = f"{server_prefix}/{uuid.uuid4().hex}.png"

        if args.order == "client_first":
            flow_order = ["client", "server"]
        elif args.order == "server_first":
            flow_order = ["server", "client"]
        else:
            flow_order = ["client", "server"] if (idx % 2 == 0) else ["server", "client"]

        one_round_client = None
        one_round_server = None

        for flow in flow_order:
            if flow == "client":
                one_round_client = run_client_upload_flow(args, s3_client, bucket, client_key)
                print(
                    "[client_upload] "
                    f"sync={one_round_client['sync_ms']:.2f} ms, "
                    f"upload={one_round_client['upload_ms']:.2f} ms, "
                    f"total={one_round_client['total_ms']:.2f} ms, "
                    f"bytes={int(one_round_client['bytes'])}"
                )
            else:
                one_round_server = run_server_upload_flow(args, s3_client, bucket, server_key)
                print(f"[server_upload] total={one_round_server['sync_total_ms']:.2f} ms")

        if not is_warmup:
            assert one_round_client is not None and one_round_server is not None
            client_results.append(one_round_client)
            server_results.append(one_round_server)
            delta_ms = one_round_server["sync_total_ms"] - one_round_client["total_ms"]
            print(f"[delta] server_total - client_total = {delta_ms:.2f} ms")

    client_sync_list = [x["sync_ms"] for x in client_results]
    client_upload_list = [x["upload_ms"] for x in client_results]
    client_total_list = [x["total_ms"] for x in client_results]
    server_total_list = [x["sync_total_ms"] for x in server_results]
    delta_list = [s - c for s, c in zip(server_total_list, client_total_list)]

    print("\n=== Benchmark Summary ===")
    print_summary("client_upload.sync_ms", client_sync_list)
    print_summary("client_upload.upload_ms", client_upload_list)
    print_summary("client_upload.total_ms", client_total_list)
    print_summary("server_upload.total_ms", server_total_list)
    print_summary("delta(server-client).ms", delta_list)


if __name__ == "__main__":
    main()

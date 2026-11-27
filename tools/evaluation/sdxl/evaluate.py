"""Developer evaluation; inference and metric networks remain external dependencies."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "sdxl"


def nvidia_config(name):
    """Keep the original parity profile and BF16 measurements separate."""
    profile = os.environ.get("LIGHTX2V_EVAL_PRECISION_PROFILE", "native")
    if profile not in ("native", "bf16"):
        raise ValueError("LIGHTX2V_EVAL_PRECISION_PROFILE must be native or bf16")
    directory = HERE / "configs/nvidia"
    return (directory / "bf16" if profile == "bf16" else directory) / name


def campaign_backends(args):
    backend = getattr(args, "backend", None)
    if os.environ.get("LIGHTX2V_EVAL_PRECISION_PROFILE") == "bf16" and backend != "lightx2v":
        raise ValueError("The BF16 profile requires campaign --backend lightx2v; native precision is a different experiment")
    return (backend,) if backend else ("native", "lightx2v")


def single_backend_summary(root, dataset, backend):
    """Full quality/latency report, without claiming cross-precision equality."""
    expected, generated = completed_runs(root / backend, dataset, backend)
    quality_path = root / backend / "quality.json"
    provenance = read_json(root / backend / "quality-provenance.json")
    resource = read_json(root / "resources" / f"{backend}-score" / "resources.json")
    if not resource.get("resources_clear") or not resource.get("task_cleanup_verified") or resource.get("returncode") != 0:
        raise ValueError("Quality execution resources or cleanup are unverified")
    if provenance["quality_sha256"] != sha256(quality_path) or provenance["source_lock_sha256"] != sha256(nvidia_config("source-lock.json")):
        raise ValueError("Quality result/source provenance differs")
    output_digest = hashlib.sha256(json.dumps([(item["id"], generated[item["id"]]["sha256"]) for item in expected]).encode()).hexdigest()
    if provenance["outputs_sha256"] != output_digest:
        raise ValueError("Scored output set differs from the complete generated set")
    result = {
        "status": "COMPLETE",
        "model": MODEL,
        "dataset": dataset,
        "backend": backend,
        "precision_profile": os.environ.get("LIGHTX2V_EVAL_PRECISION_PROFILE", "native"),
        "precision_policy": read_json(nvidia_config("runtime.json")).get("precision_policy"),
        "coverage": len(generated),
        "repetitions": 1,
        "gpu": next(iter(generated.values()))["gpu"],
        "source_lock": read_json(nvidia_config("source-lock.json")),
        "quality": read_json(quality_path),
        "latency": latency_summary(generated),
        "resources_clear": True,
        "task_cleanup_verified": True,
        "cross_precision_exact_alignment_claimed": False,
    }
    write_json(root / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def suite():
    return read_json(HERE / "configs/common/suite.json")


def inputs(dataset):
    specification = suite()["datasets"][dataset]
    path = HERE / specification["manifest"]
    if sha256(path) != specification["sha256"]:
        raise ValueError("Input manifest was modified")
    records = rows(path)
    ids = [entry["id"] for entry in records]
    if len(records) != specification["count"] or len(set(ids)) != len(ids):
        raise ValueError("Wrong input coverage or duplicate IDs")
    if any(Path(name).name != name or name in (".", "..") for name in ids):
        raise ValueError("Unsafe sample ID")
    return records


def resolve(paths, key, relative="", allow_symlinks=False):
    root = Path(paths[key]).resolve()
    lexical = Path(os.path.abspath(root / relative))
    result = lexical.resolve()
    if not lexical.is_relative_to(root) or (not allow_symlinks and not result.is_relative_to(root)):
        raise ValueError("Path escapes configured root")
    return result


def verify_source(root, specification):
    if "files" in specification:
        for entry in specification["files"]:
            if sha256(root / entry["path"]) != entry.get("runtime_sha256", entry["sha256"]):
                raise ValueError(f"Source file mismatch: {root / entry['path']}")
        return
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != specification["revision"]:
        raise ValueError(f"Wrong source revision: {root}: {actual}")
    # LFS binaries can appear dirty before git-lfs smudging. Executable tracked
    # source must still exactly match the pinned tree.
    subprocess.run(["git", "-C", str(root), "diff", "--exit-code", "HEAD", "--", "*.py", "*.sh", "*.json", "*.yaml", "*.yml", "*.toml"], check=True, stdout=subprocess.DEVNULL)


def prepare_sources(destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name, specification in read_json(nvidia_config("source-lock.json")).items():
        if "files" in specification:
            import re
            import urllib.request

            root = destination / name
            root.mkdir(exist_ok=True)
            for entry in specification["files"]:
                target = root / entry["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                expected = entry.get("runtime_sha256", entry["sha256"])
                if target.is_file() and sha256(target) == expected:
                    continue
                url = f"https://raw.githubusercontent.com/{specification['repo']}/{specification['revision']}/{entry['path']}"
                content = urllib.request.urlopen(url, timeout=30).read()
                if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                    raise ValueError(f"Downloaded source mismatch: {url}")
                if "runtime_sha256" in entry:
                    source = content.decode()
                    if "numpy" in entry.get("patch", "").lower():
                        source = re.sub(r"np\.float\b", "np.float64", source)
                    else:
                        source = source.replace("np.sqrt(d1 + d2 + d3)", "np.sqrt(np.maximum(d1 + d2 + d3, 0.0))")
                    content = source.encode()
                target.write_bytes(content)
                if sha256(target) != expected:
                    raise ValueError("Compatibility patch differs from lock")
            verify_source(root, specification)
            continue
        if "repository" not in specification:
            continue
        root = destination / name
        if not root.exists():
            root.mkdir()
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "remote", "add", "origin", specification["repository"]], check=True)
            env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
            subprocess.run(["git", "-C", str(root), "fetch", "--depth=1", "origin", specification["revision"]], check=True, env=env)
            subprocess.run(["git", "-C", str(root), "checkout", "--detach", "FETCH_HEAD"], check=True, env=env)
        verify_source(root, specification)
    print(json.dumps({"sources": str(destination), "status": "PREPARED", "note": "Install dependencies and provision locked LFS/weight assets per DATA.md before running."}))


def check_inputs(paths):
    for dataset in suite()["datasets"]:
        records = inputs(dataset)
        for entry in records:
            for name in ("input", "gt", "source"):
                if name not in entry:
                    continue
                path = resolve(paths, "data", entry[name])
                if not path.is_file():
                    raise FileNotFoundError(path)
                expected = entry.get(name + "_sha256")
                if expected and sha256(path) != expected:
                    raise ValueError(f"Corrupt input: {path}")
    for entry in read_json(HERE / "configs/common/weight-lock.json")["files"]:
        # A prepared weight overlay can link a checksum-locked checkpoint in
        # another storage root. Input/output traversal remains strict.
        path = resolve(paths, entry["root"], entry["path"], allow_symlinks=True)
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Weight/asset mismatch: {path}")
    print("All datasets, weights and evaluator assets verified")


def seed_all():
    import numpy as np
    import torch

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    (torch.mlu if os.getenv("PLATFORM") == "cambricon_mlu" else torch.cuda).manual_seed_all(42)


def runtime_device(torch):
    if os.getenv("PLATFORM") == "cambricon_mlu":
        import torch_mlu
        if not torch.mlu.is_available() or torch.mlu.device_count() != 1:
            raise RuntimeError("MLU evaluation requires exactly one visible MLU device")
        return torch.device("mlu")
    return torch.device("cuda")


def synchronize(torch):
    (torch.mlu if os.getenv("PLATFORM") == "cambricon_mlu" else torch.cuda).synchronize()


def generation_environment(gpu):
    """Apply the pinned numerical profile before either CUDA worker starts."""
    env = dict(os.environ)
    profile = read_json(nvidia_config("runtime.json"))["generation_environment"]
    for key, value in profile.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def verify_generation_environment(actual):
    expected = read_json(nvidia_config("runtime.json"))["generation_environment"]
    if actual != expected:
        raise ValueError("Generation environment differs from the pinned runtime profile")


def environment():
    import torch

    versions = {}
    for package in ("torch", "transformers", "diffusers", "numpy", "Pillow", "trimesh", "rembg", "open-clip-torch", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if os.getenv("PLATFORM") == "cambricon_mlu":
        return {"python": sys.executable, "versions": versions, "platform": "cambricon_mlu",
                "device": torch.mlu.get_device_name(0), "physical_card": os.getenv("MLU_PHYSICAL_CARD", "unknown"),
                "torch_mlu": importlib.metadata.version("torch_mlu"), "deterministic": torch.are_deterministic_algorithms_enabled(),
                "cpu_threads": torch.get_num_threads()}
    return {
        "python": sys.executable,
        "versions": versions,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "visible_device": os.environ["CUDA_VISIBLE_DEVICES"],
        "cudnn": torch.backends.cudnn.version(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cpu_threads": torch.get_num_threads(),
        "generation_environment": {key: os.environ.get(key) for key in read_json(nvidia_config("runtime.json"))["generation_environment"]},
    }


def generate(args, paths):
    import torch

    if args.count < 1 or args.start < 0:
        raise ValueError("Invalid chunk")
    device = runtime_device(torch)
    if device.type == "cuda" and (torch.cuda.device_count() != 1 or not os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise RuntimeError("Exactly one explicitly visible GPU is required")
    if device.type == "mlu" and os.environ.get("MLU_VISIBLE_DEVICES") != "0":
        raise RuntimeError("The MLU container must expose exactly one remapped device")
    if device.type == "cuda":
        verify_generation_environment({key: os.environ.get(key) for key in read_json(nvidia_config("runtime.json"))["generation_environment"]})
    torch.set_grad_enabled(False)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    source_lock = read_json(HERE / "configs/mlu/source-lock.json") if device.type == "mlu" else read_json(nvidia_config("source-lock.json"))
    key = "lightx2v" if args.backend == "lightx2v" else "native"
    if key in source_lock and "repository" in source_lock[key]:
        verify_source(Path(paths[key]), source_lock[key])
    elif key in source_lock and "package" in source_lock[key]:
        package = source_lock[key]
        if importlib.metadata.version(package["package"]) != package["version"]:
            raise ValueError("Native package version mismatch")
        distribution = importlib.metadata.distribution(package["package"])
        for entry in package["package_files"]:
            if sha256(distribution.locate_file(entry["path"])) != entry["sha256"]:
                raise ValueError("Modified native package source")
    records = inputs(args.dataset)
    selected = records[args.start : args.start + args.count]
    if not selected:
        raise ValueError("Empty chunk")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "model": MODEL,
        "dataset": args.dataset,
        "backend": args.backend,
        "platform": os.getenv("PLATFORM", "cuda"),
        "start": args.start,
        "count": len(selected),
        "seed": 42,
        "evaluation_sha256": sha256(__file__),
        "suite_sha256": sha256(HERE / "configs/common/suite.json"),
        "source_lock": source_lock,
        "manifest_sha256": suite()["datasets"][args.dataset]["sha256"],
        "weight_lock_sha256": sha256(HERE / "configs/common/weight-lock.json"),
        "paths": paths,
        "timing": suite()["timing"],
        "status": "STARTED",
    }
    write_json(output / "run.json", provenance)
    seed_all()
    load_start = time.perf_counter()
    adapter = load_adapter(args.backend, paths)
    synchronize(torch)
    provenance["load_s"] = time.perf_counter() - load_start
    provenance["environment"] = environment()
    if device.type == "cuda":
        verify_generation_environment(provenance["environment"]["generation_environment"])
    # A separate synthetic/upstream demo input never contributes to dataset coverage.
    warmup = output / "warmup"
    warmup.mkdir()
    seed_all()
    with torch.inference_mode():
        adapter(warmup_input(paths), warmup / ("output." + EXTENSION))
    synchronize(torch)
    failed = 0
    with (output / "samples.jsonl").open("x") as log:
        for index, fixture in enumerate(selected, args.start):
            directory = output / fixture["id"]
            directory.mkdir()
            artifact = directory / ("output." + EXTENSION)
            seed_all()
            synchronize(torch)
            (torch.mlu if device.type == "mlu" else torch.cuda).reset_peak_memory_stats()
            started = time.perf_counter()
            record = {"id": fixture["id"], "index": index, "status": "FAILED"}
            try:
                # Deserialize the request inside the timed boundary; manifest locking,
                # RNG reset and directory creation are harness preparation.
                request = json.loads(json.dumps(fixture))
                if "input" in request:
                    request["input"] = str(resolve(paths, "data", request["input"]))
                with torch.inference_mode():
                    adapter(request, artifact)
                with artifact.open("rb") as stream:
                    os.fsync(stream.fileno())
                synchronize(torch)
                record["e2e_latency_s"] = time.perf_counter() - started
                record["peak_device_memory_bytes"] = (torch.mlu if device.type == "mlu" else torch.cuda).max_memory_allocated()
                record.update(validate_output(artifact, fixture))
                record.update(status="SUCCESS", artifact=str(artifact.relative_to(output)), sha256=sha256(artifact))
            except Exception as error:
                synchronize(torch)
                record.update(error=f"{type(error).__name__}: {error}", elapsed_until_failure_s=time.perf_counter() - started)
                failed += 1
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps({"index": index, "id": fixture["id"], "status": record["status"]}), flush=True)
    provenance.update(status="COMPLETE" if not failed else "FAILED", failed=failed)
    write_json(output / "run.json", provenance)
    close = getattr(adapter, "close", None)
    if close is not None:
        close()
    if failed:
        raise RuntimeError(f"{failed} failed requests; retained in denominator")


def completed_runs(root, dataset, backend, verify_artifacts=True):
    expected = inputs(dataset)
    indexed = {}
    versions = set()
    for metadata_path in sorted(Path(root).glob("chunks/*/run.json")):
        metadata = read_json(metadata_path)
        if metadata["dataset"] != dataset or metadata["backend"] != backend or metadata["model"] != MODEL:
            raise ValueError("Mixed campaigns")
        versions.add((metadata["evaluation_sha256"], metadata["manifest_sha256"], metadata["weight_lock_sha256"], json.dumps(metadata["source_lock"], sort_keys=True)))
        if metadata["status"] != "COMPLETE":
            raise ValueError(f"Unfinished/failed chunk: {metadata_path}")
        if metadata.get("platform") != "cambricon_mlu":
            verify_generation_environment(metadata["environment"].get("generation_environment"))
        source_lock = read_json(HERE / "configs/mlu/source-lock.json") if metadata.get("platform") == "cambricon_mlu" else read_json(nvidia_config("source-lock.json"))
        if metadata["manifest_sha256"] != suite()["datasets"][dataset]["sha256"] or metadata["source_lock"] != source_lock:
            raise ValueError("Execution used a different dataset or source lock")
        if metadata.get("platform") == "cambricon_mlu":
            if metadata.get("backend") != "lightx2v" or not metadata.get("environment", {}).get("physical_card"):
                raise ValueError("MLU run lacks its physical single-card identity")
        else:
            resources = read_json(metadata_path.parent / "resources.json")
            if not resources["resources_clear"] or not resources.get("task_cleanup_verified") or resources.get("returncode") != 0:
                raise ValueError("Unverified resources or cleanup; latency inadmissible")
        output_rows = rows(metadata_path.parent / "samples.jsonl")
        if len(output_rows) != metadata["count"]:
            raise ValueError("Missing chunk records")
        for record in output_rows:
            if record["id"] in indexed or record["status"] != "SUCCESS":
                raise ValueError("Repeated or failed sample")
            path = resolve({"chunk": str(metadata_path.parent)}, "chunk", record["artifact"])
            if verify_artifacts and sha256(path) != record["sha256"]:
                raise ValueError("Modified output artifact")
            record["path"] = str(path)
            record["gpu"] = metadata["environment"].get("physical_card", metadata["environment"].get("visible_device", "unknown"))
            indexed[record["id"]] = record
    if len(versions) != 1 or set(indexed) != {entry["id"] for entry in expected}:
        raise ValueError("Incomplete coverage or mixed source/fixture versions")
    if len({entry["gpu"] for entry in indexed.values()}) != 1:
        raise ValueError("A model-dataset campaign used multiple GPUs")
    return expected, indexed


def latency_summary(records):
    import numpy as np

    values = [entry["e2e_latency_s"] for entry in records.values()]
    if not values or not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("Invalid latency")
    return {"count": len(values), "mean_s": float(np.mean(values)), "median_s": float(np.median(values)), "p95_s": float(np.percentile(values, 95))}


def compare(args):
    import numpy as np

    _, native = completed_runs(args.native_run, args.dataset, "native")
    _, adapted = completed_runs(args.lightx2v_run, args.dataset, "lightx2v")
    if {entry["gpu"] for entry in native.values()} != {entry["gpu"] for entry in adapted.values()}:
        raise ValueError("Native and LightX2V latency must use the same physical GPU")
    differences = []
    for name, item in native.items():
        if not outputs_equal(Path(item["path"]), Path(adapted[name]["path"])):
            differences.append(name)
    native_quality = read_json(args.native_run / "quality.json")
    adapted_quality = read_json(args.lightx2v_run / "quality.json")
    metrics_equal = quality_aligned(native_quality, adapted_quality)
    reference, candidate = latency_summary(native), latency_summary(adapted)
    ratios = {stat: candidate[stat] / reference[stat] for stat in ("mean_s", "median_s", "p95_s")}
    timing_ok = ratios["mean_s"] <= args.mean_ratio and ratios["p95_s"] <= args.p95_ratio
    result = {
        "model": MODEL,
        "dataset": args.dataset,
        "coverage": len(native),
        "repetitions": 1,
        "output_difference_count": len(differences),
        "output_difference_ids": differences,
        "native_quality": native_quality,
        "lightx2v_quality": adapted_quality,
        "quality_aligned": metrics_equal,
        "native_latency": reference,
        "lightx2v_latency": candidate,
        "latency_ratios": ratios,
        "latency_gate": {"mean_ratio_max": args.mean_ratio, "p95_ratio_max": args.p95_ratio},
        "latency_aligned": timing_ok,
        "status": "PASS" if metrics_equal and not differences and timing_ok else "FAIL",
    }
    if not all(np.isfinite(list(ratios.values()))):
        raise ValueError("Non-finite latency ratio")
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


def quality_aligned(first, second):
    """Fixed absolute tolerances only for scorer roundoff; exact outputs checked separately."""
    import math

    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(quality_aligned(first[key], second[key]) for key in first)
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(quality_aligned(a, b) for a, b in zip(first, second))
    if isinstance(first, (float, int)) and isinstance(second, (float, int)):
        return math.isfinite(first) and math.isfinite(second) and math.isclose(first, second, rel_tol=0, abs_tol=1e-5)
    return first == second


def observed_execution(command, env, python, watch_tool, evidence, artifact, gpu):
    """Capture on local storage; archive evidence before publishing admissibility."""

    def execute(evidence):
        """Wire one bounded host task to the existing gpu_watch capture/check CLI."""
        import math
        import signal

        query_timeout = float(os.environ.get("LIGHTX2V_EVAL_RESOURCE_QUERY_TIMEOUT", "2.5"))
        max_gap = float(os.environ.get("LIGHTX2V_EVAL_RESOURCE_MAX_GAP", "3"))
        if not all(math.isfinite(value) and value > 0 for value in (query_timeout, max_gap)) or max_gap < query_timeout + 0.25:
            raise ValueError("Resource max gap must cover the positive query timeout and sampling interval")

        evidence.mkdir(parents=True, exist_ok=False)
        capture_log = (evidence / "capture.log").open("x")
        backend = os.environ.get("LIGHTX2V_EVAL_RESOURCE_BACKEND", "xml")
        resource_python = os.environ.get("LIGHTX2V_EVAL_RESOURCE_PYTHON", python)
        sampler = HERE.parents[1] / "runtime/nvml_capture.py"
        if backend == "nvml":
            source_sampler = sampler
            sampler = evidence / "nvml_capture.py"
            shutil.copy2(source_sampler, sampler)
            if sha256(sampler) != sha256(source_sampler):
                raise ValueError("Resource sampler changed while staging")
            watch_command = [resource_python, str(sampler), "--policy", str(watch_tool)]
        elif backend == "xml":
            watch_command = [python, str(watch_tool)]
        else:
            raise ValueError("Resource backend must be xml or nvml")
        capture = subprocess.Popen(
            [*watch_command, "capture", "--devices", gpu, "--output", str(evidence / "watch"), "--seconds", "3500", "--interval", "0.25", "--query-timeout", str(query_timeout)],
            stdout=capture_log,
            stderr=subprocess.STDOUT,
        )
        process = None
        task = None
        timed_out = False
        completion = evidence / "completion.json"
        release = evidence / "release.json"
        bootstrap = HERE / "scripts/platform/nvidia/worker.py"
        cuda_startup = ["--initialize-cuda"] if len(command) > 2 and command[2] == "run" else []
        inline_score = ["--inline-geneval"] if len(command) > 2 and command[2] == "score" and "--dataset" in command and command[command.index("--dataset") + 1] == "geneval" else []
        worker_command = [command[0], str(bootstrap), "--completion", str(completion), "--release", str(release), *cuda_startup, *inline_score, "--", *command[1:]]

        def latest():
            path = evidence / "watch/samples.jsonl"
            if not path.exists():
                return None
            lines = path.read_text().splitlines()
            if not lines:
                return None
            try:
                return json.loads(lines[-1])
            except json.JSONDecodeError:
                return json.loads(lines[-2]) if len(lines) > 1 else None

        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                point = latest()
                if point:
                    if point.get("error"):
                        raise RuntimeError(point["error"])
                    if all(not device["processes"] and device["utilization_pct"] == 0 and device["memory_mib"] <= 256 for device in point["gpus"]):
                        break
                if capture.poll() is not None:
                    raise RuntimeError("Resource capture stopped before workload")
                time.sleep(0.5)
            else:
                raise RuntimeError("Selected GPU did not become idle; no workload started")
            write_json(
                evidence / "command.json",
                {
                    "argv": command,
                    "worker_argv": worker_command,
                    "bootstrap_sha256": sha256(bootstrap),
                    "resource_capture": {
                        "interval_s": 0.25,
                        "query_timeout_s": query_timeout,
                        "max_gap_s": max_gap,
                        "backend": backend,
                        "sampler_sha256": sha256(sampler) if backend == "nvml" else sha256(watch_tool),
                    },
                    "environment": {
                        key: env.get(key)
                        for key in ("CUDA_VISIBLE_DEVICES", "PLATFORM", "DTYPE", "HF_HUB_OFFLINE", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
                    },
                },
            )
            with (evidence / "worker.log").open("x") as log:
                start = time.monotonic()
                # GNU timeout supervises this exclusive process group. The Python
                # watchdog also observes telemetry and enforces a 3300-second limit.
                process = subprocess.Popen(["timeout", "--kill-after=20s", "3300s", *worker_command], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                ticks = int(Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
                task = {"pid": process.pid, "start_ticks": ticks, "start": start, "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "devices": [gpu]}
                while process.poll() is None:
                    point = latest()
                    if not point or point.get("error") or capture.poll() is not None or time.monotonic() - point["end"] > max_gap:
                        raise RuntimeError("Resource monitoring became unavailable")
                    for device in point["gpus"]:
                        for app in device["processes"]:
                            if app["type"] in ("M", "M+C") or [process.pid, ticks] not in app["ancestors"]:
                                raise RuntimeError("Foreign/MPS GPU process observed; timing invalid")
                    if completion.exists():
                        marker = read_json(completion)
                        worker_fields = Path(f"/proc/{marker['pid']}/stat").read_text().rsplit(")", 1)[1].split()
                        if int(worker_fields[19]) != marker["start_ticks"] or os.getpgid(marker["pid"]) != process.pid:
                            raise RuntimeError("Completion marker does not belong to this live task")
                        if not start < marker["end"] <= time.monotonic():
                            raise RuntimeError("Invalid workload completion time")
                        if point["start"] >= marker["end"]:
                            task["end"] = marker["end"]
                            write_json(release, {"post_workload_sample_end": point["end"]})
                            process.wait(timeout=25)
                            break
                    if time.monotonic() - start > 3330:
                        timed_out = True
                        raise RuntimeError("Task deadline exceeded")
                    time.sleep(0.2)
                if "end" not in task:
                    raise RuntimeError("Worker exited without an acknowledged completion boundary")
            time.sleep(2)
        finally:
            if process is not None:
                # All members of this newly created process group belong to this
                # task; never signal a GPU PID obtained from telemetry.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    time.sleep(0.2)
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=25)
            if capture.poll() is None:
                capture.terminate()
            capture.wait(timeout=20)
            capture_log.close()
            if task is not None:
                task.setdefault("end", time.monotonic())
                write_json(evidence / "task.json", task)
        checked = subprocess.run(
            [*watch_command, "check", "--samples", str(evidence / "watch/samples.jsonl"), "--task", str(evidence / "task.json"), "--max-gap", str(max_gap)], text=True, capture_output=True, timeout=30
        )
        result = json.loads(checked.stdout)
        try:
            os.killpg(process.pid, 0)
            clean = False
        except ProcessLookupError:
            clean = True
        result.update(
            task_cleanup_verified=clean,
            returncode=process.returncode,
            timed_out=timed_out,
            completion_handshake=True,
            bootstrap_sha256=sha256(bootstrap),
            resource_capture={
                "interval_s": 0.25,
                "query_timeout_s": query_timeout,
                "max_gap_s": max_gap,
                "backend": backend,
                "sampler_sha256": sha256(sampler) if backend == "nvml" else sha256(watch_tool),
            },
        )
        write_json(evidence / "resources.json", result)
        if checked.returncode or process.returncode or not clean:
            raise RuntimeError(f"Invalid execution: {result}; see {evidence}")

    import shutil
    import tempfile

    destination = Path(evidence)
    destination.mkdir(parents=True, exist_ok=False)
    scratch = Path(os.environ.get("LIGHTX2V_EVAL_RESOURCE_SCRATCH", "/tmp"))
    scratch.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"lightx2v-{MODEL}-watch-", dir=str(scratch)))
    local = staging / "evidence"
    manifest = {"state": "RUNNING", "directory": str(local), "storage": "local_capture_then_archive", "controller_pid": os.getpid()}
    write_json(destination / "staging.json", manifest)
    try:
        execute(local)
    finally:
        try:
            if local.exists():
                shutil.copytree(local, destination, dirs_exist_ok=True)
                # A completed chunk becomes resumable only after its original
                # resource evidence has reached the configured result storage.
                if (local / "resources.json").exists() and artifact.exists():
                    write_json(artifact / "resources.json", read_json(local / "resources.json"))
            write_json(destination / "staging.json", {**manifest, "state": "ARCHIVED"})
        except BaseException:
            print(f"Local resource evidence retained at {local}", file=sys.stderr)
            raise
        else:
            shutil.rmtree(staging)


def checked_worker_script(paths):
    """Resume archived measurement code only when worker logic/configs are identical.

    Controller fixes may change process supervision without changing any request,
    math, timer, export, validation or metric function. Old run fingerprints are
    retained; this never relabels their measurements as a new worker version.
    """
    import ast

    worker = Path(paths.get("worker_script", str(HERE / "evaluate.py"))).resolve()
    if worker == (HERE / "evaluate.py").resolve():
        return worker
    excluded = {"observed_execution", "campaign", "checked_worker_script", "main"}

    def contract(path):
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in excluded]
        return ast.dump(tree, include_attributes=False)

    if contract(worker) != contract(HERE / "evaluate.py"):
        raise ValueError("Archived worker changes request, timing or metric behavior")
    for local in (HERE / "configs").rglob("*"):
        if local.is_file() and local.suffix in (".json", ".jsonl"):
            archived = worker.parent / local.relative_to(HERE)
            if not archived.is_file() or sha256(archived) != sha256(local):
                raise ValueError("Archived worker configuration differs")
    return worker


def campaign(args, paths):
    import uuid

    backends = campaign_backends(args)

    if not args.gpu or not args.gpu.startswith("GPU-") or str(uuid.UUID(args.gpu[4:])) != args.gpu[4:]:
        raise ValueError("Pass exactly one full GPU UUID")
    if not 1 <= args.count <= 256:
        raise ValueError("Chunk size must be in [1,256]")
    watch_tool = Path(paths["lightx2v"]) / "tools/runtime/gpu_watch.py"
    verify_source(Path(paths["lightx2v"]), read_json(nvidia_config("source-lock.json"))["lightx2v"])
    records = inputs(args.dataset)
    count = len(records)
    check_inputs(paths)
    worker_script = checked_worker_script(paths)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "model": MODEL,
        "dataset": args.dataset,
        "gpu": args.gpu,
        "count": count,
        "chunk_size": args.count,
        "evaluate_sha256": sha256(worker_script),
        "paths": {key: value for key, value in paths.items() if key != "worker_script"},
    }
    if len(backends) == 1:
        identity["backends"] = list(backends)
        identity["precision_profile"] = os.environ.get("LIGHTX2V_EVAL_PRECISION_PROFILE", "native")
    identity_path = root / "campaign.json"
    if identity_path.exists():
        previous = read_json(identity_path)
        if {k: v for k, v in previous.items() if k != "chunk_size"} != {k: v for k, v in identity.items() if k != "chunk_size"}:
            raise ValueError("Resume identity differs; preserve old evidence and use a new campaign directory")
    else:
        write_json(identity_path, identity)
    # Chunk size bounds process lifetime, not the per-input request or timer.
    # Preserve the first plan and log each subsequent controller invocation.
    write_json(root / f"controller-plan-{time.time_ns()}.json", {"controller_sha256": sha256(__file__), "requested_chunk_size": args.count, "worker_sha256": sha256(worker_script)})
    env = generation_environment(args.gpu)
    covered = {backend: set() for backend in backends}
    for backend in covered:
        for chunk in sorted((root / backend / "chunks").glob("*")):
            metadata = read_json(chunk / "run.json")
            resource = read_json(chunk / "resources.json")
            if metadata["status"] != "COMPLETE" or not resource["resources_clear"] or not resource.get("task_cleanup_verified") or resource.get("returncode") != 0:
                raise ValueError(f"Existing incomplete/invalid chunk requires investigation: {chunk}")
            begin, length = metadata["start"], metadata["count"]
            if not (type(begin) is int and type(length) is int and 0 <= begin < count and 1 <= length <= 256 and begin + length <= count):
                raise ValueError(f"Invalid recorded chunk range: {chunk}")
            if chunk.name != f"{begin:06d}" or metadata["model"] != MODEL or metadata["dataset"] != args.dataset or metadata["backend"] != backend:
                raise ValueError(f"Mixed recorded campaign: {chunk}")
            if (
                metadata["evaluation_sha256"] != sha256(worker_script)
                or metadata["source_lock"] != read_json(nvidia_config("source-lock.json"))
                or metadata["manifest_sha256"] != suite()["datasets"][args.dataset]["sha256"]
            ):
                raise ValueError(f"Recorded worker/source/fixture differs: {chunk}")
            logged = rows(chunk / "samples.jsonl")
            if [row["id"] for row in logged] != [row["id"] for row in records[begin : begin + length]] or any(row["status"] != "SUCCESS" for row in logged):
                raise ValueError(f"Incomplete/failed recorded inputs: {chunk}")
            indices = set(range(begin, begin + length))
            if covered[backend] & indices:
                raise ValueError(f"Overlapping recorded chunks: {chunk}")
            covered[backend].update(indices)
    for start in range(0, count, args.count):
        stop = min(count, start + args.count)
        for backend in backends:
            cursor = start
            while cursor < stop:
                if cursor in covered[backend]:
                    cursor += 1
                    continue
                begin = cursor
                while cursor < stop and cursor not in covered[backend]:
                    cursor += 1
                chunk = root / backend / "chunks" / f"{begin:06d}"
                command = [
                    paths[backend + "_python"],
                    str(worker_script),
                    "run",
                    "--paths",
                    str(args.paths.resolve()),
                    "--dataset",
                    args.dataset,
                    "--backend",
                    backend,
                    "--output",
                    str(chunk),
                    "--start",
                    str(begin),
                    "--count",
                    str(cursor - begin),
                ]
                observed_execution(command, env, sys.executable, watch_tool, root / "resources" / f"{backend}-{begin:06d}", chunk, args.gpu)
                covered[backend].update(range(begin, cursor))
    if args.skip_score:
        return
    for backend in backends:
        output = root / backend
        if (output / "quality.json").exists():
            continue
        metric_python = paths["geneval_python"] if args.dataset == "geneval" else paths["metric_python"]
        command = [metric_python, str(worker_script), "score", "--paths", str(args.paths.resolve()), "--dataset", args.dataset, "--backend", backend, "--run", str(output)]
        observed_execution(command, env, sys.executable, watch_tool, root / "resources" / f"{backend}-score", output, args.gpu)
    if len(backends) == 1:
        single_backend_summary(root, args.dataset, backends[0])
        return
    args.native_run, args.lightx2v_run = root / "native", root / "lightx2v"
    args.output = root / "comparison.json"
    compare(args)


def campaign_mlu(args, paths):
    import argparse as _argparse
    if os.getenv("PLATFORM") != "cambricon_mlu" or args.backend != "lightx2v":
        raise RuntimeError("MLU campaign requires PLATFORM=cambricon_mlu and backend=lightx2v")
    if not 1 <= args.count <= 256:
        raise ValueError("Chunk size must be in [1,256]")
    check_inputs(paths)
    records = inputs(args.dataset)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {"model": MODEL, "dataset": args.dataset, "count": len(records), "chunk_size": args.count,
                "platform": "cambricon_mlu", "physical_card": os.getenv("MLU_PHYSICAL_CARD", "unknown"),
                "evaluation_sha256": sha256(__file__), "manifest_sha256": suite()["datasets"][args.dataset]["sha256"],
                "source_lock": read_json(HERE / "configs/mlu/source-lock.json"), "paths": paths}
    identity_path = root / "campaign.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("Resume identity differs; use a new output directory")
    if not identity_path.exists():
        write_json(identity_path, identity)
    chunks = root / "chunks"
    chunks.mkdir(exist_ok=True)
    for start in range(0, len(records), args.count):
        stop = min(len(records), start + args.count)
        chunk = chunks / f"{start:06d}"
        if (chunk / "run.json").exists():
            metadata = read_json(chunk / "run.json")
            if metadata.get("status") != "COMPLETE" or metadata.get("start") != start or metadata.get("count") != stop-start:
                raise ValueError(f"Existing incomplete chunk requires investigation: {chunk}")
            continue
        generate(_argparse.Namespace(dataset=args.dataset, backend="lightx2v", start=start, count=stop-start, output=chunk), paths)
        print(json.dumps({"dataset": args.dataset, "completed": stop, "total": len(records)}), flush=True)
    expected, generated = completed_runs(root, args.dataset, "lightx2v")
    quality = score(expected, generated, paths, root)
    latency = latency_summary(generated)
    write_json(root / "quality.json", quality)
    write_json(root / "latency.json", latency)
    write_json(root / "quality-provenance.json", {"evaluation_sha256": sha256(__file__),
        "source_lock_sha256": sha256(HERE / "configs/mlu/source-lock.json"),
        "weight_lock_sha256": sha256(HERE / "configs/common/weight-lock.json"),
        "manifest_sha256": suite()["datasets"][args.dataset]["sha256"],
        "quality_sha256": sha256(root / "quality.json"),
        "outputs_sha256": hashlib.sha256(json.dumps([(item["id"], generated[item["id"]]["sha256"]) for item in expected]).encode()).hexdigest(),
        "physical_card": os.getenv("MLU_PHYSICAL_CARD", "unknown")})
    write_json(root / "campaign.json", {**identity, "status": "COMPLETE", "quality": quality, "latency": latency})


EXTENSION = "png"


def warmup_input(paths):
    return {"prompt": "A small red wooden cube on a plain gray background."}


def load_adapter(backend, paths):
    import torch

    if backend == "lightx2v":
        if os.getenv("PLATFORM") == "cambricon_mlu" and not os.getenv("MLU_PHYSICAL_CARD"):
            raise ValueError("Set MLU_PHYSICAL_CARD to the assigned host card")
        sys.path.insert(0, paths["lightx2v"])
        from lightx2v.models.runners.runner_factory import build_runner
        from lightx2v.utils.set_config import build_startup_config

        config = read_json(HERE / "configs/mlu/model.json") if os.getenv("PLATFORM") == "cambricon_mlu" else read_json(nvidia_config("model.json"))
        config["model_path"] = paths["weights"]
        runner = build_runner(build_startup_config(config))

        def infer(item, artifact):
            request = dict(prompt=item["prompt"], seed=42, size=[1024, 1024], save_result_path=str(artifact))
            runner.run_request(runner.prepare_request(request))

        infer.close = runner.close
        return infer
    from diffusers import StableDiffusionXLPipeline

    if importlib.metadata.version("diffusers") != "0.40.0":
        raise ValueError("Native Diffusers must match pinned 0.40.0")
    device = runtime_device(torch)
    pipeline = StableDiffusionXLPipeline.from_pretrained(paths["weights"], variant="fp16", torch_dtype=torch.float16, use_safetensors=True, local_files_only=True).to(device)
    pipeline.set_progress_bar_config(disable=True)

    def require_finite(module, inputs, output):
        if not torch.isfinite(output).all():
            raise FloatingPointError("Native SDXL decoded non-finite values")

    pipeline.vae.decoder.register_forward_hook(require_finite)

    def infer(item, artifact):
        image = pipeline(prompt=item["prompt"], num_inference_steps=30, guidance_scale=5.0, generator=torch.Generator(device=device).manual_seed(42), height=1024, width=1024).images[0]
        image.save(artifact)

    return infer


def validate_output(path, fixture):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"))
        if image.size != (1024, 1024) or not np.isfinite(pixels).all():
            raise ValueError("Invalid SDXL image")
    return {"finite_serialized_output": True, "size": [1024, 1024]}


def outputs_equal(first, second):
    return sha256(first) == sha256(second)


def score(expected, generated, paths, output):
    if "metadata" in expected[0] and "tag" in expected[0]["metadata"]:
        directory = output / "geneval"
        directory.mkdir(exist_ok=False)
        for item in expected:
            sample = directory / item["id"] / "samples"
            sample.mkdir(parents=True)
            (sample.parent / "metadata.jsonl").write_text(json.dumps(item["metadata"]) + "\n")
            (sample / "0.png").symlink_to(Path(generated[item["id"]]["path"]))
        report = output / "geneval-scores.jsonl"
        repo = Path(paths["geneval"])
        geneval_lock = (read_json(HERE / "configs/mlu/source-lock.json") if os.getenv("PLATFORM") == "cambricon_mlu" else read_json(nvidia_config("source-lock.json")))["geneval"]
        if os.getenv("PLATFORM") == "cambricon_mlu":
            actual_revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            if actual_revision != geneval_lock["revision"]:
                raise ValueError("Geneval source revision differs from its MLU lock")
        verify_source(repo, geneval_lock)
        # The official CLI has no CLIP cache option. Route this constructor's
        # existing cache_dir parameter without changing evaluator mathematics.
        launcher = """import functools, inspect, runpy, sys, open_clip
cache, script, *arguments = sys.argv[1:]
create_model = open_clip.create_model_and_transforms
def create_model_compat(*args, **kwargs):
    if "weights_only" not in inspect.signature(create_model).parameters:
        kwargs.pop("weights_only", None)
    return create_model(*args, **kwargs)
open_clip.create_model_and_transforms = functools.partial(create_model_compat, cache_dir=cache)
sys.argv = [script, *arguments]
runpy.run_path(script, run_name='__main__')
"""
        geneval_env = dict(os.environ)
        geneval_env["PYTHONPATH"] = os.pathsep.join(filter(None, [
            paths["geneval_configs"], paths.get("geneval_vendor", ""), geneval_env.get("PYTHONPATH", "")
        ]))
        subprocess.run(
            [
                paths["geneval_python"],
                "-c",
                launcher,
                paths["geneval_clip"],
                str(repo / "evaluation/evaluate_images.py"),
                str(directory.resolve()),
                "--outfile",
                str(report.resolve()),
                "--model-path",
                paths["geneval_weights"],
            "--model-config",
                paths["geneval_config"],
            "--device", "mlu" if os.getenv("PLATFORM") == "cambricon_mlu" else "cuda",
            ],
            check=True,
            timeout=3300,
            env=geneval_env,
        )
        entries = rows(report)
        if len(entries) != len(expected) or len({entry["filename"] for entry in entries}) != len(expected):
            raise ValueError("Incomplete or repeated GenEval scores")
        groups = {}
        for entry in entries:
            groups.setdefault(entry["tag"], []).append(bool(entry["correct"]))
        if set(groups) != {"single_object", "two_object", "counting", "colors", "position", "color_attr"}:
            raise ValueError("Missing GenEval dimensions")
        dimensions = {key: sum(values) / len(values) for key, values in sorted(groups.items())}
        return {"count": len(expected), "geneval_by_dimension": dimensions, "geneval_overall": sum(dimensions.values()) / 6}
    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    device = runtime_device(torch)
    model = CLIPModel.from_pretrained(paths["clip"], torch_dtype=torch.float32, local_files_only=True).to(device).eval()
    processor = CLIPProcessor.from_pretrained(paths["clip"], local_files_only=True)
    values = []
    with (output / "clip-scores.jsonl").open("x") as log, torch.inference_mode():
        for item in expected:
            with Image.open(generated[item["id"]]["path"]) as image:
                tensors = processor(text=[item["prompt"]], images=[image.convert("RGB")], return_tensors="pt", padding=True, truncation=True, max_length=77)
            result = model(**{key: value.to(device) for key, value in tensors.items()})
            cosine = float((result.image_embeds.float() * result.text_embeds.float()).sum(-1).item())
            if not np.isfinite(cosine):
                raise ValueError("Non-finite CLIP score")
            values.append(cosine)
            log.write(json.dumps({"id": item["id"], "clip_cosine_x100": 100 * cosine, "clipscore_2_5_relu_cosine": 2.5 * max(0, cosine)}) + "\n")
    return {"count": len(values), "clip_cosine_x100_mean": float(np.mean(values) * 100), "clipscore_2_5_relu_cosine_mean": float(np.mean(np.maximum(values, 0)) * 2.5)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare-sources", "check-inputs", "run", "score", "compare", "campaign", "campaign-mlu"])
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--dataset", choices=list(suite()["datasets"]))
    parser.add_argument("--backend", choices=["native", "lightx2v"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--native-run", type=Path)
    parser.add_argument("--lightx2v-run", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--mean-ratio", type=float, default=1.25)
    parser.add_argument("--p95-ratio", type=float, default=1.5)
    parser.add_argument("--gpu")
    parser.add_argument("--skip-score", action="store_true")
    args = parser.parse_args()
    if os.environ.get("LIGHTX2V_EVAL_PRECISION_PROFILE") == "bf16":
        if args.command == "compare" or args.backend == "native":
            raise ValueError("BF16 results cannot be presented as native-precision exact parity")
    if args.command == "prepare-sources":
        prepare_sources(args.workspace.resolve())
        return
    if args.command == "compare":
        compare(args)
        return
    paths = read_json(args.paths)
    if args.command in ("run", "score") and "worker_script" in paths:
        import runpy

        worker = checked_worker_script(paths)
        if worker != Path(__file__).resolve():
            sys.argv = [str(worker), *sys.argv[1:]]
            runpy.run_path(str(worker), run_name="__main__")
            return
    if args.command == "check-inputs":
        check_inputs(paths)
    elif args.command == "campaign":
        campaign(args, paths)
    elif args.command == "campaign-mlu":
        campaign_mlu(args, paths)
    elif args.command == "run":
        generate(args, paths)
    elif args.command == "score":
        expected, generated = completed_runs(args.run, args.dataset, args.backend)
        quality = score(expected, generated, paths, args.run)
        write_json(args.run / "quality.json", quality)
        write_json(
            args.run / "quality-provenance.json",
            {
                "evaluation_sha256": sha256(__file__),
                "source_lock_sha256": sha256(nvidia_config("source-lock.json")),
                "weight_lock_sha256": sha256(HERE / "configs/common/weight-lock.json"),
                "manifest_sha256": suite()["datasets"][args.dataset]["sha256"],
                "quality_sha256": sha256(args.run / "quality.json"),
                "outputs_sha256": hashlib.sha256(json.dumps([(item["id"], generated[item["id"]]["sha256"]) for item in expected]).encode()).hexdigest(),
                "python": sys.executable,
            },
        )
        write_json(args.run / "latency.json", latency_summary(generated))


if __name__ == "__main__":
    main()

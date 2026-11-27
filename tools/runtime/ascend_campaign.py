"""Run complete frozen evaluation suites with Ascend host resource evidence."""

import hashlib
import json
import os
import shlex
import subprocess
import time
import uuid
from pathlib import Path

from ascend_capture import verify


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def next_point(watch, previous_end=0, allow_errors=False):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        latest = watch / "latest.json"
        if latest.exists():
            point = read_json(latest)
            if point["end"] > previous_end:
                if point.get("error") and not allow_errors:
                    raise RuntimeError(point["error"])
                return point
        time.sleep(0.05)
    raise TimeoutError("Ascend observer stopped producing samples")


def run_host(evaluation, paths, runtime, python, card, command, evidence):
    import signal
    from ascend_capture import deployment
    from gpu_watch import lineage

    evidence.mkdir(parents=True, exist_ok=False)
    watch = evidence / "watch"
    watch.mkdir()
    write_json(evidence / "deployment.json", deployment(card))
    completion, release = evidence / "completion.json", evidence / "release"
    worker = Path(__file__).with_name("ascend_worker.py").resolve()
    pythonpath = os.pathsep.join(filter(None, [str(worker.parent), str(Path(paths["lightx2v"]) / "tools/runtime"), paths["lightx2v"], paths.get("metric_pythonpath", "")]))
    env = dict(os.environ, PLATFORM="ascend_npu", DTYPE="BF16", ASCEND_RT_VISIBLE_DEVICES=str(card),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               TOKENIZERS_PARALLELISM="false", PROFILING_DEBUG_LEVEL="0", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               OPENBLAS_NUM_THREADS="1", PYTHONPATH=pythonpath)
    observer_command = [python, str(worker.with_name("ascend_capture.py")), "--card", str(card), "--output", str(watch),
                        "--pid-file", str(evidence / "task-pid.json"), "--interval", str(runtime["resource_interval_s"]),
                        "--query-timeout", str(runtime["resource_query_timeout_s"])]
    result = dict(resources_clear=False, task_cleanup_verified=False, command=command,
                  observer_command=observer_command, physical_card=card)
    observer = process = None
    root = None
    previous_end = 0
    with (evidence / "worker.log").open("xb") as log, (evidence / "observer.log").open("xb") as observer_log:
        try:
            observer = subprocess.Popen(observer_command, env=env, stdout=observer_log, stderr=subprocess.STDOUT)
            first = next_point(watch)
            previous_end = first["end"]
            device = first["gpus"][0]
            if device["processes"] or device["utilization_pct"] or device["memory_mib"] > runtime["idle_memory_mib_max"]:
                raise RuntimeError("Selected Ascend card is occupied")
            env["ASCEND_PHYSICAL_CARD"] = device["uuid"]
            result["environment"] = {key: env[key] for key in ("PLATFORM", "DTYPE", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_PHYSICAL_CARD", "HF_HUB_OFFLINE", "OMP_NUM_THREADS", "PYTHONPATH")}
            launch = ["timeout", "--signal=TERM", "--kill-after=20", str(runtime["execution_timeout_seconds"]), python,
                      str(worker), "--initialize-npu", "--completion", str(completion), "--release", str(release), "--", *command]
            result["launch_command"] = launch
            start = time.monotonic()
            process = subprocess.Popen(launch, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            root = lineage(process.pid)[0]
            result["worker_host_identity"] = root
            write_json(evidence / "task-pid.json", {"pid": root[0]})
            deadline = start + runtime["execution_timeout_seconds"] + 25
            while time.monotonic() < deadline:
                point = next_point(watch, previous_end)
                if point["end"] - previous_end > runtime["resource_max_gap_s"]:
                    raise RuntimeError("Ascend resource observation gap")
                previous_end = point["end"]
                foreign = [p["pid"] for d in point["gpus"] for p in d["processes"] if root not in p["ancestors"]]
                if foreign:
                    raise RuntimeError(f"Foreign NPU processes: {foreign}")
                if completion.exists():
                    marker = read_json(completion)
                    if root not in lineage(marker["pid"]) or lineage(marker["pid"])[0][1] != marker["start_ticks"]:
                        raise ValueError("Completion PID identity differs")
                    if point["start"] >= marker["end"]:
                        task = dict(pid=root[0], start_ticks=root[1], start=start, end=marker["end"], boot_id=point["boot_id"], devices=[device["uuid"]])
                        write_json(evidence / "task.json", task)
                        result.update(verify(watch / "samples.jsonl", evidence / "task.json", runtime["resource_max_gap_s"], runtime["idle_memory_mib_max"]))
                        if not result["resources_clear"]:
                            raise RuntimeError(str(result))
                        release.touch(exist_ok=False)
                        break
                if process.poll() is not None:
                    raise RuntimeError(f"Worker exited before acknowledgement: {process.returncode}")
            else:
                raise TimeoutError("Ascend worker exceeded its bounded lifetime")
            result["returncode"] = process.wait(timeout=25)
            if result["returncode"] != 0:
                raise RuntimeError(f"Worker failed: {result['returncode']}")
        except BaseException as error:
            result.update(resources_clear=False, error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if process is not None:
                if process.poll() is None:
                    if root is None or lineage(process.pid)[:1] != [root]:
                        result.update(resources_clear=False, error="Cannot verify cleanup PID identity")
                    else:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            if lineage(process.pid)[:1] == [root]:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait(timeout=5)
                result["returncode"] = process.poll()
                cleanup_deadline = time.monotonic() + 15
                while time.monotonic() < cleanup_deadline:
                    point = next_point(watch, previous_end, allow_errors=True)
                    previous_end = point["end"]
                    if not point.get("error") and not point["gpus"][0]["processes"] and not lineage(process.pid):
                        result["task_cleanup_verified"] = True
                        write_json(evidence / "cleanup.json", point)
                        break
            if observer is not None:
                (watch / "stop").touch()
                try:
                    result["observer_returncode"] = observer.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    observer.terminate()
                    result["observer_returncode"] = observer.wait(timeout=5)
                if result["resources_clear"]:
                    if result["observer_returncode"] != 0:
                        result.update(resources_clear=False, error="Ascend observer failed")
                    else:
                        result.update(verify(watch / "samples.jsonl", evidence / "task.json", runtime["resource_max_gap_s"], runtime["idle_memory_mib_max"]))
            write_json(evidence / "resources.json", result)
    if not result["resources_clear"] or not result["task_cleanup_verified"] or result.get("returncode") != 0:
        raise RuntimeError("Ascend resource or process cleanup verification failed")
    return result


def smoke_indices(evaluation, dataset):
    records = evaluation.inputs(dataset)
    if evaluation.MODEL == "hymotion":
        stop = 0
        eligible = 0
        while eligible <= 300:
            eligible += records[stop]["evaluator_eligible"]
            stop += 1
        stop = max(stop, next(i + 1 for i, item in enumerate(records) if item["motion_name"] == "000021"))
        return list(range(stop))
    if dataset == "geneval":
        tags = {}
        for index, item in enumerate(records):
            tags.setdefault(item["metadata"]["tag"], index)
        return sorted(tags.values())
    return [0]


def chunks(indices, count):
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1 or index - start >= count:
            yield start, previous - start + 1
            start = index
        previous = index
    yield start, previous - start + 1


def run_campaign(evaluation, args, paths):
    if os.environ.get("PLATFORM") != "ascend_npu" or args.backend != "lightx2v":
        raise ValueError("Ascend campaign requires its platform and the LightX2V backend")
    if args.card is None or not 0 <= args.card <= 255 or not 1 <= args.count <= 256:
        raise ValueError("Select one physical card and a chunk size in [1,256]")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_relative_to(Path(paths["output_root"]).resolve()):
        raise ValueError("Campaign output must be inside the mounted output_root")
    runtime = evaluation.read_json(evaluation.HERE / "configs/ascend_npu/runtime.json")
    image_id = str(Path(paths["lightx2v_python"]).absolute())
    metric_image_id = str(Path(paths["metric_python"]).absolute())
    evaluation.verify_source(Path(paths["lightx2v"]), evaluation.read_json(evaluation.HERE / "configs/ascend_npu/source-lock.json")["lightx2v"])
    evaluation.check_inputs(paths)
    indices = smoke_indices(evaluation, args.dataset) if args.smoke else list(range(len(evaluation.inputs(args.dataset))))
    identity = {
        "model": evaluation.MODEL,
        "dataset": args.dataset,
        "physical_card": args.card,
        "smoke": args.smoke,
        "indices": indices,
        "evaluation_sha256": digest(evaluation.__file__),
        "runtime_sha256": digest(evaluation.HERE / "configs/ascend_npu/runtime.json"),
        "model_config_sha256": digest(evaluation.HERE / "configs/ascend_npu/model.json"),
        "suite_sha256": digest(evaluation.HERE / "configs/common/suite.json"),
        "weight_lock_sha256": digest(evaluation.HERE / "configs/common/weight-lock.json"),
        "worker_sha256": digest(Path(__file__).with_name("ascend_worker.py")),
        "controller_sha256": digest(__file__),
        "capture_sha256": digest(Path(__file__).with_name("ascend_capture.py")),
        "sources_sha256": digest(Path(__file__).with_name("ascend_sources.py")),
        "platform_files": {str(path.relative_to(evaluation.HERE)): digest(path)
                           for path in sorted((evaluation.HERE / "scripts/platform/ascend_npu").rglob("*"))
                           if path.is_file() and path.suffix in (".py", ".sh", ".txt")},
        "source_lock": evaluation.read_json(evaluation.HERE / "configs/ascend_npu/source-lock.json"),
        "inference_python": image_id,
        "metric_python": metric_image_id,
        "paths": paths,
    }
    campaign_file = root / "campaign.json"
    if campaign_file.exists() and read_json(campaign_file)["identity"] != identity:
        raise ValueError("Resume identity differs; preserve the existing campaign")
    write_json(campaign_file, {"status": "RUNNING", "identity": identity})
    backend = root / "lightx2v"
    backend.mkdir(exist_ok=True)
    if args.smoke:
        write_json(backend / "smoke-plan.json", {"dataset": args.dataset, "indices": indices, "formal_dataset_count": len(evaluation.inputs(args.dataset))})
    for start, count in chunks(indices, args.count):
        directory = backend / "chunks" / f"{start:06d}"
        if (directory / "run.json").exists():
            metadata = read_json(directory / "run.json")
            resources = read_json(directory / "resources.json")
            if (
                metadata["status"] != "COMPLETE"
                or metadata["start"] != start
                or metadata["count"] != count
                or not resources["resources_clear"]
                or not resources["task_cleanup_verified"]
                or resources["returncode"] != 0
            ):
                raise ValueError("An existing incomplete chunk requires investigation")
            continue
        command = [
            str(Path(evaluation.__file__).resolve()),
            "run",
            "--paths",
            str(args.paths.resolve()),
            "--backend",
            "lightx2v",
            "--dataset",
            args.dataset,
            "--start",
            str(start),
            "--count",
            str(count),
            "--output",
            str(directory),
        ]
        evidence = root / "resources" / f"generate-{start:06d}"
        result = run_host(evaluation, paths, runtime, image_id, args.card, command, evidence)
        write_json(directory / "resources.json", result)
        print(json.dumps({"dataset": args.dataset, "completed_through": start + count, "fixture_count": len(indices), "smoke": args.smoke}), flush=True)
    command = [
        str(Path(evaluation.__file__).resolve()),
        "score-ascend-smoke" if args.smoke else "score",
        "--paths",
        str(args.paths.resolve()),
        "--backend",
        "lightx2v",
        "--dataset",
        args.dataset,
        "--run",
        str(backend),
    ]
    evidence = root / "resources" / "score"
    if not (backend / "quality.json").exists():
        run_host(evaluation, paths, runtime, metric_image_id, args.card, command, evidence)
    resources = read_json(evidence / "resources.json")
    if not resources["resources_clear"] or not resources["task_cleanup_verified"] or resources["returncode"] != 0:
        raise ValueError("Quality execution resources are unverified")
    report = {
        "status": "SMOKE_COMPLETE" if args.smoke else "COMPLETE",
        "model": evaluation.MODEL,
        "dataset": args.dataset,
        "platform": "ascend_npu",
        "physical_card": args.card,
        "coverage": len(indices),
        "formal_dataset_count": len(evaluation.inputs(args.dataset)),
        "precision_policy": runtime.get("precision_policy"),
        "source_lock": identity["source_lock"],
        "quality": read_json(backend / "quality.json"),
        "latency": read_json(backend / "latency.json"),
        "resources_clear": True,
        "task_cleanup_verified": True,
        "inference_python": image_id,
    }
    write_json(root / "summary.json", report)
    write_json(campaign_file, {"status": report["status"], "identity": identity})
    print(json.dumps(report, indent=2), flush=True)


def score_smoke(evaluation, args, paths):
    plan = read_json(args.run / "smoke-plan.json")
    expected = [evaluation.inputs(args.dataset)[index] for index in plan["indices"]]
    generated = {}
    for metadata_path in sorted(args.run.glob("chunks/*/run.json")):
        metadata = read_json(metadata_path)
        resource = read_json(metadata_path.parent / "resources.json")
        if metadata["status"] != "COMPLETE" or not resource["resources_clear"] or not resource["task_cleanup_verified"]:
            raise ValueError("Unverified smoke chunk")
        for line in (metadata_path.parent / "samples.jsonl").read_text().splitlines():
            record = json.loads(line)
            artifact = evaluation.resolve({"root": str(metadata_path.parent)}, "root", record["artifact"])
            if record["status"] != "SUCCESS" or record["id"] in generated or evaluation.sha256(artifact) != record["sha256"]:
                raise ValueError("Failed, duplicate or modified smoke output")
            generated[record["id"]] = {**record, "path": str(artifact)}
    if set(generated) != {item["id"] for item in expected}:
        raise ValueError("Incomplete smoke fixture")
    quality = evaluation.score(expected, generated, paths, args.run)
    evaluation.write_json(args.run / "quality.json", quality)
    evaluation.write_json(args.run / "latency.json", evaluation.latency_summary(generated))
    evaluation.write_json(
        args.run / "quality-provenance.json",
        {"scope": "smoke", "coverage": len(expected), "formal_dataset_count": plan["formal_dataset_count"], "evaluation_sha256": evaluation.sha256(evaluation.__file__)},
    )

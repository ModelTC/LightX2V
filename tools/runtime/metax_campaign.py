"""Run complete frozen evaluation suites with MetaX host resource evidence."""

import hashlib
import json
import os
import shlex
import subprocess
import time
import uuid
from pathlib import Path

from metax_capture import verify


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def docker_state(name):
    return json.loads(subprocess.check_output(["docker", "inspect", name], text=True))[0]["State"]


def worker_identity(name, script):
    listing = subprocess.check_output(["docker", "top", name, "-eo", "pid,args"], text=True)
    candidates = []
    for row in listing.splitlines()[1:]:
        pid, command = row.split(maxsplit=1)
        arguments = shlex.split(command)
        if Path(arguments[0]).name.startswith("python") and script in arguments:
            candidates.append(int(pid))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError(f"Cannot identify the dedicated worker PID: {candidates}")
    return candidates[0]


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
    raise TimeoutError("MetaX observer stopped producing samples")


def run_container(evaluation, paths, runtime, image, card, command, evidence):
    evidence.mkdir(parents=True, exist_ok=False)
    watch = evidence / "watch"
    watch.mkdir()
    os.chown(watch, 65534, 65534)
    watch.chmod(0o755)
    completion = evidence / "completion.json"
    release = evidence / "release"
    name = "lightx2v-metax-" + uuid.uuid4().hex[:16]
    observer_name = name + "-observer"
    worker = Path(__file__).with_name("metax_worker.py").resolve()
    runtime_root = str(Path(__file__).resolve().parent)
    observer = [
        "docker",
        "run",
        "-d",
        "--name",
        observer_name,
        "--pid=host",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user=65534:65534",
        "--group-add=0",
        "--device=/dev/mxcd",
        "--device=/dev/dri",
        "-w",
        "/",
        "-v",
        "/usr/bin/mx-smi:/usr/bin/mx-smi:ro",
        "-v",
        "/opt/mxdriver:/opt/mxdriver:ro",
        "-v",
        runtime_root + ":/observer:ro",
        "-v",
        str(evidence) + ":/control:ro",
        "-v",
        str(watch) + ":/evidence",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "--entrypoint",
        "python",
        image,
        "/observer/metax_capture.py",
        "--card",
        str(card),
        "--output",
        "/evidence",
        "--pid-file",
        "/control/task-pid.json",
        "--interval",
        str(runtime["resource_interval_s"]),
        "--query-timeout",
        str(runtime["resource_query_timeout_s"]),
    ]
    result = {
        "resources_clear": False,
        "task_cleanup_verified": False,
        "container": name,
        "observer_container": observer_name,
        "command": command,
        "observer_command": observer,
        "image": image,
        "physical_card": card,
    }
    root = None
    worker_pid = None
    logger = None
    launched = False
    observer_launched = False
    previous_end = 0
    with (evidence / "worker.log").open("xb") as worker_log:
        try:
            subprocess.run(observer, check=True, stdout=subprocess.DEVNULL)
            observer_launched = True
            first = next_point(watch)
            previous_end = first["end"]
            gpu = first["gpus"][0]
            if gpu["processes"] or gpu["utilization_pct"] or gpu["memory_mib"] > runtime["idle_memory_mib_max"]:
                raise RuntimeError("Selected MetaX card is occupied")
            environment = {
                "PLATFORM": "metax_cuda",
                "DTYPE": "BF16",
                "CUDA_VISIBLE_DEVICES": str(card),
                "MACA_VISIBLE_DEVICES": str(card),
                "METAX_PHYSICAL_CARD": gpu["uuid"],
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PROFILING_DEBUG_LEVEL": "0",
                "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "8",
                "OPENBLAS_NUM_THREADS": "1",
                "PYTHONPATH": os.pathsep.join(filter(None, [str(evaluation.HERE.parents[2]), paths["lightx2v"], paths.get("metric_pythonpath", "")])),
            }
            docker = ["docker", "run", "-d", "--name", name, "--log-driver=json-file", "--network=none", "--ipc=host", "--device=/dev/mxcd", "--device=/dev/dri"]
            for mount in paths["mounts"]:
                docker.extend(["-v", mount + ":" + mount + ":ro"])
            docker.extend(["-v", paths["output_root"] + ":" + paths["output_root"]])
            for key, value in environment.items():
                docker.extend(["-e", key + "=" + value])
            docker.extend(
                [
                    "--entrypoint",
                    "/usr/bin/env",
                    image,
                    "-u",
                    "CUBLAS_WORKSPACE_CONFIG",
                    "timeout",
                    "--kill-after=20",
                    "3300",
                    paths.get("python", "python"),
                    str(worker),
                    "--initialize-cuda",
                    "--completion",
                    str(completion),
                    "--release",
                    str(release),
                    "--",
                    *command,
                ]
            )
            start = time.monotonic()
            subprocess.run(docker, check=True, stdout=subprocess.DEVNULL)
            launched = True
            result["docker_command"] = docker
            logger = subprocess.Popen(["docker", "logs", "--follow", name], stdout=worker_log, stderr=subprocess.STDOUT)
            deadline = start + 3325
            while time.monotonic() < deadline:
                state = docker_state(name)
                if worker_pid is None and state["Running"]:
                    worker_pid = worker_identity(name, str(worker))
                    if worker_pid is not None:
                        write_json(evidence / "task-pid.json", {"pid": worker_pid})
                point = next_point(watch, previous_end)
                previous_end = point["end"]
                if root is None and point.get("task_identity"):
                    root = point["task_identity"][0]
                    if root[0] != worker_pid:
                        raise ValueError("Observer returned another worker identity")
                    result["worker_host_identity"] = root
                if root is not None:
                    foreign = [process["pid"] for device in point["gpus"] for process in device["processes"] if root not in process["ancestors"]]
                    if foreign:
                        raise RuntimeError(f"Foreign MetaX GPU processes: {foreign}")
                if completion.exists():
                    marker = read_json(completion)
                    if root is None or marker["start_ticks"] != root[1]:
                        raise ValueError("Completion worker identity differs")
                    if point["start"] >= marker["end"]:
                        task = {"pid": root[0], "start_ticks": root[1], "start": start, "end": marker["end"], "boot_id": point["boot_id"], "devices": [gpu["uuid"]]}
                        write_json(evidence / "task.json", task)
                        resource = verify(watch / "samples.jsonl", evidence / "task.json", runtime["resource_max_gap_s"], runtime["idle_memory_mib_max"])
                        if not resource["resources_clear"]:
                            raise RuntimeError(str(resource))
                        release.touch(exist_ok=False)
                        result.update(resource)
                        break
                if not state["Running"]:
                    raise RuntimeError(f"Worker exited before completion acknowledgement: {state['ExitCode']}")
            else:
                raise TimeoutError("MetaX worker exceeded its bounded lifetime")
            while docker_state(name)["Running"]:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Worker did not finish after acknowledgement")
                time.sleep(0.1)
            result["returncode"] = docker_state(name)["ExitCode"]
            if result["returncode"] != 0:
                raise RuntimeError(f"Worker failed with exit code {result['returncode']}")
        except BaseException as error:
            result.update(resources_clear=False, error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if launched:
                state = docker_state(name)
                if state["Running"]:
                    subprocess.run(["docker", "stop", "--timeout", "10", name], check=True, stdout=subprocess.DEVNULL)
                result["returncode"] = docker_state(name)["ExitCode"]
                if logger is not None:
                    logger.wait(timeout=10)
                if root is not None:
                    cleanup_deadline = time.monotonic() + 10
                    while time.monotonic() < cleanup_deadline:
                        point = next_point(watch, previous_end, allow_errors=True)
                        previous_end = point["end"]
                        if not point.get("error") and point.get("task_identity") != [root]:
                            result["task_cleanup_verified"] = True
                            write_json(evidence / "cleanup.json", point)
                            break
                subprocess.run(["docker", "rm", name], check=True, stdout=subprocess.DEVNULL)
            if observer_launched:
                (watch / "stop").touch()
                stopped = time.monotonic() + 5
                while docker_state(observer_name)["Running"] and time.monotonic() < stopped:
                    time.sleep(0.1)
                if docker_state(observer_name)["Running"]:
                    subprocess.run(["docker", "stop", "--timeout", "5", observer_name], check=True, stdout=subprocess.DEVNULL)
                result["observer_returncode"] = docker_state(observer_name)["ExitCode"]
                with (evidence / "observer.log").open("wb") as log:
                    subprocess.run(["docker", "logs", observer_name], stdout=log, stderr=subprocess.STDOUT, check=True)
                subprocess.run(["docker", "rm", observer_name], check=True, stdout=subprocess.DEVNULL)
                if result["resources_clear"]:
                    # Recheck the closed capture after observer exit; the earlier
                    # check only permits the held worker to finish teardown.
                    if result["observer_returncode"] != 0:
                        result.update(resources_clear=False, error="MetaX observer did not exit successfully")
                    else:
                        result.update(verify(watch / "samples.jsonl", evidence / "task.json", runtime["resource_max_gap_s"], runtime["idle_memory_mib_max"]))
            write_json(evidence / "resources.json", result)
    if not result["resources_clear"] or not result["task_cleanup_verified"]:
        raise RuntimeError("MetaX resource or process cleanup verification failed")
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
    if os.environ.get("PLATFORM") != "metax_cuda" or args.backend != "lightx2v":
        raise ValueError("MetaX campaign requires its platform and the LightX2V backend")
    if args.card is None or not 0 <= args.card <= 255 or not 1 <= args.count <= 256:
        raise ValueError("Select one physical card and a chunk size in [1,256]")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_relative_to(Path(paths["output_root"]).resolve()):
        raise ValueError("Campaign output must be inside the mounted output_root")
    runtime = evaluation.read_json(evaluation.HERE / "configs/metax/runtime.json")
    image = args.image or runtime["image"]
    image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True).strip()
    metric_image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", paths.get("metric_image", image)], text=True).strip()
    evaluation.verify_source(Path(paths["lightx2v"]), evaluation.read_json(evaluation.HERE / "configs/metax/source-lock.json")["lightx2v"])
    evaluation.check_inputs(paths)
    indices = smoke_indices(evaluation, args.dataset) if args.smoke else list(range(len(evaluation.inputs(args.dataset))))
    identity = {
        "model": evaluation.MODEL,
        "dataset": args.dataset,
        "physical_card": args.card,
        "smoke": args.smoke,
        "indices": indices,
        "evaluation_sha256": digest(evaluation.__file__),
        "runtime_sha256": digest(evaluation.HERE / "configs/metax/runtime.json"),
        "model_config_sha256": digest(evaluation.HERE / "configs/metax/model.json"),
        "suite_sha256": digest(evaluation.HERE / "configs/common/suite.json"),
        "weight_lock_sha256": digest(evaluation.HERE / "configs/common/weight-lock.json"),
        "worker_sha256": digest(Path(__file__).with_name("metax_worker.py")),
        "controller_sha256": digest(__file__),
        "capture_sha256": digest(Path(__file__).with_name("metax_capture.py")),
        "source_lock": evaluation.read_json(evaluation.HERE / "configs/metax/source-lock.json"),
        "image_id": image_id,
        "metric_image_id": metric_image_id,
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
        result = run_container(evaluation, paths, runtime, image_id, args.card, command, evidence)
        write_json(directory / "resources.json", result)
        print(json.dumps({"dataset": args.dataset, "completed_through": start + count, "fixture_count": len(indices), "smoke": args.smoke}), flush=True)
    command = [
        str(Path(evaluation.__file__).resolve()),
        "score-metax-smoke" if args.smoke else "score",
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
        run_container(evaluation, paths, runtime, metric_image_id, args.card, command, evidence)
    resources = read_json(evidence / "resources.json")
    if not resources["resources_clear"] or not resources["task_cleanup_verified"] or resources["returncode"] != 0:
        raise ValueError("Quality execution resources are unverified")
    report = {
        "status": "SMOKE_COMPLETE" if args.smoke else "COMPLETE",
        "model": evaluation.MODEL,
        "dataset": args.dataset,
        "platform": "metax_cuda",
        "physical_card": args.card,
        "coverage": len(indices),
        "formal_dataset_count": len(evaluation.inputs(args.dataset)),
        "precision_policy": runtime.get("precision_policy"),
        "source_lock": identity["source_lock"],
        "quality": read_json(backend / "quality.json"),
        "latency": read_json(backend / "latency.json"),
        "resources_clear": True,
        "task_cleanup_verified": True,
        "image_id": image_id,
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

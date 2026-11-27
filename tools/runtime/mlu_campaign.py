"""MLU Docker task execution with host cnmon/PID evidence and bounded cleanup.

The model's evaluate.py owns inputs, generation, scoring and request timing.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from gpu_watch import lineage
from metax_campaign import docker_state, next_point, read_json, worker_identity, write_json, digest, smoke_indices, chunks
from mlu_capture import verify


def run_container(evaluation, paths, runtime, image, card, command, evidence):
    if not 1 <= runtime['execution_timeout_seconds'] <= 3300:
        raise ValueError('MLU worker deadline must be at most 3300 seconds')
    if any(not math.isfinite(runtime[k]) or runtime[k] <= 0 for k in ('resource_interval_s', 'resource_query_timeout_s', 'resource_max_gap_s')):
        raise ValueError('Resource bounds must be finite and positive')
    evidence = Path(evidence).resolve()
    evidence.mkdir(parents=True, exist_ok=False)
    watch = evidence / 'watch'
    watch.mkdir()
    completion, release = evidence / 'completion.json', evidence / 'release'
    worker = Path(__file__).with_name('mlu_worker.py').resolve()
    name = 'lightx2v-mlu-' + uuid.uuid4().hex[:16]
    observer_command = [sys.executable, str(worker.with_name('mlu_capture.py')), '--card', str(card),
                        '--output', str(watch), '--pid-file', str(evidence / 'task-pid.json'),
                        '--interval', str(runtime['resource_interval_s']),
                        '--query-timeout', str(runtime['resource_query_timeout_s'])]
    result = dict(resources_clear=False, task_cleanup_verified=False, container=name,
                  command=command, observer_command=observer_command, image=image, physical_card=card)
    observer = logger = None
    root = worker_pid = None
    launched = False
    previous_end = 0
    with (evidence / 'worker.log').open('xb') as log, (evidence / 'observer.log').open('xb') as observer_log:
        try:
            observer = subprocess.Popen(observer_command, stdout=observer_log, stderr=subprocess.STDOUT)
            first = next_point(watch)
            previous_end = first['end']
            device = first['gpus'][0]
            if device['processes'] or device['utilization_pct'] or device['memory_mib'] > runtime['idle_memory_mib_max']:
                raise RuntimeError('Selected MLU card is occupied')
            env = dict(PLATFORM='cambricon_mlu', DTYPE='BF16', MLU_VISIBLE_DEVICES='0',
                       MLU_PHYSICAL_CARD=str(card), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
                       LIGHTX2V_MLU_RESOURCE_CAPTURE='1', PROFILING_DEBUG_LEVEL='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                       PYTHONPATH=os.pathsep.join(filter(None, [str(worker.parent), str(evaluation.HERE.parents[2]), paths['lightx2v'], paths.get('metric_pythonpath', '')])))
            result['environment'] = env
            devices = ['/dev/cambricon_ctl', f'/dev/cambricon_dev{card}', f'/dev/cambricon_ipcm{card}']
            if Path('/dev/cambricon_gdr').exists(): devices.append('/dev/cambricon_gdr')
            docker = ['docker', 'run', '-d', '--name', name, '--network=none', '--ipc=host', '--shm-size=32g']
            for node in devices: docker += ['--device', node]
            # Preserve host absolute paths for data, source locks and evidence.
            mounts = {str(worker.parent): 'ro', str(evaluation.HERE.parents[2]): 'ro',
                      paths['lightx2v']: 'ro', str(evidence): 'rw'}
            mounts.update(paths.get('mounts', {}))
            for source, mode in mounts.items():
                if not Path(source).is_absolute() or mode not in ('ro', 'rw'):
                    raise ValueError('Mounts must use absolute paths and explicit ro/rw modes')
                docker += ['-v', f'{source}:{source}:{mode}']
            for key, value in env.items(): docker += ['-e', f'{key}={value}']
            docker += ['--entrypoint', '/usr/bin/env', image, 'timeout', '--kill-after=20',
                       str(runtime['execution_timeout_seconds']), paths.get('python', 'python'), str(worker),
                       '--initialize-mlu', '--completion', str(completion), '--release', str(release), '--', *command]
            result['docker_command'] = docker
            start = time.monotonic()
            subprocess.run(docker, check=True, stdout=subprocess.DEVNULL)
            launched = True
            logger = subprocess.Popen(['docker', 'logs', '--follow', name], stdout=log, stderr=subprocess.STDOUT)
            deadline = start + runtime['execution_timeout_seconds'] + 25
            while time.monotonic() < deadline:
                state = docker_state(name)
                if worker_pid is None and state['Running']:
                    worker_pid = worker_identity(name, str(worker))
                    if worker_pid is not None: write_json(evidence / 'task-pid.json', {'pid': worker_pid})
                point = next_point(watch, previous_end)
                if point['end'] - previous_end > runtime['resource_max_gap_s']:
                    raise RuntimeError('MLU resource observation gap')
                previous_end = point['end']
                if root is None and point.get('task_identity'):
                    root = point['task_identity'][0]
                    if root[0] != worker_pid: raise ValueError('Observer worker identity differs')
                    result['worker_host_identity'] = root
                if root is not None:
                    foreign = [p['pid'] for d in point['gpus'] for p in d['processes'] if root not in p['ancestors']]
                    if foreign: raise RuntimeError(f'Foreign MLU processes: {foreign}')
                if completion.exists():
                    marker = read_json(completion)
                    if root is None or marker['start_ticks'] != root[1]: raise ValueError('Completion PID identity differs')
                    if point['start'] >= marker['end']:
                        task = dict(pid=root[0], start_ticks=root[1], start=start, end=marker['end'], boot_id=point['boot_id'], devices=[device['uuid']])
                        write_json(evidence / 'task.json', task)
                        result.update(verify(watch / 'samples.jsonl', evidence / 'task.json', runtime['resource_max_gap_s'], runtime['idle_memory_mib_max']))
                        if not result['resources_clear']: raise RuntimeError(str(result))
                        release.touch(exist_ok=False)
                        break
                if not state['Running']: raise RuntimeError(f"Worker exited before acknowledgement: {state['ExitCode']}")
            else: raise TimeoutError('MLU worker exceeded its bounded lifetime')
            while docker_state(name)['Running']:
                if time.monotonic() >= deadline: raise TimeoutError('Worker did not finish after acknowledgement')
                time.sleep(.1)
            result['returncode'] = docker_state(name)['ExitCode']
            if result['returncode'] != 0: raise RuntimeError(f"Worker failed: {result['returncode']}")
        except BaseException as error:
            result.update(resources_clear=False, error=f'{type(error).__name__}: {error}')
            raise
        finally:
            if launched:
                if docker_state(name)['Running']:
                    subprocess.run(['docker', 'stop', '--timeout', '10', name], check=True, stdout=subprocess.DEVNULL)
                result['returncode'] = docker_state(name)['ExitCode']
                if logger is not None: logger.wait(timeout=10)
                if root is not None:
                    cleanup_deadline = time.monotonic() + 10
                    while time.monotonic() < cleanup_deadline:
                        point = next_point(watch, previous_end, allow_errors=True)
                        previous_end = point['end']
                        if not point.get('error') and not lineage(root[0]):
                            result['task_cleanup_verified'] = True
                            write_json(evidence / 'cleanup.json', point)
                            break
                subprocess.run(['docker', 'rm', name], check=True, stdout=subprocess.DEVNULL)
            if observer is not None:
                (watch / 'stop').touch()
                try: result['observer_returncode'] = observer.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    observer.terminate(); result['observer_returncode'] = observer.wait(timeout=5)
                if result['resources_clear']:
                    if result['observer_returncode'] != 0:
                        result.update(resources_clear=False, error='MLU observer failed')
                    else:
                        result.update(verify(watch / 'samples.jsonl', evidence / 'task.json', runtime['resource_max_gap_s'], runtime['idle_memory_mib_max']))
            write_json(evidence / 'resources.json', result)
    if not result['resources_clear'] or not result['task_cleanup_verified']:
        raise RuntimeError('MLU resource or process cleanup verification failed')
    return result

def run_campaign(evaluation, args, paths):
    if os.environ.get("PLATFORM") != "cambricon_mlu" or args.backend not in ("lightx2v", "native"):
        raise ValueError("MLU campaign requires its platform and an explicit backend")
    if args.card is None or not 0 <= args.card <= 255 or not 1 <= args.count <= 256:
        raise ValueError("Select one physical card and a chunk size in [1,256]")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_relative_to(Path(paths["output_root"]).resolve()):
        raise ValueError("Campaign output must be inside the mounted output_root")
    runtime = evaluation.read_json(evaluation.HERE / "configs/mlu/runtime.json")
    image = args.image or (paths.get("native_image", runtime["image"]) if args.backend == "native" else runtime["image"])
    image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True).strip()
    metric_image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", paths.get("metric_image", image)], text=True).strip()
    evaluation.verify_source(Path(paths["lightx2v"]), evaluation.read_json(evaluation.HERE / "configs/mlu/source-lock.json")["lightx2v"])
    evaluation.check_inputs(paths)
    indices = ([0] if args.skip_score else smoke_indices(evaluation, args.dataset)) if args.smoke else list(range(len(evaluation.inputs(args.dataset))))
    identity = {
        "model": evaluation.MODEL,
        "backend": args.backend,
        "dataset": args.dataset,
        "physical_card": args.card,
        "smoke": args.smoke,
        "indices": indices,
        "evaluation_sha256": digest(evaluation.__file__),
        "runtime_sha256": digest(evaluation.HERE / "configs/mlu/runtime.json"),
        "model_config_sha256": digest(evaluation.HERE / "configs/mlu/model.json"),
        "suite_sha256": digest(evaluation.HERE / "configs/common/suite.json"),
        "weight_lock_sha256": digest(evaluation.HERE / "configs/common/weight-lock.json"),
        "worker_sha256": digest(Path(__file__).with_name("mlu_worker.py")),
        "controller_sha256": digest(__file__),
        "capture_sha256": digest(Path(__file__).with_name("mlu_capture.py")),
        "policy_sha256": digest(Path(__file__).with_name("gpu_watch.py")),
        "helpers_sha256": digest(Path(__file__).with_name("metax_campaign.py")),
        "native_adapter_sha256": digest(Path(__file__).with_name("mlu_native.py")),
        "native_sources_sha256": digest(Path(__file__).with_name("mlu_sources.py")),
        "source_lock": evaluation.read_json(evaluation.HERE / "configs/mlu/source-lock.json"),
        "image_id": image_id,
        "metric_image_id": metric_image_id,
        "paths": paths,
    }
    campaign_file = root / "campaign.json"
    if campaign_file.exists() and read_json(campaign_file)["identity"] != identity:
        raise ValueError("Resume identity differs; preserve the existing campaign")
    write_json(campaign_file, {"status": "RUNNING", "identity": identity})
    backend = root / args.backend
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
            args.backend,
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
    if args.skip_score:
        report = {"status": "SMOKE_GENERATION_COMPLETE" if args.smoke else "GENERATION_COMPLETE", "coverage": len(indices),
                  "formal_dataset_count": len(evaluation.inputs(args.dataset)), "quality_evaluated": False, "identity": identity}
        write_json(root / "summary.json", report)
        write_json(campaign_file, {"status": report["status"], "identity": identity})
        return
    command = [
        str(Path(evaluation.__file__).resolve()),
        "score-mlu-smoke" if args.smoke else "score",
        "--paths",
        str(args.paths.resolve()),
        "--backend",
        args.backend,
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
        "platform": "cambricon_mlu",
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


def compare(evaluation, args):
    """Report native versus PR measurements using mean latency only."""
    _, native = evaluation.completed_runs(args.native_run, args.dataset, 'native')
    _, adapted = evaluation.completed_runs(args.lightx2v_run, args.dataset, 'lightx2v')
    if {r['gpu'] for r in native.values()} != {r['gpu'] for r in adapted.values()}:
        raise ValueError('Native and PR comparison must use the same physical MLU')
    for root in (args.native_run,args.lightx2v_run):
        for metadata in root.glob('chunks/*/run.json'):
            resource=read_json(metadata.parent/'resources.json')
            if not resource.get('resources_clear') or not resource.get('task_cleanup_verified') or resource.get('returncode')!=0:
                raise ValueError('Comparison requires complete host resource evidence')
    reference=evaluation.latency_summary(native)['mean_s']
    candidate=evaluation.latency_summary(adapted)['mean_s']
    result=dict(status='COMPARISON_COMPLETE',model=evaluation.MODEL,dataset=args.dataset,coverage=len(native),
                native_quality=read_json(args.native_run/'quality.json'),lightx2v_quality=read_json(args.lightx2v_run/'quality.json'),
                native_mean_latency_s=reference,lightx2v_mean_latency_s=candidate,mean_latency_ratio=candidate/reference,
                cross_precision_exact_alignment_claimed=False)
    evaluation.write_json(args.output,result)
    print(json.dumps(result,indent=2),flush=True)

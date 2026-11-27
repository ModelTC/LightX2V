"""Host npu-smi observations in the gpu_watch process ancestry format."""
import json
import re
import subprocess
import time
from pathlib import Path

from gpu_watch import check, lineage


def parse_sample(output, card):
    lines = output.splitlines()
    matches = []
    for index, line in enumerate(lines[:-1]):
        if re.match(r'\|\s*' + str(card) + r'\s+\S+\s*\|\s*OK\s*\|', line):
            row = re.search(r'\|\s*0\s*\|\s*([0-9A-Fa-f:.]+)\s*\|\s*(\d+)\s+(?:\d+\s*/\s*\d+)\s+(\d+)\s*/\s*(\d+)\s*\|', lines[index + 1])
            if not row:
                raise ValueError('Missing NPU chip identity, utilization or HBM memory')
            matches.append(row)
    if len(matches) != 1 or 'Process id' not in output:
        raise ValueError('Incomplete or ambiguous npu-smi card inventory')
    identity = matches[0]
    processes = []
    process_section = output.split('Process id', 1)[1]
    for line in process_section.splitlines():
        row = re.match(r'\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*[^|]+\|\s*(\d+)\s*\|', line)
        if row and int(row[1]) == card:
            if int(row[2]) != 0:
                raise ValueError('Only whole single-chip NPU cards are supported')
            pid = int(row[3])
            ancestors = lineage(pid)
            if not ancestors:
                raise ValueError(f'NPU host PID {pid} is not visible in /proc')
            processes.append(dict(pid=pid, type='C', memory_mib=int(row[4]), ancestors=ancestors))
    idle = re.search(r'No running processes found in NPU\s+' + str(card) + r'\s*\|', process_section)
    if (not processes and not idle) or (processes and idle):
        raise ValueError('Unrecognized or contradictory NPU process inventory')
    return dict(uuid='Ascend-' + identity[1].lower(), physical_card=card, pci_bus_id=identity[1].lower(),
                memory_mib=int(identity[3]), utilization_pct=int(identity[2]), processes=processes)


def sample(card, timeout=3):
    result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=True, timeout=timeout)
    return parse_sample(result.stdout, card)


def verify(samples, task, max_gap, idle_memory):
    from types import SimpleNamespace
    return check(SimpleNamespace(samples=Path(samples), task=Path(task), max_gap=max_gap, idle_memory=idle_memory))


def deployment(card):
    evidence = {}
    for kind in ('info-vnpu', 'device-share', 'vnpu-mode'):
        command = ['npu-smi', 'info', '-t', kind, '-i', str(card), '-c', '0']
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
        evidence[kind] = dict(command=command, stdout=result.stdout)
    if not re.search(r'Total number of vnpu:\s*0\b', evidence['info-vnpu']['stdout']):
        raise ValueError('Virtual NPU partitions are not supported by this capture')
    if not re.search(r'Device-share Status\s*:\s*False', evidence['device-share']['stdout']):
        raise ValueError('NPU device sharing must be disabled')
    return evidence


def main():
    import argparse
    import signal
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pid-file', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=0.2)
    parser.add_argument('--query-timeout', type=float, default=3)
    parser.add_argument('--seconds', type=float, default=3500)
    args = parser.parse_args()
    if not 0 < args.seconds <= 3600 or args.interval <= 0 or args.query_timeout <= 0:
        parser.error('Invalid capture budget')
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    deadline = time.monotonic() + args.seconds
    count = 0
    with (args.output / 'samples.jsonl').open('x') as stream:
        while True:
            point = dict(start=time.monotonic(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
            try:
                device = sample(args.card, args.query_timeout)
                point.update(devices=[device['uuid']], gpus=[device])
                if args.pid_file.exists():
                    point['task_identity'] = lineage(json.loads(args.pid_file.read_text())['pid'])[:1]
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                point['error'] = f'{type(error).__name__}: {error}'
            point['end'] = time.monotonic()
            stream.write(json.dumps(point) + '\n')
            stream.flush()
            temporary = args.output / 'latest.tmp'
            temporary.write_text(json.dumps(point) + '\n')
            temporary.replace(args.output / 'latest.json')
            count += 1
            if stopped or (args.output / 'stop').exists() or time.monotonic() >= deadline:
                break
            time.sleep(args.interval)
    (args.output / 'finished.json').write_text(json.dumps(dict(samples=count)) + '\n')


if __name__ == '__main__':
    main()

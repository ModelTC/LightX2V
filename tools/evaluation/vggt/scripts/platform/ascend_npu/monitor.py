"""Read Ascend npu-smi on the host and emit the evaluator's JSON protocol."""

import argparse
import json
import re
import subprocess


def parse_snapshot(text, card):
    devices = {}
    pending = None
    processes = False
    process_reported = set()
    for line in text.splitlines():
        cells = [part.strip() for part in line.split('|')[1:-1]]
        if 'Process id' in line and 'Process memory' in line:
            processes = True
            continue
        if not cells:
            continue
        if processes:
            empty = re.search(r'No running processes found in NPU (\d+)', line)
            if empty:
                process_reported.add(int(empty[1]))
            location = re.fullmatch(r'(\d+)\s+(\d+)', cells[0])
            if location and len(cells) >= 4 and cells[1].isdigit():
                index = int(location[1])
                process_reported.add(index)
                if index in devices:
                    devices[index]['pids'].append(int(cells[1]))
            continue
        match = re.fullmatch(r'(\d+)\s+(\S+)', cells[0])
        if match and len(cells) >= 3 and cells[1] in ('OK', 'Warning', 'Alarm'):
            pending = int(match[1])
            values = cells[2].split()
            devices[pending] = {'card': pending, 'name': match[2], 'health': cells[1], 'pids': []}
            if len(values) >= 2:
                devices[pending].update(power_w=float(values[0]), temperature_c=float(values[1]))
        elif pending is not None and len(cells) >= 3 and re.fullmatch(r'\d+', cells[0]):
            values = re.fullmatch(r'([\d.]+)\s+(\d+)\s*/\s*(\d+)\s+(\d+)\s*/\s*(\d+)', cells[2])
            if values is None:
                raise ValueError('Unsupported npu-smi usage row: ' + line)
            devices[pending].update(util_percent=float(values[1]), memory_mib=int(values[4]), capacity_mib=int(values[5]), device_id=cells[1], chip=int(cells[0]))
            pending = None
    # Fail closed if the process section or required usage fields disappeared.
    if not processes or card not in process_reported or card not in devices or 'memory_mib' not in devices[card]:
        raise ValueError(f'Incomplete npu-smi snapshot for physical NPU {card}')
    if devices[card]['health'] != 'OK':
        raise ValueError(f'Physical NPU {card} is unhealthy: ' + devices[card]['health'])
    return devices[card]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--card', type=int, required=True)
    args = parser.parse_args()
    result = subprocess.run(['npu-smi', 'info'], check=True, capture_output=True, text=True, timeout=8)
    snapshot = parse_snapshot(result.stdout, args.card)
    snapshot['driver_version'] = next((line.split('Version:', 1)[1].strip(' |') for line in result.stdout.splitlines() if 'Version:' in line), 'unknown')
    print(json.dumps(snapshot, allow_nan=False))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Report actual MetaX card usage; fail closed if the vendor output changes."""
import argparse
import json
import re
import subprocess


def query(*args):
    return subprocess.check_output(['mx-smi', *args], text=True, timeout=8)


def snapshot(card):
    summary = query('-i', str(card))
    line = re.search(r'^\|\s*\d+\s+MetaX\s+[^|]+\|\s*(\d+)\s+[^|]*\|\s*([^|]+)\|\s*(\d+)%[^|]*\|\s*\n\|\s*(\d+)W\s*/\s*\d+W\s*\|\s*(\d+)C\s+[^|]*\|\s*(\d+)\s*/\s*(\d+)\s*MiB', summary, re.M)
    if not line or int(line.group(1)) != card:
        raise RuntimeError('Unable to parse requested MetaX card summary')
    processes = query('--show-all-process', '-i', str(card))
    if 'Process:' not in processes or 'End of Log' not in processes:
        raise RuntimeError('MetaX process query did not return a complete process table')
    section = processes.split('|=================================================================================|', 1)
    if len(section) != 2:
        raise RuntimeError('Unable to locate MetaX process table')
    section = section[1].split('End of Log', 1)[0]
    pids = []
    for item in section.splitlines():
        content = item.strip().strip('|').strip()
        if not content or set(content) <= {'-', '+'} or content == 'no process found':
            continue
        row = re.match(r'^(\d+)\s+(\d+)\s+\S+', content)
        if not row:
            raise RuntimeError('Unrecognized MetaX process row: ' + content)
        if int(row.group(1)) == card:
            pids.append(int(row.group(2)))
    return {'card': card, 'pci_bus_id': line.group(2).strip(), 'util_percent': int(line.group(3)), 'power_w': int(line.group(4)), 'temperature_c': int(line.group(5)), 'memory_mib': int(line.group(6)), 'capacity_mib': int(line.group(7)), 'pids': sorted(set(pids))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--card', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.card), allow_nan=False))


if __name__ == '__main__':
    main()

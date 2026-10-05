#!/usr/bin/env python3
"""Audit embedded-audio Ref2AV cases without modifying inputs or calling APIs.

An 'affected' row has a reference video with an audio stream: it needs mapping
review, NOT necessarily prompt re-enhancement. Both video+standalone-audio and
video-without-standalone-audio cases are counted. Images may occur in either.
Only reference videos are probed, never target videos. Repeated video paths
share one probe; reference occurrences still count separately for numbering.
"""

import argparse
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

DEFAULT_INPUT = "/mnt/lm_data_afs/gushiqiao/datasets/h3_ref2av_ir_simple/ref2av.minimaxi.com.jsonl"
DEFAULT_ROOT = Path("/mnt/lm_data_afs/gushiqiao/datasets")
KINDS = {name: kind for kind in ("image", "video", "audio") for name in (kind, "reference_" + kind, kind + "_url")}
METRICS = ("rows", "invalid_references", "with_video", "with_standalone_audio", "candidates", "affected", "affected_incomplete", "no_embedded_audio", "unknown")
URI = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def source_name(row):
    sample_id = str(row.get("sample_id") or "").lower()
    source = str(row.get("source_name") or "").lower()
    seko = row.get("seko") or {}
    provider = str(seko.get("provider") or "").lower() if isinstance(seko, dict) else ""
    if sample_id.startswith("seko_minimax_") or source == "seko_minimax":
        return "seko_minimax"
    if sample_id.startswith("seko_volc_") or source == "seko_volc":
        return "seko_volc"
    if sample_id.startswith("r2v_") or source in {"omni", "omni_r2v", "omni-r2v", "omini-r2v"}:
        return "omni"
    if source.startswith("seko") or isinstance(seko, dict) and seko:
        if provider in {"minimax", "volc"}:
            return "seko_" + provider
        return "seko"
    return source or "unknown"


def extract_references(row):
    """Select one authoritative list; never merge aliases or include targets."""
    entries = None
    for key in ("references", "actual_ordered_references", "ordered_references"):
        if row.get(key) is not None:
            entries = row[key]
            break
    seko = row.get("seko") or {}
    if entries is None and isinstance(seko, dict):
        entries = seko.get("input_media")
    fields = tuple((kind, "reference_" + suffix, "reference_" + kind + "_url") for kind, suffix in (("image", "images"), ("video", "videos"), ("audio", "audios")))
    if entries is None and any(key in row for _, *keys in fields for key in keys):
        entries = []
        for kind, *keys in fields:
            values = next((row[key] for key in keys if key in row), [])
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                raise ValueError(f"{keys[0]} must be a list or filename")
            entries.extend({"kind": kind, "path": value} for value in values)
    if entries is None:
        entries = []
        groups = row.get("inputs") or []
        if not isinstance(groups, list):
            raise ValueError("inputs must be a list")
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("assets", []), list):
                raise ValueError("inputs must contain asset groups")
            entries.extend(group.get("assets", []))
    if not isinstance(entries, list):
        raise ValueError("references/input_media must be a list")
    result = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("reference entries must be objects")
        kind = entry.get("kind") or entry.get("modality") or entry.get("type") or entry.get("role")
        if kind not in KINDS:
            raise ValueError(f"unknown reference kind: {kind!r}")
        path = next((entry[key] for key in ("path", "local_path", "rel_path", "media_path", "url") if entry.get(key)), None)
        result.append({"kind": KINDS[kind], "path": path})
    return result


def probe_video(path, ffprobe, timeout):
    """Unknown/missing/unreadable media must never be treated as no audio."""
    result = {"path": str(path) if path is not None else None}
    if not isinstance(path, str) or not path or URI.match(path):
        return dict(result, status="unsupported", error="A server-local video filename is required")
    try:
        info = Path(path).stat()
        if not stat.S_ISREG(info.st_mode) or not info.st_size:
            raise ValueError("Video is empty or is not a regular file")
        completed = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "stream=index,codec_type,codec_name,sample_rate,channels", "-of", "json", path],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
        data = json.loads(completed.stdout)
        streams = data.get("streams") if isinstance(data, dict) else None
        if not isinstance(streams, list) or not all(isinstance(item, dict) for item in streams):
            raise ValueError("ffprobe did not return a valid streams list")
        if not any(stream.get("codec_type") == "video" for stream in streams):
            raise ValueError("Reference file has no video stream")
        audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
        return dict(result, status="ok", has_audio=bool(audios), audio_streams=audios)
    except FileNotFoundError as error:
        return dict(result, status="missing", error=str(error))
    except subprocess.CalledProcessError as error:
        return dict(result, status="error", error=(error.stderr or str(error))[:500])
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return dict(result, status="error", error=str(error)[:500])


def classify_case(video_results, explicit_audio_count):
    known = [result for result in video_results if result.get("status") == "ok"]
    embedded = sum(bool(result.get("has_audio")) for result in known)
    complete = bool(video_results) and len(known) == len(video_results)
    return {
        "classification": "affected" if embedded else "no_embedded_audio" if complete else "unknown",
        "mapping_complete": complete,
        "embedded_audio_count_lower_bound": embedded,
        # This is the offset under image -> video -> standalone audio ordering.
        # It is NOT a recommendation to blindly renumber enhanced prompts.
        "audio_index_shift": embedded if complete else None,
        "explicit_audio_count": explicit_audio_count,
        "case_type": "video_with_standalone_audio" if explicit_audio_count else "video_without_standalone_audio",
    }


def resolve_path(value, source, input_path, args):
    if not isinstance(value, str) or not value or URI.match(value):
        return value
    path = Path(value).expanduser()
    if not path.is_absolute():
        root = args.media_root
        if root is None:
            root = args.seko_media_root if source.startswith("seko") else args.omni_media_root if source == "omni" else input_path.parent
        path = root / path
    return os.path.realpath(path)


class Progress:
    def __init__(self, interval):
        self.interval = interval
        self.started = time.monotonic()
        self.state = {}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)

    def emit(self, event):
        message = {**self.state, "elapsed_s": round(time.monotonic() - self.started, 1)}
        print(f"[audio-scan] {event} {json.dumps(message, ensure_ascii=False)}", file=sys.stderr, flush=True)

    def heartbeat(self):
        while not self.stop.wait(self.interval):
            self.emit("heartbeat")

    def __enter__(self):
        self.emit("start")
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


def scan(args):
    input_paths = [path.expanduser().resolve() for path in args.input]
    if len(set(input_paths)) != len(input_paths):
        raise ValueError("Duplicate --input paths would double count rows")
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Input JSONL not found: {path}")
    if shutil.which(args.ffprobe) is None:
        raise FileNotFoundError(f"ffprobe executable not found: {args.ffprobe}")
    output = args.output_dir.expanduser().resolve()
    # Never overwrite another audit or any source file.
    output.mkdir(parents=True, exist_ok=False)

    def counts():
        return dict.fromkeys(METRICS, 0)

    summary = {
        "complete": False,
        "inputs": [],
        "output_dir": str(output),
        "classification_note": "affected means video audio is present and mapping needs review, NOT proven prompt error or automatic retry",
        "offset_contract": "image -> video -> standalone audio; count video-reference occurrences with audio, not unique files or individual tracks",
        "totals": counts(),
        "by_source": {},
        "by_ir_status": {},
        "by_case_type": {},
        "invalid_json_lines": 0,
        "invalid_record_lines": 0,
        "incomplete_tail_lines": 0,
        "unique_video_paths": 0,
        "probe_status_counts": {},
        "examples_of_input_errors": [],
    }
    cache, pending = {}, deque()

    def increment(groups, key):
        for group in groups:
            group[key] += 1

    def input_error(path, number, reason):
        if len(summary["examples_of_input_errors"]) < 20:
            summary["examples_of_input_errors"].append({"input": str(path), "line": number, "error": reason[:500]})

    try:
        with (output / "cases.jsonl").open("x", encoding="utf-8") as report, Progress(args.log_interval) as progress, ThreadPoolExecutor(max_workers=args.workers) as pool:

            def snapshot():
                progress.state = {**summary["totals"], "unique_video_paths": len(cache), "pending_rows": len(pending)}

            def finish_one():
                case, groups, futures = pending.popleft()
                results = [future.result() for future in futures]
                case.update(classify_case(results, case["explicit_audio_count"]), videos=results)
                increment(groups, case["classification"])
                if case["classification"] == "affected" and not case["mapping_complete"]:
                    increment(groups, "affected_incomplete")
                report.write(json.dumps(case, ensure_ascii=False) + "\n")
                report.flush()
                snapshot()

            for path in input_paths:
                with path.open("rb") as stream:
                    end = os.fstat(stream.fileno()).st_size
                    summary["inputs"].append({"path": str(path), "snapshot_bytes": end})
                    number = 0
                    while stream.tell() < end:
                        raw = stream.readline(end - stream.tell())
                        if not raw:
                            raise ValueError(f"Input was truncated during scanning: {path}")
                        number += 1
                        if not raw.strip():
                            continue
                        try:
                            row = json.loads(raw)
                        except (ValueError, UnicodeError) as error:
                            key = "incomplete_tail_lines" if stream.tell() == end and not raw.endswith(b"\n") else "invalid_json_lines"
                            summary[key] += 1
                            input_error(path, number, str(error))
                            continue
                        if not isinstance(row, dict):
                            summary["invalid_record_lines"] += 1
                            input_error(path, number, "JSON record must be an object")
                            continue
                        source = source_name(row)
                        ir_status = str(row.get("ir_status") or "not_recorded")
                        groups = [summary["totals"], summary["by_source"].setdefault(source, counts()), summary["by_ir_status"].setdefault(ir_status, counts())]
                        increment(groups, "rows")
                        try:
                            refs = extract_references(row)
                        except (ValueError, TypeError) as error:
                            increment(groups, "invalid_references")
                            input_error(path, number, str(error))
                            continue
                        videos = [ref for ref in refs if ref["kind"] == "video"]
                        audio_count = sum(ref["kind"] == "audio" for ref in refs)
                        if audio_count:
                            increment(groups, "with_standalone_audio")
                        if videos:
                            increment(groups, "with_video")
                            kind = "video_with_standalone_audio" if audio_count else "video_without_standalone_audio"
                            bucket = summary["by_case_type"].setdefault(kind, counts())
                            bucket["rows"] += 1
                            bucket["with_video"] += 1
                            bucket["with_standalone_audio"] += int(audio_count > 0)
                            groups.append(bucket)
                            increment(groups, "candidates")
                            futures = []
                            for ref in videos:
                                filename = resolve_path(ref["path"], source, path, args)
                                # Invalid path shapes are still represented as unknown probes.
                                key = json.dumps(filename, sort_keys=True, ensure_ascii=False)
                                if key not in cache:
                                    cache[key] = pool.submit(probe_video, filename, args.ffprobe, args.probe_timeout)
                                futures.append(cache[key])
                            case = {
                                "input": str(path),
                                "input_line": number,
                                "sample_id": row.get("sample_id"),
                                "metadata_id": row.get("metadata_id"),
                                "source": source,
                                "ir_status": ir_status,
                                "video_reference_count": len(videos),
                                "explicit_audio_count": audio_count,
                            }
                            pending.append((case, groups, futures))
                            if len(pending) >= args.workers * 4:
                                finish_one()
                        snapshot()
                        if summary["totals"]["rows"] % args.log_every == 0:
                            progress.emit("rows")
            while pending:
                finish_one()
            summary["complete"] = True
            progress.emit("complete")
    finally:
        summary["unique_video_paths"] = len(cache)
        for future in cache.values():
            if future.done() and not future.cancelled() and future.exception() is None:
                status = future.result()["status"]
                summary["probe_status_counts"][status] = summary["probe_status_counts"].get(status, 0) + 1
        with (output / "summary.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, help="JSONL path; repeat for multiple nonoverlapping manifests")
    parser.add_argument("--output-dir", type=Path, default=Path("save_results") / ("audio_case_scan_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--workers", type=int, default=8, help="Concurrent local ffprobe processes, not API calls")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--probe-timeout", type=float, default=15)
    parser.add_argument("--log-every", type=int, default=10000)
    parser.add_argument("--log-interval", type=float, default=30)
    parser.add_argument("--media-root", type=Path, help="Override relative media root for every source")
    parser.add_argument("--seko-media-root", type=Path, default=DEFAULT_ROOT / "seko_data")
    parser.add_argument("--omni-media-root", type=Path, default=DEFAULT_ROOT / "omini-r2v")
    args = parser.parse_args(argv)
    args.input = args.input or [Path(DEFAULT_INPUT)]
    if args.workers < 1 or args.log_every < 1 or any(not math.isfinite(value) or value <= 0 for value in (args.probe_timeout, args.log_interval)):
        parser.error("Workers, log frequency, and timeouts must be positive")
    return args


def main(argv=None):
    try:
        summary = scan(parse_args(argv))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("扫描完成；待核查 ≠ 已确认错误，不会自动重跑。")
    print("来源\t记录数\t含视频\t有视频音轨/待核查\t无视频音轨\t音轨状态未知")
    for source, item in summary["by_source"].items():
        print(f"{source}\t{item['rows']}\t{item['candidates']}\t{item['affected']}\t{item['no_embedded_audio']}\t{item['unknown']}")
    for kind, item in summary["by_case_type"].items():
        label = "视频＋独立音频" if kind == "video_with_standalone_audio" else "有视频、无独立音频（可含图片）"
        print(f"{label}：候选={item['candidates']}，待核查={item['affected']}，无音轨={item['no_embedded_audio']}，未知={item['unknown']}")
    print("成功增强记录中的待核查数：", summary["by_ir_status"].get("succeeded", {}).get("affected", 0))
    print("探测失败等原因导致偏移量尚不完整的待核查数：", summary["totals"]["affected_incomplete"])
    print(f"统计：{summary['output_dir']}/summary.json")
    print(f"逐条明细（所有含参考视频的候选）：{summary['output_dir']}/cases.jsonl")
    if summary["invalid_json_lines"] or summary["invalid_record_lines"] or summary["incomplete_tail_lines"] or summary["totals"]["invalid_references"]:
        print("注意：有损坏/未写完/引用结构异常的输入记录，详见 summary.json。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

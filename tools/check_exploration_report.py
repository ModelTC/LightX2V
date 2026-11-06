"""Offline checks for exploration-report structure and review-entry reachability."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

PENDING = re.compile(r"待审|待(?:质量|画面|媒体)审核|pending review", re.I)
LINK = re.compile(r"\[[^\]\n]*\]\((<[^>\n]+>|[^)\n]+)\)|\[[^\]\n]+\]\[([^\]\n]+)\]")
LIMITS = [
    "Only numbered report sections, table-row dispositions, local links and review-entry reachability are checked.",
    "A reachable review entry does not prove correct media, parent/candidate selection, playback or quality approval.",
    "Coverage completeness, evidence sufficiency, deferral value and overall exploration completion require semantic review.",
    "Unparsed linked payloads are checked for existence only; media/numerical identities remain with their own receipts.",
    "This CLI cannot prevent an agent or external publisher from bypassing it; enforce verification at that consumer if needed.",
]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def visible_markdown_lines(text):
    """Keep source line numbers; ignore fenced examples and HTML comments."""
    text = re.sub(r"<!--[\s\S]*?-->", lambda match: "\n" * match.group().count("\n"), text)
    lines = text.splitlines()
    fence = None
    for index, line in enumerate(lines):
        match = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            lines[index] = ""
        elif fence:
            lines[index] = ""
    return lines


def markdown_lines(path):
    return visible_markdown_lines(path.read_text(encoding="utf-8"))


class AnchorParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.anchors = {}

    def handle_starttag(self, tag, attrs):
        names = {value for key, value in attrs if value and (key == "id" or (key == "name" and tag == "a"))}
        for name in names:
            if name in self.anchors:
                raise ValueError(f"duplicate anchor: {name}")
            self.anchors[name] = self.getpos()[0] - 1


def html_anchors(text):
    parser = AnchorParser()
    parser.feed(text)
    return parser.anchors


def mask_inline_code(line):
    """Ignore same-line code spans during link/HTML-anchor discovery; retain columns."""
    runs = list(re.finditer(r"`+", line))
    following = {}
    closing = {}
    for index in range(len(runs) - 1, -1, -1):
        width = len(runs[index].group())
        closing[index] = following.get(width)
        following[width] = index
    result = list(line)
    consumed = 0
    for index, run in enumerate(runs):
        if run.start() < consumed:
            continue
        before = run.start()
        while before > 0 and line[before - 1] == "\\":
            before -= 1
        end = closing[index]
        if (run.start() - before) % 2 or end is None:
            continue
        consumed = runs[end].end()
        result[run.start() : consumed] = [" "] * (consumed - run.start())
    return "".join(result)


def anchors(lines):
    result = html_anchors("\n".join(mask_inline_code(line) for line in lines))
    occurrences = {}
    for index, line in enumerate(lines):
        heading = re.match(r"^#{1,6}\s+(.+?)\s*#*$", line)
        if heading:
            title = re.sub(r"<[^>]+>", "", heading.group(1)).lower()
            slug = re.sub(r"[^\w\- ]", "", title).replace(" ", "-")
            number = occurrences.get(slug, 0)
            occurrences[slug] = number + 1
            name = slug + (f"-{number}" if number else "")
            if name in result and result[name] != index:
                raise ValueError(f"ambiguous Markdown anchor: {name}")
            result[name] = index
    return result


def reference_label(value):
    return re.sub(r"[ \t\r\n]+", " ", value.strip(" \t\r\n")).casefold()


def references(lines):
    definitions = {}
    for line in lines:
        match = re.match(r"^ {0,3}\[([^]]+)\]:\s*(<[^>]+>|\S+)\s*$", line)
        if match:
            # Markdown resolves repeated reference labels using the first definition.
            definitions.setdefault(reference_label(match.group(1)), match.group(2).strip("<>"))
    return definitions


def links(line, definitions):
    return [m.group(1).strip("<>") if m.group(1) is not None else definitions.get(reference_label(m.group(2))) for m in LINK.finditer(mask_inline_code(line))]


def check(report):
    report = report.resolve()
    errors = []
    inputs = {}
    documents = {}
    local_targets = {report}

    def remember(path):
        if path not in inputs:
            inputs[path] = digest(path)

    def document(path):
        if path not in documents:
            remember(path)
            lines = markdown_lines(path)
            documents[path] = (lines, anchors(lines), references(lines))
        return documents[path]

    def resolve(source, raw):
        if raw is None:
            raise ValueError("undefined reference link")
        parsed = urlsplit(raw)
        if parsed.scheme or parsed.netloc:
            return None
        target = (source.parent / unquote(parsed.path)).resolve() if parsed.path else source
        local_targets.add(target)
        if not target.exists():
            raise ValueError(f"missing target: {raw}")
        fragment = unquote(parsed.fragment)
        if target.is_file():
            if target.suffix.lower() in (".md", ".html", ".htm"):
                remember(target)
            if fragment and target.suffix.lower() == ".md" and fragment not in document(target)[1]:
                raise ValueError(f"missing Markdown anchor: {raw}")
            if fragment and target.suffix.lower() in (".html", ".htm"):
                ids = html_anchors(target.read_text(encoding="utf-8"))
                if fragment not in ids:
                    raise ValueError(f"missing HTML anchor: {raw}")
        return target, fragment

    def review_entry(source, raw, seen):
        resolved = resolve(source, raw)
        if resolved is None:
            return False  # Offline verification cannot certify an external page.
        target, fragment = resolved
        key = (target, fragment)
        if key in seen or not target.is_file():
            return False
        if target.suffix.lower() in (".html", ".htm"):
            return target.stat().st_size > 0
        if target.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm"):
            return target.stat().st_size > 0
        if target.suffix.lower() != ".md":
            return False
        lines, ids, definitions = document(target)
        start = ids[fragment] if fragment else 0
        # An anchored table row is one candidate; do not borrow another row's link.
        end = start + 1
        if not fragment:
            end = len(lines)
        elif not lines[start].lstrip().startswith("|"):
            while end < len(lines) and end not in ids.values() and not re.match(r"^#{1,6}\s", lines[end]):
                end += 1
        return any(review_entry(target, link, seen | {key}) for line in lines[start:end] for link in links(line, definitions))

    lines, _, definitions = document(report)
    sections = [int(m.group(1)) for line in lines if (m := re.match(r"^#{1,6}\s+([1-7])\.\s", line))]
    if sections != list(range(1, 8)):
        errors.append({"line": 1, "code": "report_sections", "message": "expected seven numbered sections in order"})
    pending = []
    checked_links = 0
    for index, line in enumerate(lines):
        for raw in links(line, definitions):
            try:
                resolve(report, raw)
                checked_links += 1
            except ValueError as error:
                errors.append({"line": index + 1, "code": "local_link", "message": str(error)})
        if not line.lstrip().startswith("|"):
            continue
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if re.match(r"^\s*\|[\s:|\-]+\|\s*$", next_line) or re.match(r"^\s*\|[\s:|\-]+\|\s*$", line):
            continue  # Table header or separator, not a candidate.
        if "关联待核" in line:
            errors.append({"line": index + 1, "code": "unresolved_relation", "message": "unresolved applicability requires investigation or an evidenced disposition"})
        if "未接线" in line and not any(word in line for word in ("暂缓", "不适用", "范围外", "不选", "外部阻塞")):
            errors.append({"line": index + 1, "code": "unwired_disposition", "message": "unwired describes implementation state; give an actual disposition"})
        quality_line = re.sub(r"(?:源码|代码|实现)待审", "", line)
        quality_line = re.sub(r"(?:无|没有|无需|不再)(?:待审|待(?:质量|画面|媒体)审核)|no pending review", "", quality_line, flags=re.I)
        if not PENDING.search(quality_line):
            continue
        reachable = False
        for raw in links(line, definitions):
            try:
                if review_entry(report, raw, set()):
                    reachable = True
                    break
            except ValueError:
                pass  # The broken link is already reported above or is not a review entry.
        pending.append({"line": index + 1, "reachable_entry": reachable})
        if not reachable:
            errors.append({"line": index + 1, "code": "missing_review_entry", "message": "pending quality row needs local review media, an HTML page or a Markdown entry reaching it"})
    # Bind the checker too: a changed checker requires a new receipt.
    remember(Path(__file__).resolve())
    return {
        "kind": "exploration_report_check_v1",
        "passed": not errors,
        "report": report,
        "inputs": inputs,
        "local_targets": local_targets,
        "local_links_checked": checked_links,
        "pending_rows": pending,
        "errors": errors,
        "limitations": LIMITS,
    }


def write_receipt(result, output):
    output = output.resolve()
    originals = set(result["inputs"]) | result["local_targets"]
    if output in originals or (output.exists() and any(path.exists() and output.samefile(path) for path in originals)):
        raise ValueError("receipt must not overwrite or be linked as a checked input; link its location in the task work package")
    serializable = {
        **{key: value for key, value in result.items() if key != "local_targets"},
        "report": os.path.relpath(result["report"], output.parent),
        "inputs": {os.path.relpath(path, output.parent): value for path, value in result["inputs"].items()},
    }
    output.write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify(receipt, expected_report=None):
    receipt = receipt.resolve()
    saved = json.loads(receipt.read_text(encoding="utf-8"))
    if saved.get("kind") != "exploration_report_check_v1" or saved.get("passed") is not True:
        raise ValueError("no passing report-check receipt")
    report = (receipt.parent / saved["report"]).resolve()
    if expected_report is not None and report != expected_report.resolve():
        raise ValueError("receipt belongs to another report")
    current = check(report)
    expected = {(receipt.parent / path).resolve(): value for path, value in saved["inputs"].items()}
    if not current["passed"] or current["inputs"] != expected:
        raise ValueError("report, linked input or checker changed; run a fresh check")
    return current


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--report", type=Path)
    source.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path, help="write a passing or failing receipt; only with --report")
    parser.add_argument("--expected-report", type=Path, help="bind verification to the caller task report; only with --verify")
    args = parser.parse_args(argv)
    if args.expected_report and not args.verify:
        parser.error("--expected-report is only valid with --verify")
    if args.verify and args.output:
        parser.error("--output is only valid with --report")
    try:
        result = verify(args.verify, args.expected_report) if args.verify else check(args.report)
        if args.output:
            write_receipt(result, args.output)
        print(json.dumps({key: result[key] for key in ("passed", "local_links_checked", "pending_rows", "errors", "limitations")}, ensure_ascii=False))
        return 0 if result["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"report check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

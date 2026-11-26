#!/usr/bin/env python3
"""Collect existing H3 T2AV/I2AV assets into one absolute-path JSON manifest.

No inference, resizing, prompt rewriting, or tensor re-encoding is performed.
Source files are retained. Association uses condition identity or exact input
signatures; missing and ambiguous associations are reported, never guessed.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import zipfile


CACHE_NAME = 'minimax_h3_self_generated_latents_124_bf16'
FIELDS = {
    'condition': 'condition_path', 'image': 'first_frame',
    'teacher_video': 'video_path', 'teacher_video_latent': 'video_latent_path',
    'teacher_audio_latent': 'audio_latent_path',
}
TEACHER_ASSETS = ('teacher_video', 'teacher_video_latent', 'teacher_audio_latent')


def text(value):
    return value.strip() if isinstance(value, str) and value.strip() else None


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def read_rows(path):
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                yield number, json.loads(line)


class Paths:
    def __init__(self, root, mappings):
        self.root = root
        self.mappings = sorted(mappings, key=lambda pair: len(str(pair[0])), reverse=True)
        self.hashes = {}

    def mapped(self, path):
        for old, new in self.mappings:
            if path == old or old in path.parents:
                return new / path.relative_to(old)
        return path

    def resolve(self, value, manifest):
        if not value:
            return None
        path = Path(value).expanduser()
        original = Path(os.path.abspath(path if path.is_absolute() else manifest.parent / path))
        mapped = self.mapped(original)
        return mapped if mapped.is_file() or not original.is_file() else original

    def identity(self, path):
        return str(self.mapped(path)) if path else None

    def image_identity(self, path, declared=None):
        if path and path.is_file():
            if path not in self.hashes:
                self.hashes[path] = file_sha(path)
            if declared and declared != self.hashes[path]:
                raise ValueError(f'Image SHA256 differs from metadata: {path}')
            return 'sha256:' + self.hashes[path]
        return 'sha256:' + declared if declared else self.identity(path)


def prompt_info(row, teacher=False):
    original = text(row.get('prompt')) or text(row.get('original_prompt'))
    enhanced = text(row.get('enhanced_prompt'))
    actual = (text(row.get('generation_prompt')) if teacher else None) or text(row.get('caption'))
    declared = row.get('prompt_variant')
    if actual is None and declared in ('original', 'enhanced'):
        actual = {'original': original, 'enhanced': enhanced}[declared]
    if actual is None and original and not enhanced:
        actual = original
    if actual is None and enhanced and not original:
        actual = enhanced
    matches = [name for name, value in [('original', original), ('enhanced', enhanced)] if value and value == actual]
    if len(matches) == 1:
        variant = matches[0]
    elif declared in matches:
        variant = declared
    elif not matches and actual and declared in ('original', 'enhanced'):
        existing = original if declared == 'original' else enhanced
        if existing and existing != actual:
            raise ValueError('Declared prompt variant disagrees with actual caption/generation_prompt')
        variant = declared
        if declared == 'original':
            original = actual
        else:
            enhanced = actual
    else:
        variant = 'unknown'
    return {'original': original, 'enhanced': enhanced}, {'variant': variant, 'text': actual, 'sha256': sha(actual) if actual else None}


def unpack_teacher_zip(root):
    path = root / 'h3_teacher_t2av_i2av_18882.zip'
    if not path.is_file():
        return
    prefixes = ('minimax_h3_teacher_api_5s/', 'minimax_h3_teacher_api_5s_768p/i2av/')
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            relative = Path(member.filename)
            if not member.filename.startswith(prefixes) or member.is_dir():
                continue
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'Unsafe ZIP member: {member.filename}')
            destination = root / relative
            if destination.is_file():
                if destination.stat().st_size != member.file_size:
                    raise ValueError(f'Existing extracted file differs in size: {destination}')
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + '.partial')
            with archive.open(member) as source, temporary.open('wb') as target:
                shutil.copyfileobj(source, target, 8 << 20)
            os.replace(temporary, destination)
    print('Teacher ZIP extracted; source metadata preserved.', flush=True)


def discover(root, teacher_latents):
    sources = []
    legacy = root / 'minimax_h3_dmd_cache_5s_768p'
    for task in ('t2av', 'i2av'):
        for directory in (legacy / task, legacy / 'task_aware_view_3task_portrait' / task):
            if directory.is_dir():
                sources.extend(('condition', p, task) for p in sorted(directory.rglob('metadata.jsonl')))
    own = root / CACHE_NAME / 't2av/metadata.jsonl'
    if own.is_file():
        sources.append(('selfgen', own, 't2av'))
    for directory, task in [('minimax_h3_teacher_api_5s', 't2av'), ('minimax_h3_teacher_api_5s_768p/i2av', 'i2av')]:
        sources.extend(('teacher', p, task) for p in sorted((root / directory).rglob('metadata.jsonl')))
    if teacher_latents:
        path = teacher_latents / 'metadata.jsonl' if teacher_latents.is_dir() else teacher_latents
        if not path.is_file():
            raise FileNotFoundError(path)
        sources.append(('latent', path, 't2av'))
    return sources


def normalize(row, kind, manifest, number, default_task, paths):
    task = row.get('task', default_task).lower()
    if task not in ('t2av', 'i2av'):
        return None
    dimensions = tuple(int(row.get(a, row.get(b, 0))) for a, b in (
        ('target_width', 'width'), ('target_height', 'height'), ('target_num_frames', 'num_frames')))
    if min(dimensions) <= 0:
        raise ValueError('Missing or invalid width/height/num_frames')
    width, height, frames = dimensions
    if width % 32 or height % 32 or frames % 17 != 5:
        raise ValueError('H3 dimensions must be 32-aligned and frame count must be 17*n+5')
    prompt, condition_used = prompt_info(row)
    teacher_prompt, teacher_used = prompt_info(row, teacher=True)
    if kind in ('teacher', 'latent', 'selfgen'):
        for key, value in teacher_prompt.items():
            if value and prompt[key] is None:
                prompt[key] = value
    used = teacher_used if kind == 'teacher' else condition_used
    if not used['text']:
        raise ValueError('Actual encoded/generated prompt cannot be identified')
    assets = {field: paths.resolve(row.get(key), manifest) for field, key in FIELDS.items()}
    if kind == 'condition':
        for field in ('teacher_video', 'teacher_video_latent', 'teacher_audio_latent'):
            assets[field] = None
    if assets['teacher_video'] is None and kind in ('teacher', 'latent'):
        assets['teacher_video'] = paths.resolve(row.get('teacher_video_path'), manifest)
    if task == 't2av':
        assets['image'] = None
    image_hash = row.get('output_first_frame_sha256') or row.get('first_frame_sha256')
    image_key = paths.image_identity(assets['image'], image_hash) if task == 'i2av' else None
    fps = float(row.get('fps', 24))
    expected_video = [24, (frames - 5) // 17 * 5 + 2, height // 16, width // 16]
    expected_audio = [2, 32, round(frames / fps * 40)]
    for key, expected in [('video_latent_shape', expected_video), ('audio_latent_shape', expected_audio)]:
        if row.get(key) is not None and row[key] != expected:
            raise ValueError(f'{key} disagrees with sample dimensions: {row[key]} != {expected}')
    signature = (task, *dimensions, fps, used['sha256'], image_key)
    return {
        'signature': signature, 'assets': assets, 'prompt': prompt, 'used': used,
        'condition_used': condition_used, 'teacher_used': teacher_used,
        'kind': kind, 'width': dimensions[0], 'height': dimensions[1], 'num_frames': dimensions[2],
        'fps': fps, 'task': task, 'source': {'manifest': str(manifest), 'line': number, 'source_id': row.get('source_id'), 'id': row.get('id')},
        'cache_fingerprint': row.get('cache_fingerprint'), 'latent_fingerprint': row.get('latent_fingerprint'),
        'video_latent_shape': row.get('video_latent_shape'), 'audio_latent_shape': row.get('audio_latent_shape'),
        'cache_dtype': row.get('cache_storage_dtype'), 'normalized': row.get('vae_normalized'),
        'teacher_model': row.get('generation_model'), 'teacher_task_id': row.get('generation_task_id'),
        'teacher_origin': 'api' if kind in ('teacher', 'latent') else ('selfgen' if kind == 'selfgen' else None),
    }


class Catalog:
    def __init__(self, paths):
        self.paths = paths
        self.records = []
        self.inputs = defaultdict(set)
        self.conditions = defaultdict(set)
        self.videos = defaultdict(set)
        self.issues = []

    def add(self, item):
        assets, signature = item['assets'], item['signature']
        condition_key = self.paths.identity(assets['condition'])
        video_key = self.paths.identity(assets['teacher_video'])
        candidates = set()
        if video_key:
            candidates = self.videos.get(video_key, set()).copy()
        if not candidates and condition_key:
            candidates = self.conditions.get(condition_key, set()).copy()
        candidates = {i for i in candidates if self.records[i]['signature'] == signature}
        if not candidates:
            candidates = self.inputs.get(signature, set()).copy()
        # Never attach a second video or a conflicting cache to a record by row order.
        compatible = []
        for i in candidates:
            previous = self.records[i]['assets']
            if all(not assets[k] or not previous[k] or self.paths.identity(assets[k]) == self.paths.identity(previous[k]) for k in FIELDS if k != 'image'):
                compatible.append(i)
        if len(compatible) == 1:
            index = compatible[0]
            record = self.records[index]
            for key, value in assets.items():
                if value and (not record['assets'][key] or (not record['assets'][key].is_file() and value.is_file())):
                    record['assets'][key] = value
            for key, value in item['prompt'].items():
                old = record['prompt'][key]
                if old and value and old != value:
                    record['issues'].append('conflicting_' + key + '_prompt')
                elif value:
                    record['prompt'][key] = value
            for key in ('cache_fingerprint', 'latent_fingerprint', 'video_latent_shape', 'audio_latent_shape', 'cache_dtype', 'normalized', 'teacher_model', 'teacher_task_id', 'teacher_origin'):
                if item[key] is not None:
                    record[key] = item[key]
        else:
            index = len(self.records)
            record = {**item, 'assets': dict(assets), 'prompt': dict(item['prompt']), 'sources': [], 'issues': []}
            record['sample_id'] = 'h3_' + sha(json.dumps([signature, condition_key, video_key], ensure_ascii=False))[:24]
            record['condition_prompt'] = None
            record['teacher_prompt'] = None
            self.records.append(record)
            if len(compatible) > 1:
                record['issues'].append('ambiguous_association')
        record['sources'].append(item['source'])
        if item['kind'] in ('condition', 'selfgen', 'latent') and assets['condition']:
            record['condition_prompt'] = item['condition_used']
        if item['kind'] in ('selfgen', 'teacher', 'latent'):
            record['teacher_prompt'] = item['teacher_used']
        if condition_key:
            self.conditions[condition_key].add(index)
        if video_key:
            self.videos[video_key].add(index)
        self.inputs[signature].add(index)


def select_training_assets(record):
    assets = dict(record['assets'])
    prompt = record['teacher_prompt']
    reason = None
    if record['teacher_origin'] == 'api' and prompt and prompt['variant'] == 'original':
        reason = 'api_teacher_original_prompt'
        for field in TEACHER_ASSETS:
            assets[field] = None
    return assets, reason


class Assets:
    """Copy each asset once; persisted hashes allow interrupted runs to resume."""
    def __init__(self, output, dry_run):
        self.output, self.dry_run = output, dry_run
        self.db = sqlite3.connect(output / 'asset_index.sqlite')
        self.db.execute('CREATE TABLE IF NOT EXISTS assets (source TEXT PRIMARY KEY, size INTEGER, mtime INTEGER, destination TEXT, sha256 TEXT)')

    def put(self, source, category):
        if source is None or not source.is_file():
            return None
        if self.dry_run:
            return str(source)
        stat = source.stat()
        cached = self.db.execute('SELECT size,mtime,destination FROM assets WHERE source=?', (str(source),)).fetchone()
        if cached and cached[:2] == (stat.st_size, stat.st_mtime_ns) and Path(cached[2]).is_file() and Path(cached[2]).stat().st_size == stat.st_size:
            return cached[2]
        directory = self.output / 'data' / category
        directory.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        fd, temporary = tempfile.mkstemp(prefix='.copy-', dir=directory)
        try:
            with os.fdopen(fd, 'wb') as target, source.open('rb') as handle:
                for block in iter(lambda: handle.read(8 << 20), b''):
                    target.write(block)
                    digest.update(block)
            after = source.stat()
            if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise RuntimeError(f'Source changed during copy: {source}')
            destination = directory / (digest.hexdigest() + source.suffix)
            os.replace(temporary, destination)
            self.db.execute('INSERT OR REPLACE INTO assets VALUES (?,?,?,?,?)', (str(source), stat.st_size, stat.st_mtime_ns, str(destination), digest.hexdigest()))
            self.db.commit()
            return str(destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def probe_video(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,width,height,nb_frames,avg_frame_rate,duration,sample_rate,channels', '-of', 'json', str(path)], capture_output=True, text=True, timeout=90)
    if result.returncode:
        return {'error': result.stderr.strip()}
    streams = json.loads(result.stdout)['streams']
    video = next((s for s in streams if s['codec_type'] == 'video'), None)
    if video is None:
        return {'error': 'no_video_stream'}
    return {'width': video['width'], 'height': video['height'], 'num_frames': int(video['nb_frames']) if video.get('nb_frames', '').isdigit() else None, 'fps': float(Fraction(video['avg_frame_rate'])), 'audio': next((s for s in streams if s['codec_type'] == 'audio'), None)}


def check_latents(record):
    import torch
    expected = {
        'teacher_video_latent': [24, (record['num_frames'] - 5) // 17 * 5 + 2, record['height'] // 16, record['width'] // 16],
        'teacher_audio_latent': [2, 32, round(record['num_frames'] / record['fps'] * 40)],
    }
    for field, shape in expected.items():
        path = record['assets'][field]
        if path is None or not path.is_file():
            continue
        payload = torch.load(path, map_location='cpu', weights_only=True)
        tensor = payload.get('latents') if isinstance(payload, dict) else payload
        if not torch.is_tensor(tensor) or not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f'Invalid/nonfinite latent: {path}')
        if list(tensor.shape) != shape:
            raise ValueError(f'Latent shape disagrees with declared shape: {path}')
        if isinstance(payload, dict) and payload.get('normalized') is False:
            raise ValueError(f'Latent is explicitly marked unnormalized: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Destination h3_self_gen directory')
    parser.add_argument('--teacher-latents', type=Path, help='Optional additional teacher-latent cache; copied directly into data/')
    parser.add_argument('--path-map', action='append', default=[], metavar='OLD=NEW')
    parser.add_argument('--extract-teacher-zip', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Association report only, no asset copies and no training JSON')
    parser.add_argument('--check-latents', action='store_true', help='Check actual tensors on CPU; requires torch')
    parser.add_argument('--workers', type=int, default=8, help='CPU ffprobe concurrency')
    parser.add_argument('--skip-video-probe', action='store_true', help='Explicitly leave actual media dimensions unverified')
    args = parser.parse_args()
    root, output = args.datasets_root.absolute(), args.output.absolute()
    output.mkdir(parents=True, exist_ok=True)
    if args.extract_teacher_zip:
        unpack_teacher_zip(root)
    mappings = [(Path('/data/nvme6/gushiqiao/datasets'), root), (Path('/mnt/lm_data_afs/gushiqiao/datasets'), root)]
    mappings = [(Path(a), Path(b)) for a, b in (s.split('=', 1) for s in args.path_map)] + mappings
    paths = Paths(root, mappings)
    sources = discover(root, args.teacher_latents.absolute() if args.teacher_latents else None)
    if not sources:
        raise ValueError('No supported manifests found')
    catalog = Catalog(paths)
    for kind, manifest, task in sources:
        count = 0
        for number, row in read_rows(manifest):
            try:
                item = normalize(row, kind, manifest, number, task, paths)
                if item:
                    catalog.add(item)
                    count += 1
            except (ValueError, KeyError, TypeError) as error:
                catalog.issues.append({'manifest': str(manifest), 'line': number, 'error': str(error)})
        print(f'{kind}: {count} rows from {manifest}', flush=True)
    for item in catalog.records:
        item['training_assets'], item['teacher_exclusion_reason'] = select_training_assets(item)
    videos = sorted({r['training_assets']['teacher_video'] for r in catalog.records if r['training_assets']['teacher_video'] and r['training_assets']['teacher_video'].is_file()})
    media = {}
    if not args.skip_video_probe:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for index, (path, info) in enumerate(zip(videos, pool.map(probe_video, videos)), 1):
                media[path] = info
                if index % 1000 == 0:
                    print(f'Video headers checked: {index}/{len(videos)}', flush=True)
    assets = Assets(output, args.dry_run)
    records = []
    for index, item in enumerate(catalog.records, 1):
        issues = list(item['issues'])
        if args.check_latents:
            check_latents({**item, 'assets': item['training_assets']})
        record = {key: item[key] for key in ('sample_id', 'task', 'width', 'height', 'num_frames', 'fps', 'prompt', 'condition_prompt', 'teacher_prompt')}
        record['prompt_variant'] = (item['condition_prompt'] or item['teacher_prompt'] or item['used'])['variant']
        record['training_prompt'] = (item['condition_prompt'] or item['teacher_prompt'] or item['used'])['text']
        identities = []
        for field in ('condition', 'image', 'teacher_video'):
            path = item['assets'][field]
            portable_path = paths.mapped(path) if path else None
            if portable_path and root in portable_path.parents:
                portable_path = portable_path.relative_to(root)
            identities.append(str(portable_path) if portable_path else None)
        record['sample_id'] = 'h3_' + sha(json.dumps([item['signature'][:6], item['used']['sha256'], identities], ensure_ascii=False))[:24]
        for field, path in item['training_assets'].items():
            record[field] = assets.put(path, field)
            if path and not record[field]:
                issues.append('missing_' + field)
        for field in ('condition', 'teacher_video', 'teacher_video_latent', 'teacher_audio_latent'):
            if record[field] is None:
                issues.append('missing_' + field)
        if item['task'] == 'i2av' and record['image'] is None:
            issues.append('missing_image')
        if record['condition'] and not record['condition_prompt']:
            issues.append('condition_prompt_unverified')
        if record['condition_prompt'] and record['teacher_prompt'] and record['condition_prompt']['sha256'] != record['teacher_prompt']['sha256']:
            issues.append('condition_teacher_prompt_mismatch')
        info = media.get(item['training_assets']['teacher_video'])
        if info:
            if 'error' in info:
                issues.append('unreadable_teacher_video')
            elif any(info[k] is not None and info[k] != item[k] for k in ('width', 'height', 'num_frames', 'fps')):
                issues.append('teacher_video_dimensions_differ_from_cache_metadata')
        record.update(
            teacher_video_info=info, target=None, quality=None,
            teacher_model=item['teacher_model'], teacher_task_id=item['teacher_task_id'],
            teacher_origin=item['teacher_origin'], teacher_exclusion_reason=item['teacher_exclusion_reason'],
            teacher_latent_prompt=item['teacher_prompt'] if record['teacher_video_latent'] or record['teacher_audio_latent'] else None,
            latent_metadata={k: item[k] for k in ('video_latent_shape', 'audio_latent_shape', 'cache_dtype', 'normalized', 'latent_fingerprint')},
            cache_fingerprint=item['cache_fingerprint'],
            validation={'latent_tensors_checked': args.check_latents, 'issues': sorted(set(issues))},
            provenance=item['sources'],
        )
        records.append(record)
        if index % 100 == 0:
            print(f'Records {index}/{len(catalog.records)}', flush=True)
    assets.db.close()
    summary = {
        'schema_version': 1, 'dry_run': args.dry_run, 'records': len(records),
        'tasks': dict(Counter(r['task'] for r in records)),
        'resolutions': dict(Counter(f"{r['width']}x{r['height']}" for r in records)),
        'prompt_variants': dict(Counter(r['prompt_variant'] for r in records)),
        'available': {k: sum(r[k] is not None for r in records) for k in FIELDS},
        'excluded_teachers': dict(Counter(r['teacher_exclusion_reason'] for r in records if r['teacher_exclusion_reason'])),
        'issues': dict(Counter(issue for r in records for issue in r['validation']['issues'])),
        'rejected_rows': catalog.issues,
    }
    if not args.dry_run:
        assets = Assets(output, False)
        for _, source, _ in sources:
            assets.put(source, 'source_metadata')
        assets.db.close()
    save_json(output / ('preview.json' if args.dry_run else 'dataset.json'), records)
    save_json(output / ('preview_report.json' if args.dry_run else 'report.json'), summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'rejected_rows'}, ensure_ascii=False, indent=2), flush=True)
    print(f'Rejected rows: {len(catalog.issues)}; inspect report before training.', flush=True)


if __name__ == '__main__':
    main()

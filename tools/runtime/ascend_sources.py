"""Reconstruct one pinned main checkout with the three inference PR overlays."""
import hashlib
import os
import subprocess
from pathlib import Path


def git(root, *arguments):
    return subprocess.check_output(['git', '-C', str(root), *arguments], env=dict(os.environ, GIT_LFS_SKIP_SMUDGE='1'))


def verify(root, specification):
    root = Path(root)
    if git(root, 'rev-parse', 'HEAD').decode().strip() != specification['base_revision']:
        raise ValueError('Ascend runtime base revision differs')
    expected = {item['path']: item['sha256'] for item in specification['runtime_files']}
    for name, checksum in expected.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f'Ascend source file differs: {name}')
    changed = set(git(root, 'diff', '--name-only', 'HEAD', '--', '*.py', '*.sh', '*.json', '*.yaml', '*.yml', '*.toml').decode().splitlines())
    changed.update(git(root, 'ls-files', '--others', '--exclude-standard', '--', '*.py', '*.sh', '*.json', '*.yaml', '*.yml', '*.toml').decode().splitlines())
    if changed - expected.keys():
        raise ValueError(f'Unexpected runtime source changes: {sorted(changed - expected.keys())}')


def prepare(root, specification):
    root = Path(root)
    if not root.exists():
        root.mkdir(parents=True)
        subprocess.run(['git', 'init', '--quiet', str(root)], check=True)
        git(root, 'fetch', '--depth=1', specification['base_repository'], specification['base_revision'])
        git(root, 'checkout', '--detach', 'FETCH_HEAD')
    if git(root, 'rev-parse', 'HEAD').decode().strip() != specification['base_revision']:
        raise ValueError('Refusing to overwrite an unrelated checkout')
    registry = 'lightx2v/models/runners/registry.py'
    for overlay in specification['overlays']:
        try:
            git(root, 'cat-file', '-e', overlay['revision'] + '^{commit}')
        except subprocess.CalledProcessError:
            git(root, 'fetch', '--depth=1', overlay['repository'], overlay['revision'])
        for name in overlay['files']:
            if name == registry:
                continue
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(git(root, 'show', overlay['revision'] + ':' + name))
    # Preserve all upstream runner registrations while adding each native runner.
    text = git(root, 'show', specification['base_revision'] + ':' + registry).decode()
    for line in specification['registry_additions']:
        if line.strip() not in text:
            position = text.index('{', text.index('RUNNER')) + 1
            text = text[:position] + '\n' + line + text[position:]
    (root / registry).write_text(text)
    verify(root, specification)

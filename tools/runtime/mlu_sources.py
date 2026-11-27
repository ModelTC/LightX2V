"""Reconstruct and verify explicit device-only overlays on pinned native source."""
import hashlib
import os
from pathlib import Path
import subprocess


def git(root, *args):
    return subprocess.check_output(['git','-C',str(root),*args], env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1'))


def verify(root, specification):
    root=Path(root)
    if git(root,'rev-parse','HEAD').decode().strip()!=specification['revision']:
        raise ValueError('Native base revision differs')
    expected={entry['path']:entry['runtime_sha256'] for entry in specification['mlu_patches']}
    for name,checksum in expected.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=checksum:
            raise ValueError(f'Native MLU overlay differs: {name}')
    changed=set(git(root,'diff','--name-only','HEAD','--','*.py','*.sh','*.json','*.yaml','*.yml','*.toml').decode().splitlines())
    changed.update(git(root,'ls-files','--others','--exclude-standard','--','*.py','*.sh','*.json','*.yaml','*.yml','*.toml').decode().splitlines())
    if changed-expected.keys(): raise ValueError(f'Unexpected native source changes: {sorted(changed-expected.keys())}')


def prepare(root, specification):
    root=Path(root)
    if not root.exists():
        root.mkdir(parents=True)
        subprocess.run(['git','init','--quiet',str(root)],check=True)
        git(root,'fetch','--depth=1',specification['repository'],specification['revision'])
        git(root,'checkout','--detach','FETCH_HEAD')
    if git(root,'rev-parse','HEAD').decode().strip()!=specification['revision']:
        raise ValueError('Refusing to overwrite unrelated native checkout')
    for entry in specification['mlu_patches']:
        content=git(root,'show','HEAD:'+entry['path'])
        if hashlib.sha256(content).hexdigest()!=entry['sha256']: raise ValueError('Original native source hash differs')
        source=content.decode()
        for edit in entry['replacements']:
            if source.count(edit['before'])!=1: raise ValueError('Native patch anchor is not unique')
            source=source.replace(edit['before'],edit['after'],1)
        if hashlib.sha256(source.encode()).hexdigest()!=entry['runtime_sha256']: raise ValueError('Native patch reconstruction differs')
        target=root/entry['path']
        if target.read_bytes() not in (content,source.encode()): raise ValueError('Refusing to overwrite unrelated native edits')
        target.write_text(source)
    verify(root,specification)

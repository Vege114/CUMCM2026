"""Hash-locked historical sources that also work in a single-branch clone."""
from __future__ import annotations
import hashlib
from pathlib import Path
import subprocess
import warnings

ROOT=Path(__file__).resolve().parents[2]
EXP005_COMMIT='8108d0475ae4f2ea25565a806dfec5e836f62c0a'
EXP005_REGISTRY_SPEC=EXP005_COMMIT+':reports/registry/exp005.json'
EXP005_REGISTRY_SHA='5af3de5bdf08706696eea4ceb4a57d072b9a957bad61eef16d46bb0b6b52c43f'
EXP005_REGISTRY_SNAPSHOT='reports/experiments/exp008/evidence/forecast/source_registry_exp005.json'
EXP005_DISPATCH_SPEC=EXP005_COMMIT+':data/results/exp005/soft-penalty-beta-0.1/dispatch.npz'
EXP005_DISPATCH_SHA='06abf3df24461cab95916aa08ad25f93246f03d948f223e0233ffba8a0cfb5ea'
EXP005_DISPATCH_SNAPSHOT='reports/experiments/exp008/evidence/battery/exp005_primary_source.npz'
TEMPLATE_MAIN_COMMIT='7ecd9b988665ad628dbf048ed295457166170ffc'
TEMPLATE_SHA256={
    'reports/templates/report.md':'03c218e21c13b5f05419629cdc238b58dc0ef8f7ef39e0dd6855047ea76a5cb4',
    'reports/templates/battery-power.md':'31226bfbdccec7eec858711ace76c03e2ad1e36b929d18a40f071d94da6cc058',
    'reports/templates/history-comparison.md':'9bc10fd8bb5eef6934c8f2b9049310876309e2271c2fac115f7816f040c5931e',
}

def read_frozen_git_source(spec,expected_sha256,snapshot,*,root=ROOT):
    root=Path(root)
    git=subprocess.run(['git','show',spec],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    fallback=False
    if git.returncode==0:
        raw=git.stdout;source='git:'+spec
    else:
        commit=spec.split(':',1)[0]
        exists=subprocess.run(['git','cat-file','-e',commit+'^{commit}'],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if exists.returncode==0:
            raise RuntimeError(f'Historical Git commit exists but requested object cannot be read: {spec}: {git.stderr.decode(errors="replace")}')
        path=root/snapshot
        if not path.is_file():
            raise FileNotFoundError(f'Historical commit {commit} is unavailable and its required frozen snapshot is missing: {path}')
        raw=path.read_bytes();source='snapshot:'+snapshot;fallback=True
    actual=hashlib.sha256(raw).hexdigest()
    if actual!=expected_sha256:
        raise ValueError(f'Historical source SHA-256 mismatch for {source}: expected {expected_sha256}, got {actual}')
    if fallback:
        warnings.warn(f'Git source {spec} unavailable; using explicitly hash-verified frozen snapshot {snapshot}',RuntimeWarning,stacklevel=2)
    return raw,dict(source=source,sha256=actual,git_source_requested=spec,
        snapshot_fallback_used=fallback,snapshot=snapshot if fallback else None,
        original_source_bytes_verified=True)

def exp005_registry(root=ROOT):
    return read_frozen_git_source(EXP005_REGISTRY_SPEC,EXP005_REGISTRY_SHA,EXP005_REGISTRY_SNAPSHOT,root=root)

def exp005_dispatch(root=ROOT):
    return read_frozen_git_source(EXP005_DISPATCH_SPEC,EXP005_DISPATCH_SHA,EXP005_DISPATCH_SNAPSHOT,root=root)

def final_template(path,root=ROOT):
    """The latest main template verified at user-accepted finalization, now frozen."""
    return read_frozen_git_source(TEMPLATE_MAIN_COMMIT+':'+path,TEMPLATE_SHA256[path],path,root=root)

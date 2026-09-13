"""Portable historical evidence without fetching unrelated local branches."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import warnings

from experiments.exp008 import frozen_sources as sources

class FrozenHistoricalSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='exp008-frozen-source-')
        self.root=Path(self.tmp.name)
        subprocess.run(['git','init','--quiet',str(self.root)],check=True,capture_output=True)
        for relative in [sources.EXP005_REGISTRY_SNAPSHOT,sources.EXP005_DISPATCH_SNAPSHOT]:
            p=self.root/relative;p.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(sources.ROOT/relative,p)

    def tearDown(self): self.tmp.cleanup()

    def test_missing_commit_uses_explicit_hash_locked_registry_and_dispatch(self):
        self.assertNotEqual(subprocess.run(['git','cat-file','-e',sources.EXP005_COMMIT],cwd=self.root,capture_output=True).returncode,0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            registry,r=sources.exp005_registry(self.root)
            dispatch,d=sources.exp005_dispatch(self.root)
        self.assertEqual(json.loads(registry)['experiment_id'],'exp005')
        self.assertEqual(hashlib.sha256(dispatch).hexdigest(),sources.EXP005_DISPATCH_SHA)
        self.assertTrue(r['snapshot_fallback_used'] and d['snapshot_fallback_used'])
        self.assertEqual(len(caught),2)

    def test_changed_snapshot_is_rejected(self):
        (self.root/sources.EXP005_REGISTRY_SNAPSHOT).write_text('{"wrong":"source"}')
        with self.assertRaisesRegex(ValueError,'SHA-256 mismatch'):
            sources.exp005_registry(self.root)

    def test_missing_snapshot_is_rejected(self):
        (self.root/sources.EXP005_DISPATCH_SNAPSHOT).unlink()
        with self.assertRaisesRegex(FileNotFoundError,'snapshot is missing'):
            sources.exp005_dispatch(self.root)

    def test_templates_remain_locked_without_main_branch(self):
        for relative, expected in sources.TEMPLATE_SHA256.items():
            p=self.root/relative;p.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(sources.ROOT/relative,p)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                raw,meta=sources.final_template(relative,root=self.root)
            self.assertEqual(hashlib.sha256(raw).hexdigest(),expected)
            self.assertTrue(meta['snapshot_fallback_used'])
            self.assertEqual(len(caught),1)

    def test_payload_history_rebuild_works_without_exp005_git_commit(self):
        from experiments.exp008 import report_payload
        for n in (1,2,3,4,6):
            p=self.root/f'reports/registry/exp{n:03d}.json';p.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(sources.ROOT/p.relative_to(self.root),p)
        p=self.root/'data/results/exp008/baselines.json';p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(sources.ROOT/p.relative_to(self.root),p)
        previous=report_payload.ROOT
        try:
            report_payload.ROOT=self.root
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                issues=[];history=report_payload.history_payload(issues)
            self.assertEqual(issues,[])
            exp005=next(r for r in history['experiments'] if r['experiment_id']=='exp005')
            self.assertTrue(exp005['available'] and exp005['source']['snapshot_fallback_used'])
        finally: report_payload.ROOT=previous

if __name__=='__main__':unittest.main()

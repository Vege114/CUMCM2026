"""Restore duplicate January/evaluation slices omitted only from the small ZIP.

The 365-day arrays remain complete. No training or optimization is performed.
Every restored file must equal its original archived SHA-256 before publishing.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/results/exp009'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((OUT/'复现清单.json').read_text())
    restored = []
    for item in manifest['support_package']['derived_slices']:
        source, target = ROOT/item['source'], ROOT/item['target']
        assert digest(source) == manifest['trajectories'][item['source']], str(source)
        expected = manifest['trajectories'][item['target']]
        if target.exists():
            assert digest(target) == expected, f'Existing file differs; retained: {target}'
        else:
            with np.load(source, allow_pickle=False) as z:
                mask = z['days'] >= 31 if item['period'] == 'feb_dec' else z['days'] < 31
                arrays = {key:z[key][mask] for key in z.files}
            temporary = target.with_suffix('.restore.tmp')
            with temporary.open('wb') as handle:
                np.savez_compressed(handle, **arrays)
            assert digest(temporary) == expected, f'Restored hash differs: {target}'
            temporary.replace(target)
        restored.append({'path':item['target'], 'sha256':expected, 'passed':True})
    print(json.dumps({'passed':True, 'slices':restored}, ensure_ascii=False))


if __name__ == '__main__':
    main()

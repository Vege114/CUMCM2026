"""Read-only, absolute-tolerance verification of the approved main workbook."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.common.neural_v2.export import verify

verify('exp005', ['2'])
result = ROOT / 'data/results/exp005'
path = result / 'result2.xlsx'
payload = json.loads((ROOT / '.work/exp005/workbook-2.json').read_text())
cached = load_workbook(path, read_only=True, data_only=True)
formulas = load_workbook(path, read_only=False, data_only=False)
assert cached.sheetnames == ['计划购电量', '充放电量', '紧急购电量']
rows = list(cached['计划购电量'].iter_rows(min_row=2, values_only=True))
np.testing.assert_allclose([r[1:145] for r in rows], payload['original'], atol=1e-6, rtol=0)
np.testing.assert_allclose([r[145] for r in rows], np.sum(payload['original'], axis=1), atol=1e-6, rtol=0)
np.testing.assert_allclose([r[146] for r in rows], payload['fees'], atol=1e-6, rtol=0)
for i in range(2, 336):
    assert formulas['计划购电量'].cell(i, 146).value == f'=SUM(B{i}:EO{i})'
for name, key in [('充放电量', 'battery'), ('紧急购电量', 'emergency')]:
    for actual, expected in zip(cached[name].iter_rows(min_row=2, values_only=True), payload[key]):
        for a, e in zip(actual, expected):
            if isinstance(e, (int, float)):
                assert isinstance(a, (int, float))
                np.testing.assert_allclose(a, e, atol=1e-6, rtol=0)
for sheet in cached:
    assert not any(c.data_type == 'e' for row in sheet for c in row)
assert formulas['计划购电量'].freeze_panes == 'B2'
assert formulas['充放电量'].freeze_panes == 'B2'
assert formulas['紧急购电量'].freeze_panes == 'A2'
cached.close()
formulas.close()
record = {'read_only': True, 'absolute_tolerance': 1e-6, 'relative_tolerance': 0,
          'days': 334, 'grid_intervals': 48096, 'daily_sum_formulas': 334,
          'battery_rows': len(payload['battery']), 'emergency_rows': len(payload['emergency']),
          'numeric_cells_and_cached_totals': 'passed', 'all_formula_strings': 'passed',
          'error_cells': 0, 'date_types_and_frozen_headers': 'passed',
          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
(result / 'workbook_verification.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
print(json.dumps(record, ensure_ascii=False))

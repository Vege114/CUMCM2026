"""Read saved XLSX XML and independently rebuild every output cell from NPZ.

No workbook authoring library, report payload builder or optimizer is imported.
Use the bundled Python runtime for this read-only workbook analysis.
"""
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/experiments/exp009'
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def col(number):
    result = ''
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def interval(start, stop=None):
    stop = start + 1 if stop is None else stop
    clock = lambda slot: f'{slot // 6:02d}:{slot % 6 * 10:02d}'
    return f'{clock(start)}-{clock(stop)}'


def read_xlsx(path):
    with ZipFile(path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rels = {e.attrib['Id']: e.attrib['Target'] for e in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        sheets = {}
        for sheet in ET.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet', NS):
            target = rels[sheet.attrib[f'{{{NS["r"]}}}id']]
            source = target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('xl', target))
            root = ET.fromstring(z.read(source))
            cells, formulas = {}, {}
            for cell in root.findall('s:sheetData/s:row/s:c', NS):
                address, kind = cell.attrib['r'], cell.attrib.get('t')
                value = cell.find('s:v', NS)
                text = value.text if value is not None else None
                if kind == 'e':
                    raise AssertionError(('saved_formula_error', path, sheet.attrib['name'], address, text))
                if kind == 's':
                    parsed = strings[int(text)]
                elif kind == 'inlineStr':
                    parsed = ''.join(cell.find('s:is', NS).itertext())
                elif kind == 'b':
                    parsed = text == '1'
                elif kind in ('str', 'd'):
                    parsed = text
                else:
                    parsed = float(text) if text is not None else None
                cells[address] = parsed
                formula = cell.find('s:f', NS)
                if formula is not None:
                    formulas[address] = formula.text
            sheets[sheet.attrib['name']] = {'cells': cells, 'formulas': formulas,
                'merge_cells': [v.attrib['ref'] for v in root.findall('s:mergeCells/s:mergeCell', NS)]}
        return sheets


def audit(scenario, final_payload):
    result = final_payload['q1'] if scenario == '1' else final_payload['scenarios'][scenario]
    archive = ROOT / result['archive']['path']
    assert sha(archive) == result['archive']['sha256']
    file = OUT / f'result{scenario}.xlsx'
    sheets = read_xlsx(file)
    original_sheets = read_xlsx(ROOT / f'data/templates/result{scenario}.xlsx')
    assert list(sheets) == list(original_sheets), 'template sheet names/order changed'
    assert all(not row['merge_cells'] for row in sheets.values())
    count, max_error, formula_count = 0, 0., 0

    def compare(name, row, column, expected):
        nonlocal count, max_error
        count += 1
        address = f'{col(column)}{row}'
        actual = sheets[name]['cells'].get(address)
        if isinstance(expected, (int, float, np.number)):
            assert isinstance(actual, (int, float)) and np.isfinite(actual), (scenario, name, address, actual)
            error = abs(actual - float(expected))
            max_error = max(max_error, error)
            assert error < 1e-6, (scenario, name, address, actual, expected)
        else:
            assert actual == expected, (scenario, name, address, actual, expected)

    def compare_rows(name, rows):
        for row_index, values in enumerate(rows, 2):
            for column, value in enumerate(values, 1):
                compare(name, row_index, column, value)

    if scenario == '1':
        raw = json.loads(archive.read_text())
        assert len(raw['stages']) == 1 and raw['stages'][0]['stage'] == 1
        a = {k: np.asarray(v) for k, v in raw['trajectory'].items()}
        compare_rows('计划购电量', [[interval(t), a['g'][t]] for t in range(144)])
        battery = [[interval(b * 24, (b + 1) * 24),
                    float(a['c'][b * 24:(b + 1) * 24].sum() / 6),
                    float(a['d'][b * 24:(b + 1) * 24].sum() / 6),
                    '0:00' if b == 0 else '24:00' if b == 1 else None,
                    a['E'][0 if b == 0 else -1] if b < 2 else None] for b in range(6)]
        compare_rows('充放电量', battery)
        assert all(not sheet['formulas'] for sheet in sheets.values())
        energy = float(a['g'].sum())
        days, emergency_rows = 1, 0
    else:
        with np.load(archive, allow_pickle=False) as z:
            a = {k: z[k].copy() for k in z.files}
        assert a['original'].shape == (334, 144)
        dates = [date(2025, 2, 1) + timedelta(days=i) for i in range(334)]
        serials = [(value - date(1899, 12, 30)).days for value in dates]
        for name, key in [('计划购电量', 'original')] + ([('调整购电量', 'final')] if scenario in ('3','4-3') else []):
            for t in range(144):
                compare(name, 1, t + 2, interval(t))
            for day in range(334):
                values = [serials[day], *a[key][day], float(a[key][day].sum()), float(a['fees'][day].sum())]
                for c, value in enumerate(values, 1):
                    compare(name, day + 2, c, value)
                assert sheets[name]['formulas'][f'EP{day + 2}'] == f'SUM(B{day + 2}:EO{day + 2})'
                formula_count += 1
            assert len(sheets[name]['formulas']) == 334
        battery, emergency = [], []
        for day in range(334):
            for block in range(6):
                sl = slice(block * 24, (block + 1) * 24)
                battery.append([serials[day] if block == 0 else None, interval(block * 24, (block + 1) * 24),
                                float(a['charge'][day, sl].sum()), float(a['discharge'][day, sl].sum()),
                                '0:00' if block == 0 else '24:00' if block == 1 else None,
                                a['states'][day, 0 if block == 0 else -1] if block < 2 else None])
            positive = np.flatnonzero(a['emergency'][day] > 1e-6)
            groups = np.split(positive, np.flatnonzero(np.diff(positive) > 1) + 1) if len(positive) else []
            if not groups:
                emergency.append([serials[day], '无', 0.])
            for i, group in enumerate(groups):
                emergency.append([serials[day] if i == 0 else None,
                                  interval(int(group[0]), int(group[-1] + 1)),
                                  float(a['emergency'][day, group].sum())])
        compare_rows('充放电量', battery)
        compare_rows('紧急购电量', emergency)
        assert len(battery) == 2004
        assert not sheets['充放电量']['formulas'] and not sheets['紧急购电量']['formulas']
        energy, days, emergency_rows = float(a['original'].sum()), 334, len(emergency)
    duplicate = ROOT / 'outputs/exp009-cost-only-control' / file.name
    assert sha(file) == sha(duplicate)
    return {'scenario': scenario, 'passed': True, 'path': str(file.relative_to(ROOT)),
            'sha256': sha(file), 'source_archive': result['archive'],
            'days': days, 'scalar_cells_compared_to_independently_rebuilt_source': count,
            'max_absolute_numeric_error': max_error, 'sum_formulas_verified': formula_count,
            'emergency_event_rows_reconstructed': emergency_rows,
            'original_purchase_total_kwh': energy,
            'template_sheet_names_and_order_preserved': True, 'unexpected_formula_errors': 0,
            'no_double_kW_to_kWh_conversion': True,
            'repeated_fee_column_not_double_counted': True, 'duplicate_bytes_identical': True}


def main():
    payload_path = OUT / 'final_payload.json'
    final_payload = json.loads(payload_path.read_text())
    assert final_payload['experiment_id'] == 'exp009'
    files = [audit(s, final_payload) for s in ('1', '2', '3', '4-2', '4-3')]
    evidence = {'passed': True, 'files': files, 'final_payload_sha256': sha(payload_path),
                'method': 'Independent Python standard-library XLSX XML reader and NumPy archive arithmetic. No authoring, payload-building, prediction or optimizer functions called.',
                'total_scalar_cells': sum(r['scalar_cells_compared_to_independently_rebuilt_source'] for r in files),
                'visual_and_live_formula_recalculation_evidence': 'saved_workbook_audit.json and visual_review.json'}
    target = OUT / 'evidence/workbooks/independent_cell_audit.json'
    target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'passed': True, 'files': len(files), 'total_scalar_cells': evidence['total_scalar_cells']}))


if __name__ == '__main__':
    main()

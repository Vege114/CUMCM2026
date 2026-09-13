"""Independent numerical and artifact checks for this paper-figure delivery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'data/results/exp008/robustness'
REPORT=ROOT/'reports/experiments/exp008/robustness'
FIG=ROOT/'paper/figures/exp008_robustness'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks={}
    for section,key in [('q1','source_hashes'),('q1','protected_hashes'),
                         ('forecast','input_sha256'),('execution','sources_sha256')]:
        protocol=json.loads((DATA/section/'protocol.json').read_text())
        for path,value in protocol[key].items():
            assert sha(ROOT/path)==value,(section,path)
        checks[section+'_'+key]=len(protocol[key])
    selection=json.loads((ROOT/'experiments/exp008/final_selection.json').read_text())
    assert sha(ROOT/selection['scenarios']['2']['archive'])==selection['finalization_acceptance']['accepted_q2_archive_sha256']
    q1=json.loads((DATA/'q1/summary.json').read_text())
    assert q1['center_reproduction']['passed'] and q1['center_reproduction']['max_trajectory_absolute_difference']==0
    q1_rows=pd.read_csv(DATA/'q1/sensitivity.csv')
    assert len(q1_rows)==15 and q1_rows.eligible_for_comparison.all() and (q1_rows.violations==0).all()
    raw=np.stack([pd.read_csv(ROOT/'data/raw'/name).iloc[31:,1:].to_numpy(float)
        for name in ['附件2_小区负载.csv','附件2_光伏发电实际功率.csv']],axis=-1)
    annual=pd.read_csv(DATA/'forecast/sensitivity_annual.csv')
    differences=[]
    for name,group in annual.groupby('name'):
        with np.load(DATA/'forecast'/f'{name}.npz') as z:
            values=z['values']
        for row in group.itertuples():
            if row.channel=='net':
                error=(values[:,:,0]-values[:,:,1])-(raw[:,:,0]-raw[:,:,1])
            else:
                c=0 if row.channel=='load' else 1
                error=values[:,:,c]-raw[:,:,c]
            measured=[np.sqrt(np.mean(error**2)),np.mean(np.abs(error)),
                np.sqrt(np.mean((error.sum(axis=1)/6)**2)),
                np.sqrt(np.mean((np.cumsum(error,axis=1)/6)**2))]
            recorded=[row.rmse_kw,row.mae_kw,row.daily_energy_rmse_kwh,row.cumulative_error_rmse_kwh]
            np.testing.assert_allclose(measured,recorded,rtol=1e-12,atol=1e-8)
            differences.extend(np.abs(np.asarray(measured)-recorded))
    checks['independent_forecast_metric_rows']=len(annual)
    checks['max_independent_forecast_metric_difference']=float(max(differences))
    cv=json.loads((DATA/'forecast/cv_training_audit.json').read_text())
    for row in cv:
        assert row['train_label_stop_exclusive']<=row['validation_start_day']*144
        assert row['validation_label_stop_exclusive']<=row['first_test_origin']
        assert row['genuine_refit']
    cross=json.loads((DATA/'forecast/hyperparameter_cv_input_crosscheck.json').read_text())
    assert cross['passed']
    checks['genuine_cv_fits']=len(cv)
    checks['q1_center_and_all15_physical_checks']=True
    checks['hyperparameter_independent_input_hash_crosscheck']=True
    temporal=json.loads((DATA/'temporal/summary.json').read_text())
    assert temporal['all_independent_checks_passed']
    assert temporal['csv_readback_additive_checks_passed']
    checks['four_scenarios_independent_physics_billing']=True
    execution=json.loads((DATA/'execution/summary.json').read_text())
    assert execution['passed'] and execution['all_physics_passed'] and execution['future_actual_prefix_mutation_passed']
    checks['execution_conditional_records']=execution['n_scenarios']
    figures=json.loads((FIG/'figure_manifest.json').read_text())
    assert len(figures)==15
    dimensions={}
    for row in figures:
        for source in row['sources']:
            assert (DATA/source).is_file(),source
        for ext,path in row['outputs'].items():
            p=ROOT/path
            assert p.stat().st_size>1000
            if ext=='png':
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    assert min(im.size)>1800
                    assert abs(im.info['dpi'][0]-450)<.1
                    dimensions[row['id']]=list(im.size)
            elif ext=='svg':
                assert ET.parse(p).getroot().tag.endswith('svg')
            else:
                info=subprocess.check_output(['pdfinfo',str(p)],text=True)
                assert re.search(r'Pages:\s+1\b',info)
                assert b'/FontFile2' in p.read_bytes(), 'PDF font not embedded'
    info=subprocess.check_output(['pdfinfo',str(FIG/'all_figures.pdf')],text=True)
    assert re.search(r'Pages:\s+15\b',info)
    broken=[]
    for path in [REPORT/'README.md',REPORT/'captions.md',FIG/'README.md']:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if target=='verification.json':
                continue
            if not (path.parent/target).resolve().exists():
                broken.append([str(path),target])
    assert not broken,broken
    checks['figure_groups']=15
    checks['image_files']=45
    checks['pdf_collection_pages']=15
    checks['png_dpi']=450
    checks['png_dimensions']=dimensions
    checks['local_links_resolved']=True
    result={'passed':True,'checks':checks,
        'visual_review':{'all15_figures_contact_sheet_inspected':True,
                         'representative_pdf_rasterization_inspected':'02_temporal_cv.pdf',
                         'repairs':['Pressure-group axis expanded to include all data points.',
                                    'Bootstrap legend moved away from interval bars.',
                                    'Q1 numerical-zero cost changes displayed as zero at stated tolerance.',
                                    'Q1 TV, combined smoothness and starts shown separately.']},
        'limitations':['Charts use only reviewed development-period evidence; no independent external-test claim.',
                       'Visual review is contact-sheet coverage plus representative full-size PDF, not all-format pixel-by-pixel equivalence.'],
        'verified_figure_sha256':{p:sha(ROOT/p) for row in figures for p in row['outputs'].values()}}
    (REPORT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'passed':True,'figure_groups':15,'metrics_recomputed':len(annual)},ensure_ascii=False))


if __name__=='__main__':
    main()

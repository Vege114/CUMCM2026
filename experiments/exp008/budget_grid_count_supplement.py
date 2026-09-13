"""Threshold tiny numerical flows only for the grid diagnostic episode counts."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.budget_grid_diagnostic import BASE, OUT, DAYS
from experiments.exp008.verify import TOL


def main():
    paths = [OUT/'matched_metrics.csv', OUT/'deltas.csv']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    frame = pd.read_csv(paths[0]); audit = []
    for day in DAYS:
        p = np.load(BASE/f'day{day}_inputs.npz')
        previous = float(p['initial_power_kw'])/6
        first_mode = int(np.sign(previous)) if abs(previous) > TOL else 0
        for resolution, path in ((200, BASE/f'day{day}_initial_actual_replay.npz'),
                                 (100, OUT/f'day{day}_actual_replay.npz')):
            flow = np.load(path)
            net = (flow['charge']-flow['discharge']).ravel()
            modes = np.where(np.abs(net) > TOL, np.sign(net), 0).astype(int)
            episodes = int(np.sum((modes != 0) & (modes != np.r_[first_mode, modes[:-1]])))
            match = (frame.day == day) & (frame.grid_kwh == resolution)
            old = int(frame.loc[match, 'activity_episode_starts'].iloc[0])
            frame.loc[match, 'activity_episode_starts'] = episodes
            audit.append({'day': day, 'grid_kwh': resolution, 'old_unthresholded_episodes': old,
                          'corrected_episodes': episodes, 'subtolerance_nonzero_slots': int(np.sum((net != 0) & (np.abs(net) <= TOL)))})
    frame.to_csv(OUT/'matched_metrics_thresholded.csv', index=False)
    delta = frame[frame.grid_kwh == 100].set_index('day').drop(columns='grid_kwh')-frame[frame.grid_kwh == 200].set_index('day').drop(columns='grid_kwh')
    delta.to_csv(OUT/'deltas_thresholded.csv')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    report = {'reason': 'original diagnostic episode counter used sign without1e-6kWh tolerance; numerical residual flows could inflate episodes',
        'threshold_kwh': TOL, 'original_files_preserved_sha256': hashes,
        'physical_arrays_purchases_and_cash_fees_unchanged': True, 'counts': audit,
        'use_for_episode_comparison': 'matched_metrics_thresholded.csv/deltas_thresholded.csv'}
    (OUT/'episode_count_erratum.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()

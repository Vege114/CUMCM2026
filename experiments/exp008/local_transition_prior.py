"""Fixed one-hour local-clock shrinkage prior for causal residual transitions.

This changes only the transition prior at t >= 1. Conditional residual
emissions, current-observation bins and midnight initialization stay fixed.
The score diagnostic is model development, not a bill or a control result.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp008.markov_tail_feedback import fit_error_model

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'data/results/exp008/mode_budget_hold1_physical/direct_hgb_memory_hold1_cap8_kappa0_334days'
OUT = ROOT / 'data/results/exp008/local_transition_prior'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_clock_model(net, errors, origins, cutoff, radius=6):
    baseline = fit_error_model(net, errors, points=3, history_origins=origins, cutoff=cutoff)
    model = {k: v.copy() if hasattr(v, 'copy') else v for k, v in baseline.items()}
    bins = np.searchsorted(baseline['state_edges'], (errors - baseline['mean_kwh']) / baseline['scale_kwh'], side='right')
    counts = np.zeros((len(net), 3, 3))
    for t in range(1, len(net)):
        np.add.at(counts[t], (bins[:, t-1], bins[:, t]), 1.)
    for t in range(1, len(net)):
        # A truncated clock-time window, not a window of unseen current-day data.
        left, stop = max(1, t-radius), min(len(net), t+radius+1)
        pooled = counts[left:stop].sum(axis=0)
        next_prior = baseline['marginal_probabilities'][left:stop].mean(axis=0)
        prior = (pooled + 3 * next_prior[None, :]) / (pooled.sum(axis=1, keepdims=True) + 3)
        model['transition'][t] = (counts[t] + 12 * prior) / (counts[t].sum(axis=1, keepdims=True) + 12)
    model['metadata'] = {**baseline['metadata'], 'transition_prior': 'local clock +/-6 ten-minute slots, truncated at boundaries',
        'local_clock_radius_slots': radius, 'current_day_observations_in_fit': False,
        'midnight_transition_unchanged': True, 'same_slot_local_counts_also_enter_prior': True}
    return baseline, model


def score(model, actual_net):
    bins = np.searchsorted(model['state_edges'], (actual_net-model['forecast_net_kwh']-model['mean_kwh']) / model['scale_kwh'], side='right')
    previous = np.r_[model['initial_error_state'], bins[:-1]]
    probabilities = model['transition'][np.arange(len(bins)), previous]
    realized = probabilities[np.arange(len(bins)), bins]
    assert np.all(realized > 0) and np.allclose(probabilities.sum(axis=1), 1.)
    return -np.log(realized), np.sum((probabilities-np.eye(3)[bins])**2, axis=1), bins


def run():
    OUT.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__), ROOT/'experiments/exp008/markov_feedback.py', ROOT/'experiments/exp008/markov_tail_feedback.py']
    source_hashes = {}
    for path in source_files:
        shutil.copy2(path, OUT/path.name)
        source_hashes[str(path.relative_to(ROOT))] = sha(path)
    protocol = {'candidate_count': 1, 'clock_radius_slots': 6, 'shrinkage': 12, 'pooled_smoothing': 3,
        'model': 'original absolute HGB Ridge28 memory half', 'days': list(range(31, 365)),
        'forecast_and_emissions_unchanged': True, 'prespecified_next_step': 'fixed-five DP control diagnostic only if both full-year one-step log loss and Brier improve',
        'all_models_locked_before_score_actual_read': True, 'current-day controls_not_optimized': True,
        'development_year_not_independent_test': True, 'source_sha256': source_hashes}
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    with np.load(BASE/'issued_forecasts.npz') as z:
        forecast = (z['values'][:, :, 0]-z['values'][:, :, 1])/6
        np.testing.assert_array_equal(z['origins'], np.arange(31, 365)*144)
    pairs, archive_hashes = [], {str(BASE/'issued_forecasts.npz'): sha(BASE/'issued_forecasts.npz')}
    for i, day in enumerate(range(31, 365)):
        path = BASE/f'planning_day{day}.npz'
        with np.load(path) as z:
            errors = z['all_net_paths']-forecast[i][None, :]
        archive_hashes[str(path)] = sha(path)
        old, new = local_clock_model(forecast[i], errors, np.arange(day-28, day)*144, day*144)
        np.testing.assert_array_equal(old['transition'][0], new['transition'][0])
        for key in old:
            if key not in ('transition', 'metadata'):
                np.testing.assert_array_equal(old[key], new[key])
        # Independent scalar reconstruction of one local prior row per slot.
        bins = np.searchsorted(old['state_edges'], (errors-old['mean_kwh'])/old['scale_kwh'], side='right')
        for t in range(1, 144):
            last = t % 3
            times = range(max(1, t-6), min(144, t+7))
            pooled = np.array([sum(int(bins[j,s-1] == last and bins[j,s] == k) for s in times for j in range(28)) for k in range(3)])
            nxt = old['marginal_probabilities'][list(times)].mean(axis=0)
            prior = (pooled + 3*nxt)/(pooled.sum()+3)
            local = np.array([sum(int(bins[j,t-1] == last and bins[j,t] == k) for j in range(28)) for k in range(3)])
            np.testing.assert_allclose(new['transition'][t,last], (local+12*prior)/(local.sum()+12), rtol=0, atol=1e-12)
        pairs.append((old, new))
    locked = {'mean': np.stack([p[0]['mean_kwh'] for p in pairs]), 'scale': np.stack([p[0]['scale_kwh'] for p in pairs]),
        'forecast_net': forecast, 'baseline_transition': np.stack([p[0]['transition'] for p in pairs]),
        'local_transition': np.stack([p[1]['transition'] for p in pairs]), 'initial_bin': np.array([p[0]['initial_error_state'] for p in pairs])}
    np.savez_compressed(OUT/'locked_models.npz', **locked)
    (OUT/'models_locked_before_scoring.json').write_text(json.dumps({'all334_models_complete': True,
        'models_sha256': sha(OUT/'locked_models.npz'), 'input_files_sha256': archive_hashes,
        'score_actual_archive_loaded': False}, indent=2)+'\n')
    with np.load(BASE/'dispatch.npz') as z:
        actual = (z['actual'][:, :, 0]-z['actual'][:, :, 1])/6
        prices = z['price'].copy()
    rows, old_losses, new_losses, old_briers, new_briers = [], [], [], [], []
    for i, (old, new) in enumerate(pairs):
        ol, ob, old_bins = score(old, actual[i]); nl, nb, new_bins = score(new, actual[i])
        np.testing.assert_array_equal(old_bins, new_bins)
        for stop in (1, 36, 108):
            changed = actual[i].copy(); changed[stop:] += 1e5
            for model, expected in [(old, ol), (new, nl)]:
                np.testing.assert_array_equal(score(model, changed)[0][:stop], expected[:stop])
        rows.append({'day': i+31, 'baseline_log_loss': ol.mean(), 'local_log_loss': nl.mean(),
            'baseline_brier': ob.mean(), 'local_brier': nb.mean(), 'high_price_baseline_log_loss': ol[prices[i]>=1].mean(),
            'high_price_local_log_loss': nl[prices[i]>=1].mean()})
        old_losses.append(ol); new_losses.append(nl); old_briers.append(ob); new_briers.append(nb)
    frame = pd.DataFrame(rows); frame.to_csv(OUT/'daily_scores.csv', index=False)
    summary = {'complete': True, 'days': 334, 'slots': 48096, 'baseline_log_loss': float(np.mean(old_losses)),
        'local_log_loss': float(np.mean(new_losses)), 'baseline_brier': float(np.mean(old_briers)), 'local_brier': float(np.mean(new_briers)),
        'both_scores_improve': bool(np.mean(new_losses)<np.mean(old_losses) and np.mean(new_briers)<np.mean(old_briers)),
        'independent_scalar_prior_rows_checked': 334*143, 'future_score_prefix_checks': 334*3*2,
        'all_other_model_arrays_unchanged': True, 'source_unchanged': all(sha(ROOT/p)==s for p,s in source_hashes.items()),
        'billing_or_battery_improvement_claim': False}
    assert summary['source_unchanged']
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    run()

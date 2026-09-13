"""Past complete-day residual paths matched to the current release adapter.

The old midnight-only helper is retained in forecast.py for frozen controls.
Here a 12:00 forecast receives errors of prior days' 12:00 forecasts, including
the same prefix correction and PV/price release conventions. This does not
condition on any unreleased observation from the current day.
"""
import numpy as np


def issued_error_paths(forecasts, day, slot=0, scenario='3', limit=28):
    """Return history x (144-slot) errors in AC kWh, and matched price errors.

    ``forecasts`` supports get(day,slot,scenario) and the bounded reader
    _observed(start,stop,cutoff). All historical target days are complete before
    the current midnight; slot 0 agrees exactly with net_error_paths.
    """
    if (not isinstance(day, (int, np.integer)) or not 1 <= day < 365
            or slot not in (0, 36, 72, 108) or not isinstance(limit, int) or limit < 1):
        raise ValueError('Need day 1..364, legal release slot and positive history limit')
    if scenario not in ('2', '3', '4-2', '4-3') or (scenario in ('2', '4-2') and slot):
        raise ValueError('Midnight-only scenarios cannot use intraday residuals')
    day, slot = int(day), int(slot)
    cutoff = day*144+slot
    history = np.arange(max(1, day-limit), day, dtype=int)
    if not len(history):
        raise ValueError('No complete historical target day is available')
    kw, prices = [], []
    for old in history:
        begin, end = int(old)*144+slot, (int(old)+1)*144
        if end > day*144:
            raise ValueError('Historical target day is incomplete')
        prediction = forecasts.get(int(old), slot=slot, scenario=scenario)
        actual = forecasts._observed(begin, end, cutoff)
        kw.append(actual[:, :2]-np.column_stack((prediction['load_kw'], prediction['pv_kw'])))
        prices.append(actual[:, 2]-prediction['price'])
    kw = np.asarray(kw)
    origins = history*144+slot
    return {
        'errors_kwh': (kw[:, :, 0]-kw[:, :, 1])/6,
        'errors_kw': kw,
        'price_errors': np.asarray(prices),
        'origins': origins,
        'label_stops_exclusive': (history+1)*144,
        'audit': {
            'information_cutoff_exclusive': cutoff,
            'max_observed_index': int((history[-1]+1)*144-1),
            'training_origins': origins.tolist(),
            'residual_issue_slot': slot,
            'label_stops_exclusive': ((history+1)*144).tolist(),
            'fallback_days': history[history < 31].tolist(),
            'semantics': 'same_release_adapter_residuals_of_prior_complete_target_days',
            'price_errors_match_same_release': True,
        },
    }

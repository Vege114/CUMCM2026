"""Causal, lightweight tree-conditioned net-load residual distributions.

The frozen exp004 load/PV forecasts are inputs, in kW; this module does not
change or regenerate them. ``for_day`` returns per-slot net-load supports in
kWh for ten-minute intervals. The nine columns are conditional marginal
quantiles with equal weights, not nine temporally correlated daily paths.

Historical forecast labels are read only after their complete day is known.
When fewer than two issued forecast days are available, the explicitly
identified periodic-baseline fallback uses only previously completed days.
That fallback is not represented as an exp004 forecast-error distribution.
"""

from time import perf_counter

import numpy as np
from sklearn.tree import DecisionTreeRegressor

STEPS = 144
DT_HOURS = 1.0 / 6.0
HISTORY_DAYS = 28
SUPPORT_SIZE = 9
QUANTILE_LEVELS = (np.arange(SUPPORT_SIZE, dtype=float) + 0.5) / SUPPORT_SIZE
FEATURE_NAMES = ("slot_sin", "slot_cos", "forecast_net_kwh", "forecast_pv_kwh", "price")


def tree_features(forecast_kw, price144):
    """Pure (144, 5) features; accepts forecasts/prices only, never actuals.

    Column order: slot sine/cosine, predicted load minus PV in kWh, predicted
    PV in kWh, and the fixed tariff in yuan/kWh. Slot zero starts at midnight.
    """
    forecast = np.asarray(forecast_kw, dtype=float)
    price = np.asarray(price144, dtype=float)
    if forecast.shape != (STEPS, 2) or price.shape != (STEPS,):
        raise ValueError("forecast_kw must be (144,2), price144 must be (144,)")
    if not np.isfinite(forecast).all() or not np.isfinite(price).all():
        raise ValueError("forecasts and prices must be finite")
    phase = np.arange(STEPS, dtype=float) * (2.0 * np.pi / STEPS)
    return np.column_stack((
        np.sin(phase), np.cos(phase),
        (forecast[:, 0] - forecast[:, 1]) * DT_HOURS,
        forecast[:, 1] * DT_HOURS, price,
    ))


def periodic_baseline(previous_day_kw, previous_week_kw=None):
    """Pure causal fallback: mean yesterday/week load; yesterday PV.

    The caller must supply only completed days preceding the issue date.
    Before one full week exists, the load prediction is yesterday's load.
    """
    yesterday = np.asarray(previous_day_kw, dtype=float)
    weekly = yesterday if previous_week_kw is None else np.asarray(previous_week_kw, dtype=float)
    if yesterday.shape != (STEPS, 2) or weekly.shape != (STEPS, 2):
        raise ValueError("baseline inputs must each contain one complete (144,2) day")
    if not np.isfinite(yesterday).all() or not np.isfinite(weekly).all():
        raise ValueError("baseline history must be finite")
    return np.column_stack((0.5 * (yesterday[:, 0] + weekly[:, 0]), yesterday[:, 1]))


class TreeResidualScenarios:
    """Fit a small residual tree independently at each requested midnight.

    Parameters are frozen forecast ``origins`` (interval indices), forecast
    ``values`` shaped (number of origins, 144, 2), ``actual`` shaped
    (number of complete days * 144, 2), and fixed daily ``price144``. The
    application supplies the 365-day actual archive; shorter archives also
    support small synthetic checks. Days are zero-based (February 1 is 31).

    Actuals are retained as an array reference without scanning or copying
    their contents during construction. Only selected completed-day slices
    are read later. Treat all constructor inputs as immutable thereafter.
    """

    def __init__(self, origins, values, actual, price144):
        origins = np.asarray(origins)
        if origins.ndim != 1 or origins.dtype.kind not in "iu":
            raise ValueError("origins must be a one-dimensional integer array")
        if np.any(origins < 0) or np.any(origins % STEPS):
            raise ValueError("origins must be nonnegative midnight interval indices")
        if len(np.unique(origins)) != len(origins):
            raise ValueError("forecast origins must be unique")
        forecast = np.asarray(values)
        if forecast.shape != (len(origins), STEPS, 2):
            raise ValueError("values must have shape (len(origins),144,2)")
        if not np.isfinite(forecast).all():
            raise ValueError("frozen forecasts must be finite")
        observed = np.asarray(actual)
        if observed.ndim != 2 or observed.shape[1] != 2 or observed.shape[0] % STEPS:
            raise ValueError("actual must contain complete days, shaped (days*144,2)")
        if np.any(origins + STEPS > observed.shape[0]):
            raise ValueError("forecast origins must fit the actual archive's calendar")
        price = np.asarray(price144, dtype=float)
        if price.shape != (STEPS,) or not np.isfinite(price).all():
            raise ValueError("price144 must contain 144 finite tariff values")
        order = np.argsort(origins)
        self.origins = origins[order].astype(np.int64, copy=True)
        self.values = np.asarray(forecast[order], dtype=float)
        self.price144 = price.copy()
        self._actual = observed
        self._lookup = {int(origin): index for index, origin in enumerate(self.origins)}
        self.weights = np.full(SUPPORT_SIZE, 1.0 / SUPPORT_SIZE)

    def _completed_day(self, day, cutoff):
        """The sole actual-value reader; enforce an exclusive information cut."""
        start, stop = int(day) * STEPS, (int(day) + 1) * STEPS
        if start < 0 or stop > cutoff:
            raise ValueError("attempt to read a day not complete at the information cutoff")
        observed = np.asarray(self._actual[start:stop], dtype=float)
        if observed.shape != (STEPS, 2) or not np.isfinite(observed).all():
            raise ValueError("selected completed actual day must be finite and complete")
        return observed

    def for_day(self, day):
        """Return ``(net_load_kwh[144,9], audit_info)`` before ``day`` starts.

        At most the 28 most recent complete issued forecast days are used.
        If there are fewer than two, use up to 28 known periodic-baseline
        error days instead. A day-zero request without any calibration data
        raises instead of inventing a residual distribution.
        """
        if isinstance(day, (bool, np.bool_)) or not isinstance(day, (int, np.integer)):
            raise TypeError("day must be a zero-based integer day index")
        day = int(day)
        cutoff = day * STEPS
        if cutoff not in self._lookup:
            raise ValueError("requested day has no frozen forecast")
        historical_ids = np.flatnonzero(self.origins + STEPS <= cutoff)[-HISTORY_DAYS:]
        fallback = len(historical_ids) < 2
        features, residuals, training_days = [], [], []

        if fallback:
            # The target day d is already known at today's cutoff; the
            # baseline issued on d still sees only indices below d*144.
            for historical_day in range(max(1, day - HISTORY_DAYS), day):
                issue_cutoff = historical_day * STEPS
                previous = self._completed_day(historical_day - 1, issue_cutoff)
                weekly = (self._completed_day(historical_day - 7, issue_cutoff)
                          if historical_day >= 7 else None)
                predicted = periodic_baseline(previous, weekly)
                observed = self._completed_day(historical_day, cutoff)
                features.append(tree_features(predicted, self.price144))
                residuals.append(((observed[:, 0] - observed[:, 1])
                                  - (predicted[:, 0] - predicted[:, 1])) * DT_HOURS)
                training_days.append(historical_day)
        else:
            for index in historical_ids:
                historical_day = int(self.origins[index] // STEPS)
                predicted = self.values[index]
                observed = self._completed_day(historical_day, cutoff)
                features.append(tree_features(predicted, self.price144))
                residuals.append(((observed[:, 0] - observed[:, 1])
                                  - (predicted[:, 0] - predicted[:, 1])) * DT_HOURS)
                training_days.append(historical_day)

        if not training_days:
            raise ValueError("no completed day exists to calibrate even the periodic fallback")
        x_train, y_train = np.concatenate(features), np.concatenate(residuals)
        tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=48, random_state=42)
        began = perf_counter()
        tree.fit(x_train, y_train)
        fit_seconds = perf_counter() - began
        train_leaves = tree.apply(x_train)
        leaf_supports = {
            int(leaf): np.quantile(y_train[train_leaves == leaf], QUANTILE_LEVELS, method="linear")
            for leaf in np.unique(train_leaves)
        }
        current_forecast = self.values[self._lookup[cutoff]]
        x_current = tree_features(current_forecast, self.price144)
        current_leaves = tree.apply(x_current)
        errors = np.vstack([leaf_supports[int(leaf)] for leaf in current_leaves])
        scenarios = x_current[:, 2, None] + errors
        info = {
            "day": day,
            "cutoff": cutoff,
            "information_cutoff": cutoff,
            "cutoff_is_exclusive": True,
            "training_start_day": min(training_days),
            "training_end_day": max(training_days),
            "training_days": len(training_days),
            "training_origins": [int(value * STEPS) for value in training_days],
            "sample_count": len(y_train),
            "issued_history_days": len(historical_ids),
            "max_observed_index": int((max(training_days) + 1) * STEPS - 1),
            "fit_seconds": float(fit_seconds),
            "tree_nodes": int(tree.tree_.node_count),
            "tree_leaves": int(tree.tree_.n_leaves),
            "fallback": bool(fallback),
            "residual_source": "periodic_baseline" if fallback else "frozen_exp004_forecast",
            "fallback_reason": "fewer_than_two_completed_issued_forecast_days" if fallback else None,
            "fallback_formula": "load=(yesterday+previous_week)/2; pv=yesterday; early_week_load=yesterday" if fallback else None,
            "fallback_same_forecast_source": False if fallback else None,
            "feature_names": list(FEATURE_NAMES),
            "residual_units": "kWh_per_10min",
            "scenario_units": "kWh_per_10min",
            "scenario_semantics": "conditional_slot_marginals_not_joint_daily_paths",
            "quantile_levels": QUANTILE_LEVELS.tolist(),
            "weights": self.weights.tolist(),
            "max_depth": 5,
            "min_samples_leaf": 48,
            "random_state": 42,
        }
        return scenarios, info

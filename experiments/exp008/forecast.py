"""Shared, issue-time forecasts for all exp008 annual scenarios.

Q2 midnight load/PV are exactly the frozen exp004 no-season CNN output.
Q3 augments this common forecast with observed-prefix load correction and
the current official PV issue. Q4 prices are forecast from past prices.
No decision feature or calibration label reads an actual at/after its issue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common.neural_v2.data import Data, ROOT
from experiments.problem2.exp004.predict import ForecastStore

STEPS = 144
SCENARIOS = ("2", "3", "4-2", "4-3")
ISSUE_SLOTS = (0, 36, 72, 108)
OUT = ROOT / "data/results/exp008"


def _integration_basis():
    """Linear hourly-knot corrections -> ten-minute interval averages.

    The observed issue-time anchor is fixed, so its correction is zero.
    Fitting in interval space avoids treating hourly forecasts as averages.
    """
    basis = np.zeros((144, 24))
    for h in range(24):
        knot = np.zeros(25)
        knot[h + 1] = 1
        right = np.interp(np.arange(1, 145), np.arange(0, 145, 6), knot)
        basis[:, h] = (np.r_[0., right[:-1]] + right) / 2
    return basis


class Forecasts:
    """Read-only forecast adapter; arrays returned by ``get`` are copies.

    ``day`` is zero based (31 = February 1); ``slot`` is a ten-minute index.
    Formal forecasts cover the remaining natural day, never a future day.
    January is an explicitly labelled causal periodic cold start.
    """

    def __init__(self, data=None, seed=42):
        self.data = Data() if data is None else data
        self.seed = int(seed)
        self.store = ForecastStore("no_season", seed=self.seed)
        self._basis = _integration_basis()
        difference = np.diff(np.eye(24), n=2, axis=0)
        # Fixed regularization: smooth adjacent hourly bias estimates.
        self._pv_penalty = 12 * np.einsum("ni,nj->ij", difference, difference,
                                        optimize=False) + 2 * np.eye(24)

    def _observed(self, start, stop, cutoff):
        if not 0 <= start <= stop <= cutoff:
            raise ValueError("actual reader requires [start,stop) strictly before issue")
        return np.asarray(self.data.actual[start:stop], float)

    @lru_cache(maxsize=366)
    def _midnight(self, day):
        origin = day * STEPS
        if day >= 31:
            return self.store.get(origin)
        # Appendix 1 is known before operation and supplies the day-zero seed.
        target = origin + np.arange(STEPS)
        value = self.data.reference[:, [1, 2]].copy()
        for channel, lag in ((0, 7), (1, 1)):
            ids = target - lag * STEPS
            if lag == 7 and day < 7:
                ids = target - STEPS
            known = (ids >= 0) & (ids < origin)
            if known.any():
                value[known, channel] = self._observed(int(ids[known][0]),
                    int(ids[known][-1]) + 1, origin)[:, channel]
        return value

    def _integrate(self, hourly, origin, cutoff):
        if origin > cutoff:
            raise ValueError("unreleased official forecast")
        anchor = self._observed(origin - 1, origin, cutoff)[0, 1] if origin else 0.
        right = np.interp(np.arange(1, 145), np.arange(0, 145, 6),
                          np.r_[anchor, np.maximum(hourly, 0)])
        return (np.r_[anchor, right[:-1]] + right) / 2

    @lru_cache(maxsize=1500)
    def _pv(self, origin):
        # Same release hour and lead horizon; every calibration horizon has
        # completely arrived, including yesterday's 24-hour horizon.
        history = np.arange(max(origin % 144, origin - 28 * 144), origin, 144)
        history = history[history + 144 <= origin]
        residuals = []
        for old in history:
            raw = self._integrate(self.data.forecasts[int(old)], int(old), origin)
            truth = self._observed(int(old), int(old) + 144, origin)[:, 1]
            residuals.append(truth - raw)
        correction = np.zeros(24)
        if residuals:
            weight = 2 ** (-np.arange(len(history) - 1, -1, -1) / 14)
            mean_error = np.average(np.asarray(residuals), axis=0, weights=weight)
            a = self._basis
            correction = np.linalg.solve(np.einsum("ni,nj->ij", a, a, optimize=False)
                                         + self._pv_penalty,
                                         np.einsum("ni,n->i", a, mean_error, optimize=False))
            correction *= len(history) / (len(history) + 7)
        hourly = np.asarray(self.data.forecasts[origin], float)
        # Preserve official zero-generation knots. No annual capacity estimate.
        corrected = np.where(hourly > 0, np.maximum(0, hourly + correction), 0)
        value = self._integrate(corrected, origin, origin)
        return value, {
            "pv_method": "current_official_issue_with_historical_lead_knot_bias",
            "pv_issue_origin": origin,
            "pv_calibration_origins": history.astype(int).tolist(),
            "pv_calibration_last_label": int(history[-1] + 143) if len(history) else None,
            "pv_calibration_days": len(history),
            "pv_correction_knot_kw": correction.tolist(),
            "pv_calibration_semantics": "smooth hourly knot bias fitted to interval-integrated historical errors",
            "pv_zero_knots": "official_zero_knots_preserved",
        }

    def _price_design(self, indices, cutoff):
        indices = np.asarray(indices, dtype=int)
        if np.any(indices - 144 < 0) or np.any(indices - 144 >= cutoff):
            raise ValueError("price lag is not observed")
        # All reads are bounded prefix views; advanced indices remain within it.
        past = self._observed(0, cutoff, cutoff)
        week = np.where(indices >= 7 * 144, indices - 7 * 144, indices - 144)
        phase = 2 * np.pi * (indices % 144) / 144
        return np.column_stack((np.ones(len(indices)),
            self.data.fixed_price[indices % 144], past[indices - 144, 2],
            past[week, 2], np.sin(phase), np.cos(phase),
            np.sin(2 * phase), np.cos(2 * phase)))

    @lru_cache(maxsize=1500)
    def _price(self, origin, stop):
        if origin < 144:
            return self.data.fixed_price[origin:stop].copy(), {
                "price_method": "appendix1_day_zero_cold_start", "price_last_label": None}
        target = np.arange(origin, stop)
        xt = self._price_design(target, origin)
        train = np.arange(max(144, origin - 28 * 144), origin)
        # These are forecast-time valid lag features for each training label.
        x = self._price_design(train, origin)
        y = self._observed(int(train[0]), origin, origin)[:, 2]
        weights = 2 ** (-(origin - 1 - train) / (14 * 144))
        prior = np.array([0., 0., .5, .5, 0., 0., 0., 0.])
        penalty = np.diag([1e-6, 5., 5., 5., 5., 5., 5., 5.])
        # Explicit contractions avoid this host's Accelerate matmul FPE flags;
        # nonfinite values still fail get() instead of being silently ignored.
        coefficient = np.linalg.solve(np.einsum("ni,nj,n->ij", x, x, weights,
                    optimize=False) + penalty,
                    np.einsum("ni,n,n->i", x, weights, y, optimize=False)
                    + np.diag(penalty) * prior)
        prediction = np.maximum(1e-4, np.einsum("ni,i->n", xt, coefficient, optimize=False))
        return prediction, {
            "price_method": "past_only_recency_weighted_daily_weekly_harmonic_ridge",
            "price_training_start": int(train[0]), "price_last_label": origin - 1,
            "price_training_rows": len(train), "price_coefficients": coefficient.tolist(),
            "known_future_price": False,
        }

    def get(self, day, slot=0, scenario="2"):
        if (not isinstance(day, (int, np.integer)) or not 0 <= day < 365
                or slot not in ISSUE_SLOTS or scenario not in SCENARIOS):
            raise ValueError("expected day 0..364, slot 0/36/72/108 and scenario 2/3/4-2/4-3")
        if scenario in ("2", "4-2") and slot:
            raise ValueError("Q2 / Q4-2 plans may only be released at midnight")
        day, slot = int(day), int(slot)
        origin, stop = day * 144 + slot, (day + 1) * 144
        midnight = self._midnight(day)
        load, pv = midnight[slot:, 0].copy(), midnight[slot:, 1].copy()
        audit = {"day": day, "slot": slot, "scenario": scenario,
            "origin": origin, "information_cutoff_exclusive": origin,
            "max_observed_index": origin - 1 if origin else None,
            "seed": self.seed, "base_forecast": "exp004_no_season" if day >= 31 else "periodic_cold_start",
            "load_method": "common_midnight_cnn_remaining_trajectory",
            "load_correction_kw": 0., "official_pv_allowed": scenario in ("3", "4-3"),
            "known_future_price": False, "target_start": origin, "target_stop": stop,
            "units": {"load_kw": "kW", "pv_kw": "kW", "price": "yuan/kWh"}}
        if scenario in ("3", "4-3"):
            if slot:
                begin = max(0, slot - 18)
                truth = self._observed(day * 144 + begin, origin, origin)[:, 0]
                error = truth - midnight[begin:slot, 0]
                weight = 2 ** (-np.arange(len(error) - 1, -1, -1) / 6)
                bias = float(np.average(error, weights=weight)) * len(error) / (len(error) + 12)
                load = np.maximum(0, load + bias * np.exp(-np.arange(len(load)) / 36))
                audit.update(load_method="midnight_cnn_plus_observed_prefix_decaying_bias",
                    load_correction_kw=bias, load_correction_last_label=origin - 1)
            full_pv, pv_info = self._pv(origin)
            pv = full_pv[:144 - slot].copy()
            audit.update(pv_info)
        if scenario.startswith("4"):
            price, price_info = self._price(origin, stop)
            price = price.copy()
            audit.update(price_info)
        else:
            price = self.data.fixed_price[slot:].copy()
            audit["price_method"] = "known_fixed_attachment1_tariff"
        if not all(np.isfinite(v).all() and np.min(v) >= 0 for v in (load, pv, price)):
            raise ValueError("nonfinite or negative forecast")
        return {"load_kw": load, "pv_kw": pv, "price": price, "audit": audit}

    def net_error_paths(self, day, scenario="2", limit=28):
        """Joint historical midnight errors; AC kWh, actual minus forecast.

        Returns complete (history_day,144) paths, preserving time dependence.
        January fallback paths are identified and never described as CNN errors.
        Price errors are also supplied for joint Q4 scenario construction.
        """
        if not isinstance(limit, int) or limit < 1 or not 1 <= day < 365:
            raise ValueError("need positive history limit and day 1..364")
        history = np.arange(max(1, int(day) - limit), int(day))
        kw, prices = [], []
        for old in history:
            prediction = self.get(int(old), scenario=scenario)
            actual = self._observed(int(old) * 144, (int(old) + 1) * 144, int(day) * 144)
            kw.append(actual[:, :2] - np.column_stack((prediction["load_kw"], prediction["pv_kw"])))
            prices.append(actual[:, 2] - prediction["price"])
        kw = np.asarray(kw)
        return {"errors_kwh": (kw[:, :, 0] - kw[:, :, 1]) / 6,
            "errors_kw": kw, "price_errors": np.asarray(prices), "origins": history * 144,
            "audit": {"information_cutoff_exclusive": int(day) * 144,
                "max_observed_index": int(day) * 144 - 1,
                "fallback_days": history[history < 31].tolist(),
                "source": "frozen_cnn_or_labelled_january_periodic_then_same_issue_adapter",
                "semantics": "joint_completed_daily_residual_paths_not_slot_marginals"}}


def export(seed=42, out=OUT):
    """Archive all requested releases; scoring reads truth only after release."""
    import time
    began = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    forecasts = Forecasts(seed=seed)
    arrays, audits, rows, monthly_rows, lead_rows = {}, [], [], [], []
    months = pd.date_range("2025-02-01", "2025-12-31").month.to_numpy()
    def score(error, scenario, slot, target, unit):
        return {"scenario": scenario, "issue_hour": slot // 6,
            "target": target, "n": error.size, "unit": unit,
            "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "bias_predicted_minus_actual": float(error.mean()),
            "absolute_error_sum": float(np.abs(error).sum()),
            "squared_error_sum": float(np.square(error).sum())}
    for scenario in SCENARIOS:
        for slot in ISSUE_SLOTS if scenario in ("3", "4-3") else (0,):
            values = []
            for day in range(31, 365):
                result = forecasts.get(day, slot, scenario)
                values.append(np.column_stack((result["load_kw"], result["pv_kw"], result["price"])))
                audits.append(result["audit"])
            value = np.stack(values)
            arrays[f"scenario_{scenario}_slot_{slot}"] = value
            # Post-hoc evaluation is deliberately outside get().
            indices = np.arange(31, 365)[:, None] * 144 + np.arange(slot, 144)
            truth = forecasts.data.actual[indices]
            for channel, target in enumerate(("load", "pv", "price", "net_load")):
                if target == "price" and not scenario.startswith("4"):
                    continue
                error = ((value[:, :, 0] - value[:, :, 1]) - (truth[:, :, 0] - truth[:, :, 1])
                         if target == "net_load" else value[:, :, channel] - truth[:, :, channel])
                unit = "yuan/kWh" if channel == 2 else "kW"
                rows.append(score(error, scenario, slot, target, unit))
                for month in range(2, 13):
                    monthly_rows.append({**score(error[months == month], scenario, slot, target, unit),
                                         "month": month})
                for lead in range((144 - slot) // 6):
                    lead_rows.append({**score(error[:, lead * 6:(lead + 1) * 6], scenario, slot, target, unit),
                                      "lead_hour_end": lead + 1, "target_hour_end": slot // 6 + lead + 1})
    arrays["days"] = np.arange(31, 365)
    np.savez_compressed(out / "forecasts.npz", **arrays)
    pd.DataFrame(rows).to_csv(out / "forecast_metrics.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(out / "forecast_monthly.csv", index=False)
    pd.DataFrame(lead_rows).to_csv(out / "forecast_lead.csv", index=False)
    (out / "forecast_audit.json").write_text(json.dumps(audits, indent=2, ensure_ascii=False) + "\n")
    manifest = {"seed": seed, "period": ["2025-02-01", "2025-12-31"],
        "day_indices": "zero_based", "target_dimension": ["load_kw", "pv_kw", "price"],
        "archive_shapes": {k: list(v.shape) for k, v in arrays.items()},
        "source_sha256": {"experiments/exp008/forecast.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "data/results/exp004/predictions.npz": hashlib.sha256((ROOT / "data/results/exp004/predictions.npz").read_bytes()).hexdigest()},
        "raw_data_sha256": forecasts.data.hashes,
        "archive_sha256": hashlib.sha256((out / "forecasts.npz").read_bytes()).hexdigest(),
        "cnn_retrained": False, "midnight_q2_exact_frozen_exp004": True,
        "calibration_hyperparameters": {"pv_history_days": 28, "pv_half_life_days": 14,
            "pv_second_difference_penalty": 12, "pv_ridge": 2, "pv_shrink_days": 7,
            "load_observed_window_slots": 18, "load_decay_slots": 36,
            "price_history_days": 28, "price_half_life_days": 14, "price_ridge": 5},
        "price_information": "past_actual_prices_only; future realized price is evaluation-only",
        "selection": "adapter_hyperparameters_fixed_before_dispatch_iteration_not_selected_from_annual_forecast_scores",
        "generation_seconds": time.perf_counter() - began}
    (out / "forecast_provenance.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"archive": str(out / "forecasts.npz"), "seconds": manifest["generation_seconds"], "metrics": rows}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    export(args.seed, args.out)

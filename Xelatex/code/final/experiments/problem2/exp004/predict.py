"""Read-only forecast interface for the planning team. No solver is called."""

import json

import numpy as np

from experiments.problem2.exp003.data import STEPS, Data

from .data import OUT, VARIANTS, sha256


class ForecastStore:
    def __init__(self, variant="causal_season", seed=42, allow_oracle=False, directory=OUT):
        if variant not in VARIANTS:
            raise ValueError("Unknown variant")
        if variant == "oracle_season" and not allow_oracle:
            raise ValueError("oracle_season uses future actuals; explicitly opt in for exploration only")
        manifest = json.loads((directory / "prediction_manifest.json").read_text())
        if not manifest["complete"] or sha256(directory / "predictions.npz") != manifest["archive_sha256"]:
            raise RuntimeError("Prediction archive is incomplete or changed")
        with np.load(directory / "predictions.npz") as z:
            self.origins = z["origins"].copy()
            self.values = z[f"{variant}_seed_{int(seed)}"].copy()
        self.variant = variant
        self.seed = int(seed)
        self.lookup = {int(o): i for i, o in enumerate(self.origins)}

    def get(self, origin):
        """Return an independent float64 (144,2) load/PV array, in kW."""
        if not isinstance(origin, (int, np.integer)) or isinstance(origin, bool):
            raise TypeError("origin is an integer index of a 10-minute interval")
        if origin not in self.lookup:
            raise ValueError("origin must be a formal 2025 February–December midnight")
        return self.values[self.lookup[origin]].copy()

    def completed_error_paths(self, origin, limit=28):
        """Return complete issued forecast errors (actual minus forecast, kW).

        Preserve cross-slot and load/PV dependence. Only paths whose last
        label is already observed are exposed. Empty history stays empty;
        this method chooses neither scenarios nor a purchasing policy.
        """
        self.get(origin)
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        ids = np.flatnonzero(self.origins + STEPS <= origin)[-limit:]
        origins = self.origins[ids]
        actual = Data().actual[origins[:, None] + np.arange(STEPS)]
        return {"origins": origins.copy(), "errors_kw": actual - self.values[ids],
                "information_cutoff": int(origin), "variant": self.variant,
                "exploratory": self.variant == "oracle_season"}

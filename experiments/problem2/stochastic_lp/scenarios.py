"""Frozen historical joint sampling and prefix-only sequential binary tree."""

import numpy as np

from .model import Tree


def sample_paths(forecast, historical_forecasts, historical_actual, historical_origins,
                 origin, daylight_mask, seed=42, window=56, maximum=16):
    """Inputs contain only archived midnight forecasts and completed actual days."""
    origins = np.asarray(historical_origins, dtype=int)
    if len(origins) and (origins + 144 > origin).any():
        raise ValueError("uncompleted future donor rejected")
    ids = np.flatnonzero((origins + 144 <= origin) & (origins >= origin - window * 144))
    if len(ids) > maximum:
        ids = np.sort(np.random.default_rng(np.random.SeedSequence([seed, origin])).choice(
            ids, size=maximum, replace=False))
    if len(ids):
        errors = historical_actual[ids] - historical_forecasts[ids]
        raw = forecast[None, :, :] + errors
        scale = np.maximum(np.std(errors, axis=(0, 1)), 1.)
        donor_origins = origins[ids]
    else:
        raw = forecast[None, :, :].copy()
        errors = np.zeros_like(raw)
        scale = np.ones(2)
        donor_origins = np.array([], dtype=int)
    paths = np.maximum(raw, 0)
    paths[:, :, 1] *= np.asarray(daylight_mask)[None, :]
    return {"paths_kw": paths, "errors_kw": errors,
            "donor_origins": donor_origins, "weights": np.full(len(paths), 1 / len(paths)),
            "scale_kw": scale, "fallback_point_only": len(ids) == 0,
            "clipped_entries": int(np.sum(raw < 0)),
            "mask_removed_pv_kw_sum": float(np.maximum(raw[:, :, 1], 0).sum()
                                             - paths[:, :, 1].sum()),
            "before_mean_kw": raw.mean(axis=0), "after_mean_kw": paths.mean(axis=0)}


def _split(ids, prefix, minimum_child):
    """Two-means initialized by prefix geometry; future suffix never accessed."""
    if len(ids) < 2 * minimum_child:
        return [ids]
    x = prefix[ids].reshape(len(ids), -1)
    distances = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    i, j = np.unravel_index(np.argmax(distances), distances.shape)
    if distances[i, j] <= 1e-12:
        return [ids]
    centers = x[[i, j]].copy()
    labels = np.zeros(len(ids), dtype=int)
    for _ in range(30):
        new = np.argmin(np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1), axis=1)
        if min(np.sum(new == 0), np.sum(new == 1)) < minimum_child:
            return [ids]
        unchanged = np.array_equal(labels, new)
        labels = new
        centers = np.stack([x[labels == label].mean(axis=0) for label in (0, 1)])
        if unchanged:
            break
    return [ids[labels == label] for label in (0, 1)]


def information_tree(paths_kw, scale_kw, start=0, current_observation=None,
                     branches=(36, 72, 108), minimum_child=2):
    """D1-A, coarse observations; branch only at fixed clock times.

    At replay the current node is deterministic actual supply. All scenarios
    retain equal weights (the approved no-weight-update baseline). Future
    branching uses only simulated observations after the common actual prefix.
    At a branch, current-interval feedback is included as in D9-A.
    """
    paths = np.asarray(paths_kw, dtype=float)
    count, total, _ = paths.shape
    if not 0 <= start < total:
        raise ValueError("invalid start")
    groups = [(np.arange(count), -1)]
    times, parents, probabilities, supply, memberships = [], [], [], [], []
    for t in range(start, total):
        next_groups = []
        for ids, parent in groups:
            parts = [ids]
            if t in branches and not (t == start and current_observation is not None):
                prefix_start = start + int(current_observation is not None)
                prefix = paths[:, prefix_start:t + 1] / np.asarray(scale_kw)[None, None, :]
                parts = _split(ids, prefix, minimum_child)
            for part in parts:
                node = len(times)
                times.append(t - start)
                parents.append(parent)
                probabilities.append(len(part) / count)
                value = (np.asarray(current_observation) if t == start and current_observation is not None
                         else paths[part, t].mean(axis=0))
                supply.append(value)
                memberships.append(part.tolist())
                next_groups.append((part, node))
        groups = next_groups
    tree = Tree(np.asarray(times), np.asarray(parents), np.asarray(probabilities),
                np.asarray(supply), total - start)
    tree.validate()
    return tree, memberships

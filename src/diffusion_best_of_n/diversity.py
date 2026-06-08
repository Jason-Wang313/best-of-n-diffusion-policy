"""Diversity diagnostics for sampled action trajectory pools."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _as_trajectories(trajectories) -> np.ndarray:
    arr = np.asarray(trajectories, dtype=float)
    if arr.ndim < 2:
        raise ValueError("trajectories must have candidate and feature dimensions")
    if arr.shape[0] == 0:
        raise ValueError("trajectory pool must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("trajectories must be finite")
    return arr


def pairwise_action_trajectory_distance(trajectories, metric: str = "l2") -> np.ndarray:
    """Pairwise distances between flattened action trajectories."""

    arr = _as_trajectories(trajectories)
    flat = arr.reshape(arr.shape[0], -1)
    diff = flat[:, None, :] - flat[None, :, :]
    if metric == "l2":
        return np.sqrt(np.sum(diff * diff, axis=-1))
    if metric == "l1":
        return np.sum(np.abs(diff), axis=-1)
    raise ValueError("metric must be 'l2' or 'l1'")


def mean_pairwise_distance(trajectories) -> float:
    distances = pairwise_action_trajectory_distance(trajectories)
    n = distances.shape[0]
    if n < 2:
        return 0.0
    return float(np.mean(distances[np.triu_indices(n, k=1)]))


def effective_sample_diversity(trajectories, sigma: float | None = None) -> float:
    """Kernel effective sample count, equal to 1 for exact collapse."""

    distances = pairwise_action_trajectory_distance(trajectories)
    n = distances.shape[0]
    if n == 1:
        return 1.0
    nonzero = distances[distances > 0]
    if sigma is None:
        sigma = float(np.median(nonzero)) if nonzero.size else 1.0
    if sigma <= 0.0 or not np.isfinite(sigma):
        raise ValueError("sigma must be positive and finite")
    sim = np.exp(-(distances**2) / (2.0 * sigma**2))
    return float((n * n) / np.sum(sim))


def mode_coverage(mode_ids: Iterable[int], expected_modes: int | Iterable[int]) -> float:
    """Fraction of expected action modes represented in the candidate pool."""

    observed = {int(x) for x in mode_ids}
    if isinstance(expected_modes, int):
        expected = set(range(int(expected_modes)))
    else:
        expected = {int(x) for x in expected_modes}
    if not expected:
        raise ValueError("expected_modes must be non-empty")
    return float(len(observed.intersection(expected)) / len(expected))


def duplicate_collapse_rate(trajectories, tolerance: float = 1e-6) -> float:
    """Fraction of candidates that are duplicates within a distance tolerance."""

    arr = _as_trajectories(trajectories)
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    unique: list[np.ndarray] = []
    for traj in arr:
        if not any(float(np.linalg.norm(traj - item)) <= tolerance for item in unique):
            unique.append(traj)
    return float(1.0 - len(unique) / arr.shape[0])


def marginal_diversity_gain(trajectories, n_values: Iterable[int]) -> dict[int, float]:
    """Incremental mean-pairwise-distance gain for growing prefix pools."""

    arr = _as_trajectories(trajectories)
    out: dict[int, float] = {}
    prev = None
    for n in sorted(int(v) for v in n_values):
        if n < 1 or n > arr.shape[0]:
            raise ValueError("all N values must be between 1 and the pool size")
        current = mean_pairwise_distance(arr[:n])
        out[n] = 0.0 if prev is None else float(current - prev)
        prev = current
    return out


def diversity_summary(trajectories, mode_ids=None, expected_modes: int | Iterable[int] | None = None) -> dict[str, float]:
    """Compact diversity summary for one candidate pool."""

    summary = {
        "mean_pairwise_distance": mean_pairwise_distance(trajectories),
        "effective_sample_diversity": effective_sample_diversity(trajectories),
        "duplicate_collapse_rate": duplicate_collapse_rate(trajectories),
    }
    if mode_ids is not None and expected_modes is not None:
        summary["mode_coverage"] = mode_coverage(mode_ids, expected_modes)
    return summary

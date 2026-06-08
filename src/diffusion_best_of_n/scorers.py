"""Reranker and critic scores for diffusion action trajectory pools."""

from __future__ import annotations

import numpy as np

from diffusion_best_of_n.toy_control import ToyObservation, trajectory_utilities


def trajectory_features(obs: ToyObservation, trajectories: np.ndarray) -> np.ndarray:
    traj = np.asarray(trajectories, dtype=float)
    final_pos = obs.block[None, :] + np.sum(traj, axis=1) * (obs.friction / obs.mass)
    final_dist = np.linalg.norm(final_pos - obs.goal[None, :], axis=1)
    energy = np.mean(np.sum(traj * traj, axis=2), axis=1)
    first_norm = np.linalg.norm(traj[:, 0, :], axis=1)
    positions = obs.block[None, None, :] + np.cumsum(traj * (obs.friction / obs.mass), axis=1)
    obstacle_dist = np.min(np.linalg.norm(positions - obs.obstacle[None, None, :], axis=2), axis=1)
    smoothness = np.mean(np.sum(np.diff(traj, axis=1) ** 2, axis=2), axis=1) if traj.shape[1] > 1 else np.zeros(traj.shape[0])
    return np.column_stack(
        [
            -final_dist,
            -energy,
            first_norm,
            -obstacle_dist,
            -smoothness,
            np.ones(traj.shape[0]),
        ]
    )


def random_scores(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=int(n))


def oracle_scores(obs: ToyObservation, trajectories: np.ndarray) -> np.ndarray:
    return trajectory_utilities(obs, trajectories)


def aligned_scores(obs: ToyObservation, trajectories: np.ndarray, seed: int | None = None, noise: float = 0.015) -> np.ndarray:
    rng = np.random.default_rng(seed)
    utility = oracle_scores(obs, trajectories)
    return utility + rng.normal(scale=float(noise), size=utility.shape)


def diffusion_likelihood_proxy(obs: ToyObservation, trajectories: np.ndarray) -> np.ndarray:
    """A plausible internal score favoring smooth, low-energy denoised actions."""

    features = trajectory_features(obs, trajectories)
    return 0.58 * features[:, 0] + 0.25 * features[:, 1] + 0.17 * features[:, 4]


def misaligned_tail_scores(obs: ToyObservation, trajectories: np.ndarray, seed: int | None = None) -> np.ndarray:
    """A scorer that likes risky high-score tails near hidden obstacles."""

    rng = np.random.default_rng(seed)
    features = trajectory_features(obs, trajectories)
    return (
        0.05 * features[:, 0]
        + 2.35 * features[:, 3]
        + 0.70 * features[:, 2]
        - 0.28 * features[:, 1]
        + rng.normal(scale=0.02, size=features.shape[0])
    )


def behavior_cloning_critic(obs: ToyObservation, trajectories: np.ndarray) -> np.ndarray:
    """Score closeness to the direct expert mode."""

    features = trajectory_features(obs, trajectories)
    return 0.85 * features[:, 0] + 0.15 * features[:, 1]


def fit_linear_value_critic(features: np.ndarray, utilities: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    y = np.asarray(utilities, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("features must be 2D and utilities must match rows")
    xtx = x.T @ x + float(ridge) * np.eye(x.shape[1])
    return np.linalg.solve(xtx, x.T @ y)


def apply_linear_critic(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.asarray(features, dtype=float) @ np.asarray(weights, dtype=float)


def calibrated_critic(obs: ToyObservation, trajectories: np.ndarray, pilot_fraction: float = 0.35) -> np.ndarray:
    """Small pilot-rollout calibrated critic for the same candidate pool."""

    features = trajectory_features(obs, trajectories)
    utilities = oracle_scores(obs, trajectories)
    n_pilot = max(4, int(np.ceil(features.shape[0] * float(pilot_fraction))))
    order = np.argsort(misaligned_tail_scores(obs, trajectories, seed=17), kind="mergesort")
    pilot_idx = np.unique(np.r_[order[: n_pilot // 2], order[-n_pilot:]])
    weights = fit_linear_value_critic(features[pilot_idx], utilities[pilot_idx])
    return apply_linear_critic(features, weights)

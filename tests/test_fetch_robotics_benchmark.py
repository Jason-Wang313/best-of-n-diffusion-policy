from __future__ import annotations

import numpy as np
import pytest

from diffusion_audit.benchmarks.fetch_push import (
    evaluate_fetch_trajectory,
    fetch_goal_progress_score,
    fetch_misaligned_speed_score,
    fetch_obs_to_features,
    make_fetch_expert_dataset,
    record_fetch_policy_trajectory,
    reset_fetch_obs,
)


def test_fetch_features_templates_and_short_rollout():
    try:
        import gymnasium as gym  # noqa: F401
        import gymnasium_robotics  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"FetchPush dependency unavailable: {exc}")

    obs = reset_fetch_obs(0, max_episode_steps=8)
    features = fetch_obs_to_features(obs)
    assert features.shape == (17,)
    assert np.all(np.isfinite(features))

    _, trajectory = record_fetch_policy_trajectory(0, horizon=6, mode=0, noise_seed=11)
    assert trajectory.shape == (6, 4)
    assert trajectory.min() >= -1.0
    assert trajectory.max() <= 1.0

    stacked = np.asarray([trajectory, -trajectory], dtype=np.float32)
    goal_scores = fetch_goal_progress_score(obs, stacked)
    bad_scores = fetch_misaligned_speed_score(stacked, seed=12)
    assert goal_scores.shape == (2,)
    assert bad_scores.shape == (2,)
    assert np.all(np.isfinite(goal_scores))
    assert np.all(np.isfinite(bad_scores))

    rollout = evaluate_fetch_trajectory(0, trajectory)
    assert rollout.steps > 0
    assert np.isfinite(rollout.utility)
    assert np.isfinite(rollout.best_distance)


def test_fetch_expert_dataset_shapes():
    try:
        import gymnasium as gym  # noqa: F401
        import gymnasium_robotics  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"FetchPush dependency unavailable: {exc}")

    obs, actions = make_fetch_expert_dataset(states=1, candidates_per_state=2, horizon=4, seed=3)
    assert obs.shape == (2, 17)
    assert actions.shape == (2, 4, 4)
    assert np.all(np.isfinite(obs))
    assert np.all(np.isfinite(actions))

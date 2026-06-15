"""Gymnasium Robotics FetchPush wrappers for benchmark trajectory search evidence."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


FETCH_PUSH_ENV_ID = "FetchPush-v4"
FETCH_ACTION_LOW = -1.0
FETCH_ACTION_HIGH = 1.0


@dataclass(frozen=True)
class FetchRollout:
    utility: float
    initial_distance: float
    best_distance: float
    final_distance: float
    normalized_progress: float
    final_progress: float
    success: bool
    steps: int
    runtime_seconds: float


def _require_fetch():
    try:
        import gymnasium as gym
        import gymnasium_robotics  # noqa: F401
    except Exception as exc:  # pragma: no cover - dependency-specific branch
        raise RuntimeError(
            "FetchPush benchmark requires gymnasium-robotics and mujoco. "
            "Install requirements.txt before running the Fetch benchmark."
        ) from exc
    return gym


def reset_fetch_obs(seed: int, *, max_episode_steps: int = 50) -> dict:
    """Return one deterministic FetchPush reset observation."""

    gym = _require_fetch()
    env = gym.make(FETCH_PUSH_ENV_ID, render_mode=None, max_episode_steps=int(max_episode_steps))
    try:
        obs, _ = env.reset(seed=int(seed))
    finally:
        env.close()
    return obs


def _fetch_parts(obs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(obs["observation"], dtype=float)
    grip = raw[:3]
    obj = np.asarray(obs["achieved_goal"], dtype=float)
    goal = np.asarray(obs["desired_goal"], dtype=float)
    return grip, obj, goal


def fetch_obs_to_features(obs: dict) -> np.ndarray:
    """Convert a FetchPush dict observation into compact conditioning features."""

    grip, obj, goal = _fetch_parts(obs)
    obj_to_goal = goal - obj
    grip_to_obj = obj - grip
    xy_distance = float(np.linalg.norm(obj_to_goal[:2]))
    xyz_distance = float(np.linalg.norm(obj_to_goal))
    return np.asarray(
        [
            *grip,
            *obj,
            *goal,
            *obj_to_goal,
            *grip_to_obj,
            xy_distance,
            xyz_distance,
        ],
        dtype=np.float32,
    )


def fetch_expert_action(obs: dict, *, mode: int = 0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Closed-loop CPU heuristic used only to create light demonstration trajectories."""

    grip, obj, goal = _fetch_parts(obs)
    obj_to_goal = goal[:2] - obj[:2]
    distance = float(np.linalg.norm(obj_to_goal))
    unit = obj_to_goal / max(distance, 1e-6)
    normal = np.asarray([-unit[1], unit[0]], dtype=float)
    if int(mode) == 1:
        unit = unit + 0.35 * normal
    elif int(mode) == 2:
        unit = unit - 0.35 * normal
    elif int(mode) == 3:
        unit = -unit
    elif int(mode) == 4:
        unit = normal
    unit = unit / max(float(np.linalg.norm(unit)), 1e-6)

    behind = obj.copy()
    behind[:2] = obj[:2] - 0.06 * unit
    behind[2] = obj[2] + 0.015
    push_target = obj.copy()
    push_target[:2] = obj[:2] + 0.08 * unit
    push_target[2] = obj[2] + 0.005
    target = behind if np.linalg.norm(grip[:2] - behind[:2]) > 0.035 else push_target

    action = np.zeros(4, dtype=float)
    action[:3] = np.clip((target - grip) * 25.0, FETCH_ACTION_LOW, FETCH_ACTION_HIGH)
    if rng is not None:
        action[:3] += rng.normal(scale=0.035, size=3)
    return np.clip(action, FETCH_ACTION_LOW, FETCH_ACTION_HIGH).astype(np.float32)


def record_fetch_policy_trajectory(
    seed: int,
    *,
    horizon: int,
    mode: int,
    noise_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Record one closed-loop heuristic action sequence from a reset state."""

    gym = _require_fetch()
    env = gym.make(FETCH_PUSH_ENV_ID, render_mode=None, max_episode_steps=int(horizon))
    rng = np.random.default_rng(noise_seed) if noise_seed is not None else None
    actions: list[np.ndarray] = []
    try:
        obs, _ = env.reset(seed=int(seed))
        initial_features = fetch_obs_to_features(obs)
        for _ in range(int(horizon)):
            action = fetch_expert_action(obs, mode=int(mode), rng=rng)
            actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()

    if not actions:
        actions = [np.zeros(4, dtype=np.float32)]
    while len(actions) < int(horizon):
        actions.append(actions[-1].copy())
    return initial_features, np.asarray(actions[: int(horizon)], dtype=np.float32)


def make_fetch_expert_dataset(
    *,
    states: int,
    candidates_per_state: int,
    horizon: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate small FetchPush demonstration trajectories with controlled mode diversity."""

    rng = np.random.default_rng(seed)
    obs_rows = []
    action_rows = []
    for state_idx in range(int(states)):
        env_seed = int(seed) * 1000 + state_idx
        for cand_idx in range(int(candidates_per_state)):
            mode = int(rng.choice([0, 1, 2], p=[0.50, 0.25, 0.25]))
            features, actions = record_fetch_policy_trajectory(
                env_seed,
                horizon=int(horizon),
                mode=mode,
                noise_seed=int(seed) * 100_000 + state_idx * 100 + cand_idx,
            )
            obs_rows.append(features)
            action_rows.append(actions)
    return np.asarray(obs_rows, dtype=np.float32), np.asarray(action_rows, dtype=np.float32)


def evaluate_fetch_trajectory(seed: int, trajectory: np.ndarray) -> FetchRollout:
    """Execute one open-loop action trajectory in FetchPush."""

    gym = _require_fetch()
    traj = np.clip(np.asarray(trajectory, dtype=np.float32), FETCH_ACTION_LOW, FETCH_ACTION_HIGH)
    env = gym.make(FETCH_PUSH_ENV_ID, render_mode=None, max_episode_steps=int(traj.shape[0]))
    start = time.perf_counter()
    try:
        obs, _ = env.reset(seed=int(seed))
        initial = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
        best = initial
        final = initial
        success = False
        steps = 0
        for action in traj:
            obs, _, terminated, truncated, info = env.step(action)
            final = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
            best = min(best, final)
            success = success or bool(info.get("is_success", False))
            steps += 1
            if terminated or truncated:
                break
    finally:
        env.close()
    smoothness = float(np.mean(np.sum(np.diff(traj, axis=0) ** 2, axis=1))) if traj.shape[0] > 1 else 0.0
    normalized_progress = (initial - best) / max(initial, 1e-6)
    final_progress = (initial - final) / max(initial, 1e-6)
    utility = normalized_progress + 0.35 * float(success) + 0.15 * final_progress - 0.01 * smoothness
    return FetchRollout(
        utility=float(utility),
        initial_distance=float(initial),
        best_distance=float(best),
        final_distance=float(final),
        normalized_progress=float(normalized_progress),
        final_progress=float(final_progress),
        success=bool(success),
        steps=int(steps),
        runtime_seconds=float(time.perf_counter() - start),
    )


def evaluate_fetch_pool(seed: int, trajectories: np.ndarray) -> tuple[np.ndarray, list[FetchRollout]]:
    """Execute all candidate trajectories from the same FetchPush reset seed."""

    gym = _require_fetch()
    trajs = np.clip(np.asarray(trajectories, dtype=np.float32), FETCH_ACTION_LOW, FETCH_ACTION_HIGH)
    horizon = int(trajs.shape[1]) if trajs.ndim == 3 else 1
    env = gym.make(FETCH_PUSH_ENV_ID, render_mode=None, max_episode_steps=horizon)
    rollouts: list[FetchRollout] = []
    try:
        for trajectory in trajs:
            start = time.perf_counter()
            obs, _ = env.reset(seed=int(seed))
            initial = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
            best = initial
            final = initial
            success = False
            steps = 0
            for action in trajectory:
                obs, _, terminated, truncated, info = env.step(action)
                final = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
                best = min(best, final)
                success = success or bool(info.get("is_success", False))
                steps += 1
                if terminated or truncated:
                    break
            smoothness = float(np.mean(np.sum(np.diff(trajectory, axis=0) ** 2, axis=1))) if trajectory.shape[0] > 1 else 0.0
            normalized_progress = (initial - best) / max(initial, 1e-6)
            final_progress = (initial - final) / max(initial, 1e-6)
            utility = normalized_progress + 0.35 * float(success) + 0.15 * final_progress - 0.01 * smoothness
            rollouts.append(
                FetchRollout(
                    utility=float(utility),
                    initial_distance=float(initial),
                    best_distance=float(best),
                    final_distance=float(final),
                    normalized_progress=float(normalized_progress),
                    final_progress=float(final_progress),
                    success=bool(success),
                    steps=int(steps),
                    runtime_seconds=float(time.perf_counter() - start),
                )
            )
    finally:
        env.close()
    return np.asarray([item.utility for item in rollouts], dtype=float), rollouts


def fetch_trajectory_features(obs: dict, trajectories: np.ndarray) -> np.ndarray:
    """Feature map for FetchPush rerankers."""

    _, obj, goal = _fetch_parts(obs)
    traj = np.asarray(trajectories, dtype=float)
    obj_to_goal = goal[:2] - obj[:2]
    unit = obj_to_goal / max(float(np.linalg.norm(obj_to_goal)), 1e-6)
    normal = np.asarray([-unit[1], unit[0]], dtype=float)
    xy = traj[:, :, :2]
    mean_xy = np.mean(xy, axis=1)
    first_xy = np.mean(xy[:, : max(1, xy.shape[1] // 3), :], axis=1)
    last_xy = np.mean(xy[:, -max(1, xy.shape[1] // 3) :, :], axis=1)
    energy = np.mean(np.sum(traj[:, :, :3] ** 2, axis=2), axis=1)
    smoothness = np.mean(np.sum(np.diff(traj[:, :, :3], axis=1) ** 2, axis=2), axis=1) if traj.shape[1] > 1 else np.zeros(traj.shape[0])
    z_abs = np.mean(np.abs(traj[:, :, 2]), axis=1)
    return np.column_stack(
        [
            mean_xy @ unit,
            first_xy @ (-unit),
            last_xy @ unit,
            np.abs(mean_xy @ normal),
            -energy,
            -smoothness,
            -z_abs,
            np.ones(traj.shape[0]),
        ]
    )


def fetch_goal_progress_score(obs: dict, trajectories: np.ndarray) -> np.ndarray:
    """Heuristic score that prefers approach-then-push actions toward the object goal."""

    features = fetch_trajectory_features(obs, trajectories)
    weights = np.asarray([0.95, 0.35, 0.80, -0.25, 0.08, 0.05, 0.04, 0.0], dtype=float)
    return features @ weights


def fetch_misaligned_speed_score(trajectories: np.ndarray, *, seed: int | None = None) -> np.ndarray:
    """Pathological scorer that likes energetic, vertically jittery action sequences."""

    rng = np.random.default_rng(seed)
    traj = np.asarray(trajectories, dtype=float)
    energy = np.mean(np.sum(traj[:, :, :3] ** 2, axis=2), axis=1)
    z_jitter = np.mean(np.abs(traj[:, :, 2]), axis=1)
    roughness = np.mean(np.sum(np.diff(traj[:, :, :3], axis=1) ** 2, axis=2), axis=1) if traj.shape[1] > 1 else 0.0
    return 0.9 * energy + 0.4 * z_jitter + 0.25 * roughness + rng.normal(scale=0.01, size=traj.shape[0])

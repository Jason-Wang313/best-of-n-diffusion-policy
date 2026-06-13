from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diffusion_audit.controller import (
    AUDIT_ROLLOUTS,
    BLOCK_HIGH_N,
    CALIBRATE_SCORER,
    INCREASE_DIVERSITY,
    INCREASE_N,
    REDUCE_K,
    STOP_EARLY,
    AuditThenSampleConfig,
    audit_then_sample,
    audit_then_sample_adaptive,
)
from diffusion_audit.deployment import expected_selection_record
from diffusion_audit.diversity import diversity_summary
from diffusion_audit.evaluation import evaluate_pool
from diffusion_audit.io import ROOT, results_dir, write_json
from diffusion_audit.scorers import (
    aligned_scores,
    diffusion_likelihood_proxy,
    tail_only_misaligned_scores,
    weakly_aligned_scores,
)
from diffusion_audit.stats import bootstrap_mean_ci
from diffusion_audit.toy_control import (
    ToyObservation,
    make_observations,
    sample_diffusion_like_pool,
    trajectory_utilities,
)


REGIME_ORDER = [
    "nominal_aligned",
    "weak_tail_alignment",
    "hidden_obstacle_tail",
    "duplicate_artifact",
    "diversity_collapse",
    "latency_spike",
    "calibration_drift",
    "missing_utility",
    "recovery_aligned",
]

NEGATIVE_REGIMES = {
    "hidden_obstacle_tail",
    "duplicate_artifact",
    "diversity_collapse",
    "latency_spike",
    "calibration_drift",
    "missing_utility",
}


@dataclass(frozen=True)
class StressPool:
    regime: str
    observation: ToyObservation
    trajectories: np.ndarray
    scores: np.ndarray
    utilities: np.ndarray
    controller_utilities: np.ndarray | None
    diversity: dict[str, float]
    lambda_cost: float
    runtime_per_step_ms: float
    runtime_overhead_ms: float
    sampler_k: int
    scorer_family: str


def n_grid(max_candidates: int) -> list[int]:
    values = [4, 8, 16, 32, 64, 96]
    out = [n for n in values if n <= int(max_candidates)]
    if not out:
        return [1]
    if out[-1] != int(max_candidates) and int(max_candidates) <= 128:
        out.append(int(max_candidates))
    return sorted(set(out))


def runtime_measurements(
    n_values: list[int],
    k_values: list[int],
    *,
    per_step_ms: float,
    overhead_ms: float,
) -> dict[tuple[int, int], float]:
    return {
        (int(n), int(k)): float(overhead_ms) + int(n) * int(k) * float(per_step_ms)
        for n in n_values
        for k in k_values
    }


def hidden_visible_observation(obs: ToyObservation) -> ToyObservation:
    return ToyObservation(
        block=obs.block,
        goal=obs.goal,
        obstacle=obs.obstacle,
        friction=obs.friction,
        mass=obs.mass,
        action_noise=obs.action_noise,
        hidden_penalty=False,
    )


def make_stress_pool(
    regime: str,
    *,
    seed: int,
    episode_idx: int,
    candidates: int,
    horizon: int,
) -> StressPool:
    if regime not in REGIME_ORDER:
        raise ValueError(f"unknown deployment stress regime {regime}")
    ood = "hidden_obstacle" if regime == "hidden_obstacle_tail" else "id"
    obs = make_observations(1, seed=90_000 + 997 * seed + 53 * episode_idx + len(regime), ood=ood)[0]
    sampler_cfg = {
        "nominal_aligned": (16, 0.90, 1.00, 1.00, False, False, False),
        "weak_tail_alignment": (16, 1.05, 0.95, 1.00, False, False, False),
        "hidden_obstacle_tail": (16, 1.20, 1.00, 1.00, False, True, False),
        "duplicate_artifact": (16, 0.90, 1.00, 1.00, False, True, False),
        "diversity_collapse": (16, 0.05, 0.01, 0.20, True, False, False),
        "latency_spike": (32, 0.72, 0.80, 0.80, False, False, False),
        "calibration_drift": (16, 1.00, 1.00, 1.00, False, False, False),
        "missing_utility": (16, 0.90, 0.90, 1.00, False, False, False),
        "recovery_aligned": (16, 0.88, 1.00, 1.00, False, False, False),
    }[regime]
    k, temperature, diversity_value, mode_coverage, collapsed, biased_bad_mode, low_k_noise = sampler_cfg
    pool = sample_diffusion_like_pool(
        obs,
        n_candidates=candidates,
        horizon=horizon,
        denoising_steps=k,
        temperature=temperature,
        diversity=diversity_value,
        mode_coverage_value=mode_coverage,
        collapsed=collapsed,
        biased_bad_mode=biased_bad_mode,
        low_k_noise=low_k_noise,
        seed=120_000 + 1009 * seed + 47 * episode_idx + len(regime),
    )
    trajectories = np.asarray(pool.trajectories, dtype=float)
    utilities = trajectory_utilities(obs, trajectories)
    rng = np.random.default_rng(150_000 + 2003 * seed + 67 * episode_idx + len(regime))
    scorer_family = "aligned"
    lambda_cost = 0.0007
    runtime_per_step_ms = 0.020
    runtime_overhead_ms = 0.0
    controller_utilities: np.ndarray | None = utilities

    if regime in {"nominal_aligned", "recovery_aligned"}:
        scores = aligned_scores(obs, trajectories, seed=seed + episode_idx, noise=0.010)
    elif regime == "weak_tail_alignment":
        scores = weakly_aligned_scores(obs, trajectories, seed=seed + episode_idx, noise=0.18)
        scorer_family = "weakly_aligned"
    elif regime == "hidden_obstacle_tail":
        visible_obs = hidden_visible_observation(obs)
        scores = tail_only_misaligned_scores(visible_obs, trajectories, seed=seed + episode_idx)
        scorer_family = "hidden-tail-misaligned"
    elif regime == "duplicate_artifact":
        worst = int(np.argmin(utilities))
        duplicate_count = max(8, candidates // 3)
        trajectories[:duplicate_count] = trajectories[worst]
        utilities = trajectory_utilities(obs, trajectories)
        scores = aligned_scores(obs, trajectories, seed=seed + episode_idx, noise=0.006)
        scores[:duplicate_count] = float(np.max(scores) + 2.5) + rng.normal(scale=0.002, size=duplicate_count)
        scorer_family = "duplicated-high-score-artifact"
    elif regime == "diversity_collapse":
        scores = aligned_scores(obs, trajectories, seed=seed + episode_idx, noise=0.004)
        scorer_family = "collapsed-aligned"
    elif regime == "latency_spike":
        scores = aligned_scores(obs, trajectories, seed=seed + episode_idx, noise=0.010)
        utilities = 0.18 * utilities
        lambda_cost = 0.012
        runtime_per_step_ms = 0.095
        runtime_overhead_ms = 1.5
        scorer_family = "aligned-latency-spike"
    elif regime == "calibration_drift":
        scores = aligned_scores(obs, trajectories, seed=seed + episode_idx, noise=0.006)
        order = np.argsort(scores, kind="mergesort")
        drift = order[-max(8, candidates // 4) :]
        scores[drift] = -utilities[drift] + float(np.max(scores)) + rng.normal(scale=0.015, size=drift.size)
        scorer_family = "calibration-drift-tail"
    elif regime == "missing_utility":
        scores = diffusion_likelihood_proxy(obs, trajectories) + rng.normal(scale=0.01, size=candidates)
        controller_utilities = None
        scorer_family = "utility-missing-likelihood-proxy"
    else:
        raise AssertionError(regime)

    utility_offset = 1.2 if regime == "latency_spike" else 2.0
    utilities = utilities + utility_offset
    if controller_utilities is not None:
        controller_utilities = utilities

    div = diversity_summary(trajectories, pool.mode_ids, expected_modes=5)
    return StressPool(
        regime=regime,
        observation=obs,
        trajectories=trajectories,
        scores=np.asarray(scores, dtype=float),
        utilities=np.asarray(utilities, dtype=float),
        controller_utilities=controller_utilities,
        diversity=div,
        lambda_cost=float(lambda_cost),
        runtime_per_step_ms=float(runtime_per_step_ms),
        runtime_overhead_ms=float(runtime_overhead_ms),
        sampler_k=int(k),
        scorer_family=scorer_family,
    )


def selected_policy_record(
    pool: StressPool,
    *,
    policy: str,
    n: int,
    k: int,
    score_source: str,
) -> dict[str, float | int | str]:
    scores = pool.utilities if score_source == "oracle" else pool.scores
    record = expected_selection_record(
        scores,
        pool.utilities,
        n=int(n),
        k=int(k),
        lambda_cost=pool.lambda_cost,
        runtime_per_candidate_ms=pool.runtime_per_step_ms,
        runtime_overhead_ms=pool.runtime_overhead_ms,
    )
    return {
        "policy": policy,
        "score_source": score_source,
        **record,
    }


def add_policy_effect_rows(policy_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    comparisons = [
        ("adaptive_ats", "fixed_high_N"),
        ("adaptive_ats", "fixed_low_N"),
        ("audit_then_sample", "fixed_high_N"),
        ("fixed_high_N", "fixed_low_N"),
        ("oracle_high_N", "adaptive_ats"),
    ]
    subsets = {
        "all": policy_rows,
        "harmful_high_N": policy_rows[policy_rows["harmful_high_N"].astype(bool)],
        "negative_regimes": policy_rows[policy_rows["negative_control"].astype(bool)],
        "aligned_recovery": policy_rows[policy_rows["regime"].isin(["nominal_aligned", "recovery_aligned"])],
    }
    for subset_name, subset in subsets.items():
        pivot = subset.pivot_table(
            index=["seed", "episode_idx", "regime"],
            columns="policy",
            values="latency_adjusted_utility",
            aggfunc="mean",
        )
        for lhs, rhs in comparisons:
            if lhs not in pivot.columns or rhs not in pivot.columns:
                effects = np.asarray([], dtype=float)
            else:
                effects = (pivot[lhs] - pivot[rhs]).dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "subset": subset_name,
                    "effect": f"{lhs}_minus_{rhs}",
                    **bootstrap_mean_ci(effects, seed=2200 + len(subset_name) + len(lhs) + len(rhs)),
                }
            )
    return pd.DataFrame(rows)


def add_figures(policy_rows: pd.DataFrame, decision_rows: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    paper_fig_dir = ROOT / "paper" / "iclr" / "figures"
    paper_fig_dir.mkdir(parents=True, exist_ok=True)

    summary = policy_rows.groupby("policy", as_index=False).agg(
        latency_adjusted_utility=("latency_adjusted_utility", "mean"),
        runtime_ms=("runtime_ms", "mean"),
    )
    order = ["fixed_low_N", "fixed_high_N", "audit_then_sample", "adaptive_ats", "oracle_high_N"]
    colors = {
        "fixed_low_N": "#7f7f7f",
        "fixed_high_N": "#d95f02",
        "audit_then_sample": "#1b9e77",
        "adaptive_ats": "#0072b2",
        "oracle_high_N": "#6a3d9a",
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for name in order:
        row = summary[summary["policy"] == name]
        if row.empty:
            continue
        ax.scatter(
            float(row["runtime_ms"].iloc[0]),
            float(row["latency_adjusted_utility"].iloc[0]),
            s=80,
            color=colors[name],
            label=name.replace("_", " "),
            edgecolor="white",
            linewidth=0.7,
        )
    ax.set_xlabel("mean deployment runtime cost (ms-equivalent)")
    ax.set_ylabel("mean latency-adjusted selected utility")
    ax.set_title("Deployment stress policy frontier")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    frontier = fig_dir / "deployment_stress_frontier.png"
    fig.savefig(frontier, dpi=180)
    plt.close(fig)

    action_counts = (
        decision_rows.pivot_table(
            index="regime",
            columns="adaptive_action",
            values="episode_idx",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(REGIME_ORDER)
        .fillna(0)
    )
    action_colors = {
        INCREASE_N: "#1b9e77",
        STOP_EARLY: "#d95f02",
        REDUCE_K: "#e6ab02",
        BLOCK_HIGH_N: "#7570b3",
        INCREASE_DIVERSITY: "#66a61e",
        CALIBRATE_SCORER: "#a6761d",
        AUDIT_ROLLOUTS: "#666666",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    bottom = np.zeros(len(action_counts), dtype=float)
    for action in sorted(action_counts.columns):
        values = action_counts[action].to_numpy(dtype=float)
        ax.bar(
            np.arange(len(action_counts)),
            values,
            bottom=bottom,
            color=action_colors.get(action, "#8dd3c7"),
            label=action.replace("_", " "),
        )
        bottom += values
    ax.set_xticks(np.arange(len(action_counts)))
    ax.set_xticklabels([label.replace("_", "\n") for label in action_counts.index], fontsize=8)
    ax.set_ylabel("adaptive-controller decisions")
    ax.set_title("Stress-regime action audit")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    actions = fig_dir / "deployment_stress_actions.png"
    fig.savefig(actions, dpi=180)
    plt.close(fig)

    for path in [frontier, actions]:
        shutil.copy2(path, paper_fig_dir / path.name)


def write_generated_tex(summary: dict, effect_rows: pd.DataFrame, out_dir: Path) -> None:
    def macro(name: str, value: str | int | float) -> str:
        if isinstance(value, float):
            text = f"{value:.3f}"
        else:
            text = str(value)
        return f"\\newcommand{{\\{name}}}{{{text}}}\n"

    def effect_value(effect: str, subset: str, col: str) -> float:
        row = effect_rows[(effect_rows["effect"] == effect) & (effect_rows["subset"] == subset)]
        return float(row.iloc[0][col]) if len(row) else float("nan")

    lines = [
        "% Generated by experiments/deployment_stress.py; do not edit by hand.\n",
        macro("AtsDeployStressRows", summary["decision_rows"]),
        macro("AtsDeployStressRegimes", summary["regime_count"]),
        macro("AtsDeployStressHarmfulRows", summary["harmful_high_n_rows"]),
        macro("AtsDeployStressAdaptiveFalseAdmit", summary["adaptive_false_admit_rate"]),
        macro("AtsDeployStressStaticFalseAdmit", summary["static_false_admit_rate"]),
        macro("AtsDeployStressFixedHighHarmRate", summary["fixed_high_harm_rate"]),
        macro("AtsDeployStressAuditRate", summary["adaptive_audit_or_abstain_rate"]),
        macro("AtsDeployStressMeanCheckedN", summary["adaptive_mean_checked_n"]),
        macro("AtsDeployStressStaticVsHigh", effect_value("audit_then_sample_minus_fixed_high_N", "all", "mean")),
        macro("AtsDeployStressStaticVsHighCILow", effect_value("audit_then_sample_minus_fixed_high_N", "all", "ci_low")),
        macro("AtsDeployStressStaticVsHighHarm", effect_value("audit_then_sample_minus_fixed_high_N", "harmful_high_N", "mean")),
        macro("AtsDeployStressStaticVsHighHarmCILow", effect_value("audit_then_sample_minus_fixed_high_N", "harmful_high_N", "ci_low")),
        macro("AtsDeployStressStaticVsHighNegative", effect_value("audit_then_sample_minus_fixed_high_N", "negative_regimes", "mean")),
        macro("AtsDeployStressStaticVsHighNegativeCILow", effect_value("audit_then_sample_minus_fixed_high_N", "negative_regimes", "ci_low")),
        macro("AtsDeployStressStaticAlignedCost", effect_value("audit_then_sample_minus_fixed_high_N", "aligned_recovery", "mean")),
        macro("AtsDeployStressStaticAlignedCostCIHigh", effect_value("audit_then_sample_minus_fixed_high_N", "aligned_recovery", "ci_high")),
        macro("AtsDeployStressAdaptiveVsHigh", effect_value("adaptive_ats_minus_fixed_high_N", "all", "mean")),
        macro("AtsDeployStressAdaptiveVsHighCILow", effect_value("adaptive_ats_minus_fixed_high_N", "all", "ci_low")),
        macro("AtsDeployStressAdaptiveVsHighHarm", effect_value("adaptive_ats_minus_fixed_high_N", "harmful_high_N", "mean")),
        macro("AtsDeployStressAdaptiveVsHighHarmCILow", effect_value("adaptive_ats_minus_fixed_high_N", "harmful_high_N", "ci_low")),
        macro("AtsDeployStressFixedHighVsLowHarm", effect_value("fixed_high_N_minus_fixed_low_N", "harmful_high_N", "mean")),
        macro("AtsDeployStressFixedHighVsLowHarmCIHigh", effect_value("fixed_high_N_minus_fixed_low_N", "harmful_high_N", "ci_high")),
    ]
    (out_dir / "deployment_stress_generated.tex").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--episodes-per-regime", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 8, 16])
    args = parser.parse_args()

    out_dir = results_dir()
    ns = n_grid(args.candidates)
    low_n = min(ns)
    high_n = max(ns)
    k_values = sorted({int(k) for k in args.k_values if int(k) >= 1})
    max_k = max(k_values)
    cfg = AuditThenSampleConfig(
        bootstrap_trials=120,
        confidence_method="both",
        min_effective_diversity=1.5,
        min_tail_rank_correlation=0.10,
        min_score_utility_correlation=0.10,
        min_audit_units=8,
        min_effect_size=0.005,
        use_effective_n_for_bounds=False,
    )

    decision_rows: list[dict] = []
    policy_rows: list[dict] = []
    curve_rows: list[dict] = []
    episode_idx = 0
    for seed in args.seeds:
        for regime in REGIME_ORDER:
            for within_regime in range(int(args.episodes_per_regime)):
                pool = make_stress_pool(
                    regime,
                    seed=int(seed),
                    episode_idx=episode_idx,
                    candidates=int(args.candidates),
                    horizon=int(args.horizon),
                )
                runtime_map = runtime_measurements(
                    ns,
                    k_values,
                    per_step_ms=pool.runtime_per_step_ms,
                    overhead_ms=pool.runtime_overhead_ms,
                )
                static_result = audit_then_sample(
                    pool.trajectories,
                    pool.scores,
                    pool.controller_utilities,
                    n_values=ns,
                    k_values=k_values,
                    lambda_cost=pool.lambda_cost,
                    runtime_measurements=runtime_map,
                    diversity_diagnostics=pool.diversity,
                    sampler_metadata={"denoising_steps": pool.sampler_k, "regime": regime},
                    config=cfg,
                    seed=300_000 + 41 * seed + episode_idx,
                )
                adaptive_result = audit_then_sample_adaptive(
                    pool.trajectories,
                    pool.scores,
                    pool.controller_utilities,
                    batch_size=16,
                    n_values=ns,
                    k_values=k_values,
                    lambda_cost=pool.lambda_cost,
                    runtime_measurements=runtime_map,
                    diversity_diagnostics=pool.diversity,
                    sampler_metadata={"denoising_steps": pool.sampler_k, "regime": regime},
                    config=cfg,
                    seed=400_000 + 43 * seed + episode_idx,
                )
                eval_payload = evaluate_pool(pool.scores, pool.utilities, ns, mc_trials=80, seed=seed + episode_idx)
                for n in ns:
                    curve_rows.append(
                        {
                            "seed": int(seed),
                            "episode_idx": int(episode_idx),
                            "within_regime": int(within_regime),
                            "regime": regime,
                            "negative_control": regime in NEGATIVE_REGIMES,
                            "N": int(n),
                            "expected_real_utility": float(eval_payload["real_curve"][n]),
                            "expected_score": float(eval_payload["score_curve"][n]),
                            "oracle_real_utility": float(eval_payload["oracle_curve"][n]),
                            "score_utility_correlation": float(eval_payload["score_utility_correlation"]),
                            "tail_rank_correlation": float(eval_payload["tail_rank_correlation"]),
                            "top_score_tail_real_utility": float(eval_payload["top_score_tail_real_utility"]),
                            "high_n_regret": float(eval_payload["high_n_regret"]),
                            **pool.diversity,
                        }
                    )
                fixed_low = selected_policy_record(
                    pool,
                    policy="fixed_low_N",
                    n=low_n,
                    k=max_k,
                    score_source="score",
                )
                fixed_high = selected_policy_record(
                    pool,
                    policy="fixed_high_N",
                    n=high_n,
                    k=max_k,
                    score_source="score",
                )
                harmful_high_n = bool(fixed_high["latency_adjusted_utility"] < fixed_low["latency_adjusted_utility"])
                records = [
                    fixed_low,
                    fixed_high,
                    selected_policy_record(pool, policy="oracle_high_N", n=high_n, k=max_k, score_source="oracle"),
                    selected_policy_record(
                        pool,
                        policy="audit_then_sample",
                        n=static_result.selected_n,
                        k=static_result.selected_k,
                        score_source="score",
                    ),
                    selected_policy_record(
                        pool,
                        policy="adaptive_ats",
                        n=adaptive_result.selected_n,
                        k=adaptive_result.selected_k,
                        score_source="score",
                    ),
                ]
                for record in records:
                    policy_rows.append(
                        {
                            "seed": int(seed),
                            "episode_idx": int(episode_idx),
                            "within_regime": int(within_regime),
                            "regime": regime,
                            "negative_control": regime in NEGATIVE_REGIMES,
                            "harmful_high_N": harmful_high_n,
                            "lambda_cost": pool.lambda_cost,
                            "runtime_per_step_ms": pool.runtime_per_step_ms,
                            "runtime_overhead_ms": pool.runtime_overhead_ms,
                            "scorer_family": pool.scorer_family,
                            **record,
                        }
                    )
                static_diag = static_result.confidence_diagnostics
                adaptive_diag = adaptive_result.confidence_diagnostics
                decision_rows.append(
                    {
                        "seed": int(seed),
                        "episode_idx": int(episode_idx),
                        "within_regime": int(within_regime),
                        "regime": regime,
                        "negative_control": regime in NEGATIVE_REGIMES,
                        "harmful_high_N": harmful_high_n,
                        "scorer_family": pool.scorer_family,
                        "static_action": static_result.action_recommendation,
                        "static_decision": static_result.decision_label,
                        "static_selected_N": static_result.selected_n,
                        "static_selected_K": static_result.selected_k,
                        "static_false_admit": bool(static_result.action_recommendation == INCREASE_N and harmful_high_n),
                        "adaptive_action": adaptive_result.action_recommendation,
                        "adaptive_decision": adaptive_result.decision_label,
                        "adaptive_selected_N": adaptive_result.selected_n,
                        "adaptive_selected_K": adaptive_result.selected_k,
                        "adaptive_checked_N": int(adaptive_diag.get("adaptive_checked_N", adaptive_result.selected_n)),
                        "adaptive_stopping_savings": float(adaptive_diag.get("adaptive_stopping_savings", 0.0) or 0.0),
                        "adaptive_false_admit": bool(adaptive_result.action_recommendation == INCREASE_N and harmful_high_n),
                        "fixed_high_minus_low_latency_adjusted": float(
                            fixed_high["latency_adjusted_utility"] - fixed_low["latency_adjusted_utility"]
                        ),
                        "fixed_high_minus_low_real": float(fixed_high["expected_real_utility"] - fixed_low["expected_real_utility"]),
                        "static_utility_gain_lcb": float(static_diag.get("utility_gain_lcb", np.nan)),
                        "adaptive_utility_gain_lcb": float(adaptive_diag.get("utility_gain_lcb", np.nan)),
                        "adaptive_tail_utility_lcb": float(adaptive_diag.get("tail_utility_lcb", np.nan)),
                        "effective_sample_diversity": float(pool.diversity["effective_sample_diversity"]),
                        "duplicate_collapse_rate": float(pool.diversity["duplicate_collapse_rate"]),
                        "lambda_cost": pool.lambda_cost,
                        "runtime_per_step_ms": pool.runtime_per_step_ms,
                    }
                )
                episode_idx += 1

    decisions = pd.DataFrame(decision_rows)
    policies = pd.DataFrame(policy_rows)
    curves = pd.DataFrame(curve_rows)
    effects = add_policy_effect_rows(policies)
    decision_summary = decisions.groupby(["regime", "adaptive_action"], as_index=False).size()
    policy_summary = policies.groupby(["regime", "policy"], as_index=False).agg(
        expected_real_utility=("expected_real_utility", "mean"),
        latency_adjusted_utility=("latency_adjusted_utility", "mean"),
        runtime_ms=("runtime_ms", "mean"),
        N=("N", "mean"),
        K=("K", "mean"),
    )

    table_dir = out_dir / "tables"
    decisions.to_csv(table_dir / "deployment_stress_decisions.csv", index=False)
    policies.to_csv(table_dir / "deployment_stress_policy_rows.csv", index=False)
    curves.to_csv(table_dir / "deployment_stress_curves.csv", index=False)
    effects.to_csv(table_dir / "deployment_stress_policy_effect_cis.csv", index=False)
    decision_summary.to_csv(table_dir / "deployment_stress_action_summary.csv", index=False)
    policy_summary.to_csv(table_dir / "deployment_stress_policy_summary.csv", index=False)

    harmful = decisions[decisions["harmful_high_N"].astype(bool)]
    adaptive_audit_or_abstain = decisions[
        decisions["adaptive_action"].isin(
            [BLOCK_HIGH_N, INCREASE_DIVERSITY, AUDIT_ROLLOUTS, CALIBRATE_SCORER, STOP_EARLY, REDUCE_K]
        )
    ]
    summary = {
        "artifact_tables": {
            "decisions": "results/tables/deployment_stress_decisions.csv",
            "policy_rows": "results/tables/deployment_stress_policy_rows.csv",
            "curves": "results/tables/deployment_stress_curves.csv",
            "policy_effect_cis": "results/tables/deployment_stress_policy_effect_cis.csv",
            "action_summary": "results/tables/deployment_stress_action_summary.csv",
            "policy_summary": "results/tables/deployment_stress_policy_summary.csv",
        },
        "artifact_figures": {
            "policy_frontier": "results/figures/deployment_stress_frontier.png",
            "action_audit": "results/figures/deployment_stress_actions.png",
        },
        "decision_rows": int(len(decisions)),
        "policy_rows": int(len(policies)),
        "regime_count": int(len(REGIME_ORDER)),
        "seeds": [int(x) for x in args.seeds],
        "episodes_per_regime": int(args.episodes_per_regime),
        "n_values": ns,
        "k_values": k_values,
        "harmful_high_n_rows": int(len(harmful)),
        "fixed_high_harm_rate": float(np.mean(decisions["harmful_high_N"].astype(float))) if len(decisions) else 0.0,
        "static_false_admit_rate": float(np.mean(harmful["static_false_admit"].astype(float))) if len(harmful) else 0.0,
        "adaptive_false_admit_rate": float(np.mean(harmful["adaptive_false_admit"].astype(float))) if len(harmful) else 0.0,
        "adaptive_audit_or_abstain_rate": float(len(adaptive_audit_or_abstain) / max(len(decisions), 1)),
        "adaptive_mean_checked_n": float(np.mean(decisions["adaptive_checked_N"])) if len(decisions) else 0.0,
        "strongest_effect": effects.to_dict(orient="records"),
        "negative_regimes": sorted(NEGATIVE_REGIMES),
    }
    write_json(out_dir / "deployment_stress_summary.json", summary)
    add_figures(policies, decisions, out_dir)
    write_generated_tex(summary, effects, out_dir)


if __name__ == "__main__":
    main()

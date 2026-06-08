from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RESULTS = Path(os.environ.get("DIFFUSION_BON_RESULTS_DIR", ROOT / "results")).expanduser()
if not RESULTS.is_absolute():
    RESULTS = ROOT / RESULTS


def load_json(name: str) -> dict[str, Any]:
    path = RESULTS / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def csv_columns(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return set(csv.DictReader(handle).fieldnames or [])


def csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def status(strong: bool, weak: bool = False) -> str:
    if strong:
        return "SUPPORTED"
    if weak:
        return "PARTIAL"
    return "UNSUPPORTED"


def is_smoke_results() -> bool:
    return any(part.lower() == "smoke" for part in RESULTS.parts)


def first_row(rows: list[dict[str, str]], **matches: Any) -> dict[str, str]:
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in matches.items()):
            return row
    return {}


def ci_ok(
    row: dict[str, Any],
    *,
    mean_min: float | None = None,
    mean_max: float | None = None,
    low_min: float | None = None,
    high_max: float | None = None,
    min_n: int = 2,
) -> bool:
    if not row:
        return False
    n = int(f(row, "n", 0.0))
    if n < int(min_n):
        return False
    mean = f(row, "mean")
    lo = f(row, "ci_low")
    hi = f(row, "ci_high")
    if mean_min is not None and mean < float(mean_min):
        return False
    if mean_max is not None and mean > float(mean_max):
        return False
    if low_min is not None and lo < float(low_min):
        return False
    if high_max is not None and hi > float(high_max):
        return False
    return True


def add(claims: list[dict[str, Any]], category: str, claim: str, stat: str, evidence: Any) -> None:
    claims.append(
        {
            "id": len(claims) + 1,
            "category": category,
            "claim": claim,
            "status": stat,
            "evidence": evidence if isinstance(evidence, str) else json.dumps(evidence, sort_keys=True),
        }
    )


def forbidden_hits() -> list[dict[str, Any]]:
    surfaces = [
        ROOT / "README.md",
        ROOT / "docs" / "claims.md",
        ROOT / "docs" / "theory.md",
        ROOT / "paper" / "intro.md",
        ROOT / "paper" / "limitations.md",
    ]
    patterns = [
        re.compile(r"\bvalidated on real robots\b", re.I),
        re.compile(r"\breal[- ]robot validation\b", re.I),
        re.compile(r"\buniversal diffusion policy improvement\b", re.I),
        re.compile(r"\bbest-of-n always helps\b", re.I),
        re.compile(r"\bsolves robot manipulation\b", re.I),
    ]
    guards = ["do not claim", "not claim", "forbidden", "unsupported", "limitation", "future work"]
    hits: list[dict[str, Any]] = []
    for path in surfaces:
        text = read_text(path)
        guarded_section = False
        for lineno, line in enumerate(text.splitlines(), start=1):
            lower = line.lower()
            if line.lstrip().startswith("#"):
                guarded_section = any(g in lower for g in guards)
            if guarded_section or any(g in lower for g in guards):
                continue
            for pat in patterns:
                if pat.search(line):
                    hits.append({"path": str(path.relative_to(ROOT)), "line": lineno, "text": line.strip()})
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    controlled = load_json("controlled_sampler_summary.json")
    scorer = load_json("scorer_comparison_summary.json")
    nk = load_json("nk_budget_summary.json")
    learned = load_json("learned_policy_lite_summary.json")
    controlled_agg = csv_rows(RESULTS / "tables" / "controlled_sampler_aggregate.csv")
    controlled_div = csv_rows(RESULTS / "tables" / "controlled_sampler_diversity.csv")
    controlled_effect_cis = csv_rows(RESULTS / "tables" / "controlled_sampler_effect_cis.csv")
    scorer_agg = csv_rows(RESULTS / "tables" / "scorer_comparison_aggregate.csv")
    scorer_effect_cis = csv_rows(RESULTS / "tables" / "scorer_comparison_effect_cis.csv")
    calibration_map = csv_rows(RESULTS / "tables" / "calibration_repair_map.csv")
    nk_latency_effect = csv_rows(RESULTS / "tables" / "nk_budget_latency_effect_ci.csv")
    learned_agg = csv_rows(RESULTS / "tables" / "learned_policy_lite_aggregate.csv")
    learned_seed_agg = csv_rows(RESULTS / "tables" / "learned_policy_lite_seed_aggregate.csv")
    learned_effect_cis = csv_rows(RESULTS / "tables" / "learned_policy_lite_effect_cis.csv")
    learned_training = csv_rows(RESULTS / "tables" / "learned_policy_lite_training.csv")
    learned_receding = csv_rows(RESULTS / "tables" / "learned_policy_lite_receding_horizon.csv")
    diff_doc = read_text(ROOT / "docs" / "differentiation_from_wam_jepa.md").lower()
    checklist_doc = read_text(ROOT / "docs" / "diffusion_policy_validity_checklist.md").lower()
    theory_doc = read_text(ROOT / "docs" / "theory.md").lower()

    claims: list[dict[str, Any]] = []
    strong_metrics: dict[str, Any] = {}

    def agg_row(rows: list[dict[str, str]], **matches: Any) -> dict[str, str]:
        for row in rows:
            ok = True
            for key, value in matches.items():
                if str(row.get(key)) != str(value):
                    ok = False
                    break
            if ok:
                return row
        return {}

    theorem_ok = (
        (ROOT / "src" / "diffusion_best_of_n" / "theory.py").exists()
        and (ROOT / "tests" / "test_theory.py").exists()
        and "tie-aware" in theory_doc
        and "finite" in theory_doc
        and "test_anti_aligned_score_degrades_selected_real_utility" in read_text(ROOT / "tests" / "test_theory.py")
        and "test_tie_aware_best_of_n_uses_tie_group_mean_utility" in read_text(ROOT / "tests" / "test_theory.py")
    )
    add(
        claims,
        "finite_law",
        "Finite tie-aware Best-of-N law is implemented and tested.",
        status(theorem_ok),
        "src/diffusion_best_of_n/theory.py; tests/test_theory.py; docs/theory.md",
    )

    aligned_high = agg_row(
        controlled_agg,
        regime="high_diversity_aligned",
        scorer="aligned",
        N=max(controlled.get("n_values") or [64]),
    )
    aligned_weak = (
        controlled.get("aligned_score_gain_high_minus_low", 0.0) > 0.01
        and controlled.get("aligned_real_gain_high_minus_low", 0.0) > 0.01
    )
    aligned_real_ci = first_row(
        controlled_effect_cis,
        regime="high_diversity_aligned",
        metric="exact_selected_real",
    )
    aligned_score_ci = first_row(
        controlled_effect_cis,
        regime="high_diversity_aligned",
        metric="exact_selected_score",
    )
    aligned_strong = (
        controlled.get("aligned_score_gain_high_minus_low", 0.0) >= 0.50
        and controlled.get("aligned_real_gain_high_minus_low", 0.0) >= 0.50
        and f(aligned_high, "score_utility_correlation") >= 0.99
        and f(aligned_high, "score_utility_correlation_ci_low") >= 0.985
        and f(aligned_high, "tail_rank_correlation") >= 0.95
        and f(aligned_high, "tail_rank_correlation_ci_low") >= 0.90
        and f(aligned_high, "high_n_regret") <= 0.01
        and ci_ok(aligned_real_ci, mean_min=0.50, low_min=0.35, min_n=3)
        and ci_ok(aligned_score_ci, mean_min=0.50, low_min=0.35, min_n=3)
    )
    strong_metrics["aligned_selection"] = {
        "score_gain": controlled.get("aligned_score_gain_high_minus_low"),
        "real_gain": controlled.get("aligned_real_gain_high_minus_low"),
        "real_gain_ci": aligned_real_ci,
        "score_gain_ci": aligned_score_ci,
        "correlation": f(aligned_high, "score_utility_correlation"),
        "correlation_ci_low": f(aligned_high, "score_utility_correlation_ci_low"),
        "tail_rank_correlation": f(aligned_high, "tail_rank_correlation"),
        "tail_rank_correlation_ci_low": f(aligned_high, "tail_rank_correlation_ci_low"),
        "high_n_regret": f(aligned_high, "high_n_regret"),
        "thresholds": {
            "score_gain_min": 0.50,
            "real_gain_min": 0.50,
            "ci_low_gain_min": 0.35,
            "correlation_min": 0.99,
            "correlation_ci_low_min": 0.985,
            "tail_rank_min": 0.95,
            "tail_rank_ci_low_min": 0.90,
            "regret_max": 0.01,
        },
    }
    add(
        claims,
        "controlled_sampler",
        "High N can help aligned diffusion trajectory selection.",
        status(aligned_strong, aligned_weak),
        strong_metrics["aligned_selection"],
    )

    misaligned_weak = (
        controlled.get("misaligned_score_gain_high_minus_low", 0.0) > 0.01
        and controlled.get("misaligned_real_change_high_minus_low", 1.0) <= 0.02
        and controlled.get("misaligned_high_n_regret", 0.0) > 0.05
    ) or (
        scorer.get("misaligned_score_gain_high_minus_low", 0.0) > 0.01
        and scorer.get("misaligned_real_change_high_minus_low", 1.0) <= 0.02
    )
    controlled_misaligned_real_ci = first_row(
        controlled_effect_cis,
        regime="high_diversity_misaligned",
        metric="exact_selected_real",
    )
    controlled_misaligned_score_ci = first_row(
        controlled_effect_cis,
        regime="high_diversity_misaligned",
        metric="exact_selected_score",
    )
    scorer_misaligned_real_ci = first_row(
        scorer_effect_cis,
        regime="hidden_obstacle_high_diversity",
        effect="high_minus_low",
        metric="exact_selected_real",
    )
    scorer_misaligned_score_ci = first_row(
        scorer_effect_cis,
        regime="hidden_obstacle_high_diversity",
        effect="high_minus_low",
        metric="exact_selected_score",
    )
    scorer_oracle_gap_ci = first_row(
        scorer_effect_cis,
        regime="hidden_obstacle_high_diversity",
        effect="oracle_real_utility_selector_minus_misaligned_tail_scorer",
    )
    misaligned_strong = (
        controlled.get("misaligned_score_gain_high_minus_low", 0.0) >= 0.35
        and controlled.get("misaligned_real_change_high_minus_low", 1.0) <= -0.25
        and controlled.get("misaligned_high_n_regret", 0.0) >= 0.80
        and scorer.get("misaligned_score_gain_high_minus_low", 0.0) >= 0.35
        and scorer.get("misaligned_real_change_high_minus_low", 1.0) <= -0.25
        and scorer.get("oracle_minus_misaligned_high_n", 0.0) >= 0.90
        and ci_ok(controlled_misaligned_score_ci, mean_min=0.35, low_min=0.20, min_n=3)
        and ci_ok(controlled_misaligned_real_ci, mean_max=-0.25, high_max=-0.12, min_n=3)
        and ci_ok(scorer_misaligned_score_ci, mean_min=0.35, low_min=0.20, min_n=3)
        and ci_ok(scorer_misaligned_real_ci, mean_max=-0.25, high_max=-0.12, min_n=3)
        and ci_ok(scorer_oracle_gap_ci, mean_min=0.90, low_min=0.65, min_n=3)
    )
    strong_metrics["misaligned_tail_selection"] = {
        "controlled_score_gain": controlled.get("misaligned_score_gain_high_minus_low"),
        "controlled_real_change": controlled.get("misaligned_real_change_high_minus_low"),
        "controlled_high_n_regret": controlled.get("misaligned_high_n_regret"),
        "controlled_score_gain_ci": controlled_misaligned_score_ci,
        "controlled_real_change_ci": controlled_misaligned_real_ci,
        "scorer_score_gain": scorer.get("misaligned_score_gain_high_minus_low"),
        "scorer_real_change": scorer.get("misaligned_real_change_high_minus_low"),
        "oracle_minus_misaligned_high_n": scorer.get("oracle_minus_misaligned_high_n"),
        "scorer_score_gain_ci": scorer_misaligned_score_ci,
        "scorer_real_change_ci": scorer_misaligned_real_ci,
        "oracle_minus_misaligned_high_n_ci": scorer_oracle_gap_ci,
        "thresholds": {
            "controlled_score_gain_min": 0.35,
            "controlled_real_change_max": -0.25,
            "controlled_regret_min": 0.80,
            "scorer_score_gain_min": 0.35,
            "scorer_real_change_max": -0.25,
            "oracle_gap_min": 0.90,
            "score_ci_low_min": 0.20,
            "real_change_ci_high_max": -0.12,
            "oracle_gap_ci_low_min": 0.65,
        },
    }
    add(
        claims,
        "controlled_sampler",
        "High N can hurt or saturate under scorer misalignment.",
        status(misaligned_strong, misaligned_weak),
        strong_metrics["misaligned_tail_selection"],
    )

    low_div_rows = [row for row in controlled_div if row.get("regime") == "low_diversity_high_quality"]
    low_div_eff = [f(row, "effective_sample_diversity") for row in low_div_rows]
    low_div_collapse = [f(row, "duplicate_collapse_rate") for row in low_div_rows]
    low_div_weak = bool(controlled.get("low_diversity_has_small_marginal_value"))
    low_div_gain_ci = first_row(
        controlled_effect_cis,
        regime="low_diversity_high_quality",
        metric="exact_selected_real",
    )
    high_div_gain_ci = first_row(
        controlled_effect_cis,
        regime="high_diversity_aligned",
        metric="exact_selected_real",
    )
    low_div_strong = (
        abs(controlled.get("low_diversity_real_gain_high_minus_low", 1.0)) <= 0.02
        and controlled.get("high_diversity_real_gain_high_minus_low", 0.0) >= 0.50
        and bool(low_div_eff)
        and max(low_div_eff) <= 1.05
        and bool(low_div_collapse)
        and min(low_div_collapse) >= 0.98
        and ci_ok(low_div_gain_ci, min_n=3)
        and f(low_div_gain_ci, "ci_low") >= -0.02
        and f(low_div_gain_ci, "ci_high") <= 0.02
        and ci_ok(high_div_gain_ci, mean_min=0.50, low_min=0.35, min_n=3)
    )
    strong_metrics["low_diversity"] = {
        "low_diversity_gain": controlled.get("low_diversity_real_gain_high_minus_low"),
        "high_diversity_gain": controlled.get("high_diversity_real_gain_high_minus_low"),
        "low_diversity_gain_ci": low_div_gain_ci,
        "high_diversity_gain_ci": high_div_gain_ci,
        "max_effective_sample_diversity": max(low_div_eff) if low_div_eff else None,
        "min_duplicate_collapse_rate": min(low_div_collapse) if low_div_collapse else None,
        "thresholds": {
            "abs_low_gain_max": 0.02,
            "low_gain_ci_bounds": [-0.02, 0.02],
            "high_diversity_gain_min": 0.50,
            "high_diversity_gain_ci_low_min": 0.35,
            "effective_diversity_max": 1.05,
            "duplicate_collapse_rate_min": 0.98,
        },
    }
    add(
        claims,
        "diversity",
        "Low diversity reduces marginal value of N.",
        status(low_div_strong, low_div_weak),
        strong_metrics["low_diversity"],
    )

    nk_phase_path = RESULTS / "tables" / "nk_budget_phase.csv"
    nk_rows = csv_rows(nk_phase_path)
    unique_n = {row.get("N") for row in nk_rows}
    unique_k = {row.get("K") for row in nk_rows}
    nk_weak = bool(nk) and {"N", "K", "B", "real_utility", "latency_adjusted_utility"}.issubset(csv_columns(nk_phase_path))
    nk_strong = nk_weak and len(unique_n) >= 6 and len(unique_k) >= 5 and csv_row_count(nk_phase_path) >= 30
    strong_metrics["nk_tradeoff"] = {
        "rows": csv_row_count(nk_phase_path),
        "unique_N": len(unique_n),
        "unique_K": len(unique_k),
        "thresholds": {"rows_min": 30, "unique_N_min": 6, "unique_K_min": 5},
    }
    add(
        claims,
        "denoising_budget",
        "N versus K tradeoff is measured.",
        status(nk_strong, nk_weak),
        strong_metrics["nk_tradeoff"],
    )

    latency_weak = bool(nk.get("latency_prefers_smaller_budget_than_max")) and bool(
        nk.get("latency_best_differs_from_real_best")
    )
    best_latency = nk.get("best_latency_adjusted") or {}
    high_budget = nk.get("high_budget_corner") or {}
    latency_margin = f(high_budget, "latency_adjusted_utility") - f(best_latency, "latency_adjusted_utility")
    budget_ratio = f(best_latency, "B") / max(f(high_budget, "B"), 1e-12)
    latency_effect_ci = first_row(
        nk_latency_effect,
        effect="best_latency_adjusted_minus_high_budget_corner",
    )
    latency_strong = (
        latency_weak
        and budget_ratio <= 0.10
        and latency_margin <= -2.0
        and ci_ok(latency_effect_ci, mean_min=2.0, low_min=1.50, min_n=3)
    )
    strong_metrics["latency_adjustment"] = {
        "best_latency_adjusted": best_latency,
        "high_budget_corner": high_budget,
        "best_budget_to_high_budget_ratio": budget_ratio,
        "high_minus_best_latency_adjusted_utility": latency_margin,
        "best_minus_high_latency_adjusted_ci": latency_effect_ci,
        "thresholds": {
            "best_budget_ratio_max": 0.10,
            "high_minus_best_latency_adjusted_utility_max": -2.0,
            "best_minus_high_ci_low_min": 1.50,
        },
    }
    add(
        claims,
        "latency",
        "Latency-adjusted utility can prefer smaller N or K.",
        status(latency_strong, latency_weak),
        strong_metrics["latency_adjustment"],
    )

    repair_weak = scorer.get("calibrated_minus_misaligned_high_n", 0.0) > 0.05
    repair_fraction = scorer.get("calibrated_minus_misaligned_high_n", 0.0) / max(
        scorer.get("oracle_minus_misaligned_high_n", 0.0), 1e-12
    )
    repair_ci = first_row(
        scorer_effect_cis,
        regime="hidden_obstacle_high_diversity",
        effect="calibrated_critic_minus_misaligned_tail_scorer",
    )
    oracle_gap_ci = first_row(
        scorer_effect_cis,
        regime="hidden_obstacle_high_diversity",
        effect="oracle_real_utility_selector_minus_misaligned_tail_scorer",
    )
    calibration_successes = [row for row in calibration_map if row.get("repair_status") == "strong_repair"]
    calibration_failures = [row for row in calibration_map if row.get("repair_status") == "no_strong_repair"]
    repair_strong = (
        scorer.get("calibrated_minus_misaligned_high_n", 0.0) >= 0.80
        and scorer.get("oracle_minus_misaligned_high_n", 0.0) >= 0.90
        and repair_fraction >= 0.75
        and ci_ok(repair_ci, mean_min=0.80, low_min=0.60, min_n=3)
        and ci_ok(oracle_gap_ci, mean_min=0.90, low_min=0.65, min_n=3)
        and len(calibration_successes) >= 1
        and len(calibration_failures) >= 1
    )
    strong_metrics["calibration_repair"] = {
        "calibrated_minus_misaligned_high_n": scorer.get("calibrated_minus_misaligned_high_n"),
        "oracle_minus_misaligned_high_n": scorer.get("oracle_minus_misaligned_high_n"),
        "repair_fraction_of_oracle_gap": repair_fraction,
        "calibrated_minus_misaligned_ci": repair_ci,
        "oracle_minus_misaligned_ci": oracle_gap_ci,
        "calibration_map": calibration_map,
        "thresholds": {
            "calibrated_gain_min": 0.80,
            "oracle_gap_min": 0.90,
            "repair_fraction_min": 0.75,
            "calibrated_gain_ci_low_min": 0.60,
            "oracle_gap_ci_low_min": 0.65,
            "success_rows_min": 1,
            "non_repair_rows_min": 1,
        },
    }
    add(
        claims,
        "calibration",
        "Calibrated scorer repairs high-N selection in at least one regime.",
        status(repair_strong, repair_weak),
        strong_metrics["calibration_repair"],
    )

    not_wam_ok = (
        "what theorem is reused" in diff_doc
        and "diffusion tail over-selection" in diff_doc
        and "denoising" in diff_doc
        and "sample diversity" in diff_doc
        and "what wam claims are forbidden" in diff_doc
    )
    add(
        claims,
        "differentiation",
        "Project is not a WAM clone.",
        status(not_wam_ok),
        "docs/differentiation_from_wam_jepa.md centers diffusion action trajectories rather than imagined rollouts.",
    )

    not_jepa_ok = (
        "what jepa claims are forbidden" in diff_doc
        and "action sequence" in checklist_doc
        and "learned diffusion policy-lite" in checklist_doc
        and "latent-real rank distortion" in diff_doc
    )
    add(
        claims,
        "differentiation",
        "Project is not a JEPA clone.",
        status(not_jepa_ok),
        "docs/differentiation_from_wam_jepa.md forbids latent-prediction framing as the central contribution.",
    )

    required_artifacts = [
        RESULTS / "controlled_sampler_summary.json",
        RESULTS / "scorer_comparison_summary.json",
        RESULTS / "nk_budget_summary.json",
        RESULTS / "learned_policy_lite_summary.json",
        RESULTS / "tables" / "controlled_sampler_aggregate.csv",
        RESULTS / "tables" / "controlled_sampler_seed_aggregate.csv",
        RESULTS / "tables" / "controlled_sampler_effect_cis.csv",
        RESULTS / "tables" / "scorer_comparison_aggregate.csv",
        RESULTS / "tables" / "scorer_comparison_seed_aggregate.csv",
        RESULTS / "tables" / "scorer_comparison_effect_cis.csv",
        RESULTS / "tables" / "calibration_repair_map.csv",
        RESULTS / "tables" / "nk_budget_phase.csv",
        RESULTS / "tables" / "nk_budget_seed_aggregate.csv",
        RESULTS / "tables" / "nk_budget_latency_effect_ci.csv",
        RESULTS / "tables" / "learned_policy_lite_aggregate.csv",
        RESULTS / "tables" / "learned_policy_lite_seed_aggregate.csv",
        RESULTS / "tables" / "learned_policy_lite_effect_cis.csv",
        RESULTS / "figures" / "nk_budget_phase_diagram.png",
    ]
    table_min_rows = {
        "controlled_sampler_aggregate.csv": 42,
        "scorer_comparison_aggregate.csv": 49,
        "nk_budget_phase.csv": 30,
        "learned_policy_lite_aggregate.csv": 200,
        "learned_policy_lite_effect_cis.csv": 80,
    }
    table_rows_ok = all(
        csv_row_count(RESULTS / "tables" / name) >= minimum
        for name, minimum in table_min_rows.items()
    )
    figure_sizes_ok = all(
        (RESULTS / "figures" / name).exists() and (RESULTS / "figures" / name).stat().st_size >= 10_000
        for name in [
            "controlled_sampler_curves.png",
            "scorer_comparison.png",
            "nk_budget_phase_diagram.png",
            "learned_policy_lite_ood.png",
            "toy_image_observations.png",
        ]
    )
    artifacts_weak = all(path.exists() and path.stat().st_size > 0 for path in required_artifacts)
    artifacts_strong = artifacts_weak and table_rows_ok and figure_sizes_ok
    strong_metrics["artifact_backing"] = {
        "required_artifacts_exist": artifacts_weak,
        "table_min_rows": table_min_rows,
        "actual_table_rows": {
            name: csv_row_count(RESULTS / "tables" / name) for name in table_min_rows
        },
        "figure_sizes_ok": figure_sizes_ok,
    }
    add(
        claims,
        "artifact_backing",
        "All major claims are backed by CSV/JSON artifacts.",
        status(artifacts_strong, artifacts_weak),
        strong_metrics["artifact_backing"],
    )

    learned_checklist = learned.get("diffusion_policy_validity_checklist") or {}
    min_learned_seeds = 1 if is_smoke_results() else 3
    learned_conditionings = {row.get("conditioning") for row in learned_agg}
    learned_training_seeds = {row.get("seed") for row in learned_training}
    learned_weak = all(
        learned_checklist.get(key) is True
        for key in [
            "stochastic_trajectory_generation",
            "iterative_denoising_or_noise_to_action_generation",
            "conditioning_on_observation_or_state",
            "action_sequence_generation",
            "receding_horizon_or_trajectory_execution_evaluation",
        ]
    ) and len(learned.get("ood_regimes") or []) >= 4 and {"state", "image"}.issubset(learned_conditionings)
    k4_calibrated_gains: dict[str, list[float]] = {"state": [], "image": []}
    k4_calibrated_ci_lows: dict[str, list[float]] = {"state": [], "image": []}
    k4_calibrated_corr: dict[str, list[float]] = {"state": [], "image": []}
    k4_calibrated_tail: dict[str, list[float]] = {"state": [], "image": []}
    k4_effect_rows = []
    for conditioning in ["state", "image"]:
        for regime in sorted({row.get("regime") for row in learned_agg}):
            ci_row = first_row(
                learned_effect_cis,
                conditioning=conditioning,
                regime=regime,
                scorer="calibrated_critic",
                K=4,
                metric="exact_selected_real",
            )
            if ci_row:
                k4_effect_rows.append(ci_row)
                k4_calibrated_ci_lows[conditioning].append(f(ci_row, "ci_low"))
                k4_calibrated_gains[conditioning].append(f(ci_row, "mean"))
            rows = [
                row
                for row in learned_agg
                if row.get("conditioning") == conditioning
                and row.get("regime") == regime
                and row.get("scorer") == "calibrated_critic"
                and row.get("K") == "4"
            ]
            if rows:
                hi = max(rows, key=lambda row: f(row, "N"))
                k4_calibrated_corr[conditioning].append(f(hi, "score_utility_correlation"))
                k4_calibrated_tail[conditioning].append(f(hi, "tail_rank_correlation"))
    loss_ratios = [f(row, "loss_ratio") for row in learned_training]
    state_loss_ratios = [f(row, "loss_ratio") for row in learned_training if row.get("conditioning") == "state"]
    image_loss_ratios = [f(row, "loss_ratio") for row in learned_training if row.get("conditioning") == "image"]
    learned_strong = (
        learned_weak
        and bool(loss_ratios)
        and bool(state_loss_ratios)
        and bool(image_loss_ratios)
        and max(state_loss_ratios) <= 0.30
        and max(image_loss_ratios) <= 0.65
        and bool(learned.get("loss_decreased_all_seed_conditioning_pairs"))
        and len(learned_agg) >= 200
        and len(learned_seed_agg) >= 200
        and len(learned_receding) >= 20
        and len(learned_training_seeds) >= min_learned_seeds
        and int(learned.get("image_size") or 0) == 32
        and len(k4_effect_rows) >= 10
        and all(ci_ok(row, mean_min=0.08, low_min=0.04, min_n=3) for row in k4_effect_rows)
        and min(k4_calibrated_corr["state"] or [0.0]) >= 0.95
        and min(k4_calibrated_tail["state"] or [0.0]) >= 0.85
        and min(k4_calibrated_corr["image"] or [0.0]) >= 0.90
        and min(k4_calibrated_tail["image"] or [0.0]) >= 0.50
    )
    strong_metrics["learned_policy_lite"] = {
        "checklist": learned_checklist,
        "conditioning_modes": sorted(learned_conditionings),
        "image_size": learned.get("image_size"),
        "ood_regimes": learned.get("ood_regimes"),
        "num_training_seeds": len(learned_training_seeds),
        "min_training_seeds_required": min_learned_seeds,
        "max_state_loss_ratio": max(state_loss_ratios) if state_loss_ratios else None,
        "max_image_loss_ratio": max(image_loss_ratios) if image_loss_ratios else None,
        "aggregate_rows": len(learned_agg),
        "seed_aggregate_rows": len(learned_seed_agg),
        "receding_rows": len(learned_receding),
        "k4_calibrated_real_gains": k4_calibrated_gains,
        "k4_calibrated_real_gain_ci_lows": k4_calibrated_ci_lows,
        "k4_calibrated_correlations": k4_calibrated_corr,
        "k4_calibrated_tail_correlations": k4_calibrated_tail,
        "k4_effect_ci_rows": k4_effect_rows,
        "thresholds": {
            "state_loss_ratio_max": 0.30,
            "image_loss_ratio_max": 0.65,
            "aggregate_rows_min": 200,
            "seed_aggregate_rows_min": 200,
            "receding_rows_min": 20,
            "image_size": 32,
            "effect_ci_mean_min": 0.08,
            "effect_ci_low_min": 0.04,
            "state_correlation_min": 0.95,
            "state_tail_correlation_min": 0.85,
            "image_correlation_min": 0.90,
            "image_tail_correlation_min": 0.50,
        },
    }
    add(
        claims,
        "learned_diffusion_policy_lite",
        "Learned experiment satisfies the Diffusion Policy-style validity checklist and includes multi-seed state/image ID/OOD evaluation.",
        status(learned_strong, learned_weak),
        strong_metrics["learned_policy_lite"],
    )

    overclaims = forbidden_hits()
    payload = {
        "claims": claims,
        "claims_by_category": {},
        "overclaims": overclaims,
        "num_supported": sum(c["status"] == "SUPPORTED" for c in claims),
        "num_partial": sum(c["status"] == "PARTIAL" for c in claims),
        "num_unsupported": sum(c["status"] == "UNSUPPORTED" for c in claims),
        "num_strong": sum(c["status"] == "SUPPORTED" for c in claims),
        "all_strong": all(c["status"] == "SUPPORTED" for c in claims) and not overclaims,
        "strong_metrics": strong_metrics,
    }
    for claim in claims:
        payload["claims_by_category"].setdefault(claim["category"], []).append(claim)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "claims_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (RESULTS / "ideal_metrics_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = ["# Claims Status", ""]
    for claim in claims:
        lines.append(
            f"- Claim {claim['id']}: **{claim['status']}** - {claim['claim']} Evidence: {claim['evidence']}"
        )
    if overclaims:
        lines.extend(["", "## Overclaims"])
        for hit in overclaims:
            lines.append(f"- {hit['path']}:{hit['line']} {hit['text']}")
    text = "\n".join(lines) + "\n"
    (RESULTS / "claims_status.md").write_text(text, encoding="utf-8")
    (RESULTS / "ideal_metrics_status.md").write_text(text, encoding="utf-8")
    print(text)

    if args.fail_on_error and (
        any(c["status"] != "SUPPORTED" for c in claims) or overclaims
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

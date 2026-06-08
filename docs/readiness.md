# Audit Readiness

This repository is ready to make only claims that pass `scripts/run_claim_audit.sh`. The strongest current claim is the finite tie-aware Best-of-N law plus CI-backed synthetic evidence that high `N` helps under aligned upper-tail scoring and hurts under misalignment.

The weakest remaining claim is external validity: the learned Diffusion Policy-lite result includes multi-seed state conditioning and 32x32 image conditioning, but it is still a CPU toy experiment. It does not establish real-robot performance, production-scale visual manipulation quality, or universal high-`N` improvement.

## Artifact Inventory

Core summaries:

- `results/controlled_sampler_summary.json`
- `results/scorer_comparison_summary.json`
- `results/nk_budget_summary.json`
- `results/learned_policy_lite_summary.json`
- `results/ideal_metrics_status.json`

Primary tables:

- `results/tables/controlled_sampler_curves.csv`
- `results/tables/controlled_sampler_seed_aggregate.csv`
- `results/tables/controlled_sampler_effect_cis.csv`
- `results/tables/scorer_comparison_curves.csv`
- `results/tables/scorer_comparison_seed_aggregate.csv`
- `results/tables/scorer_comparison_effect_cis.csv`
- `results/tables/calibration_repair_map.csv`
- `results/tables/nk_budget_phase.csv`
- `results/tables/nk_budget_latency_effect_ci.csv`
- `results/tables/learned_policy_lite_curves.csv`
- `results/tables/learned_policy_lite_seed_aggregate.csv`
- `results/tables/learned_policy_lite_effect_cis.csv`
- `results/tables/learned_policy_lite_receding_horizon.csv`

Primary figures:

- `results/figures/controlled_sampler_curves.png`
- `results/figures/scorer_comparison.png`
- `results/figures/nk_budget_phase_diagram.png`
- `results/figures/learned_policy_lite_ood.png`
- `results/figures/toy_image_observations.png`

## Scope

All acceptance runs are CPU-only by default. The image-conditioned model uses 32x32 toy renderings and a tiny CNN encoder. Heavy simulators, robot hardware, GPU-scale vision training, and real-world deployment validation are out of scope.

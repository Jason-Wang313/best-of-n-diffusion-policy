# Claims

Claims are promotable only when supported by `results/claims_status.json` and corresponding CSV/JSON artifacts.

## Promotable Claim Categories

1. The finite tie-aware Best-of-N law is implemented and tested.
2. High `N` can help aligned diffusion trajectory selection.
3. High `N` can hurt or saturate under scorer misalignment.
4. Low sample diversity reduces the marginal value of increasing `N`.
5. The `N` versus `K` denoising-budget tradeoff is measured.
6. Latency-adjusted utility can prefer smaller `N` or smaller `K`.
7. A calibrated scorer repairs high-`N` selection in at least one synthetic regime.
8. The project is not a WAM clone.
9. The project is not a JEPA clone.
10. Major promoted claims are backed by CSV/JSON artifacts.
11. The learned Diffusion Policy-lite experiment includes state-conditioned and 32x32 image-conditioned variants with seed-level summaries, visual OOD regimes, confidence intervals, and receding-horizon evaluation.

## Non-Claims

This repository does not claim real-robot validation, universal Diffusion Policy improvement, production-scale visual policy quality, or a production deployment rule. The calibration map is evidence for "repairs at least one regime," not evidence that calibration always repairs high-`N` selection. The deployment gate is a conservative diagnostic for synthetic and toy learned experiments.

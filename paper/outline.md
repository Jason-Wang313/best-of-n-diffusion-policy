# Paper Outline

## Title

How Many Diffusion Trajectories Should a Robot Sample? Best-of-N Laws for Diffusion Policy Reranking

## Thesis

Best-of-N inference is valuable for diffusion action policies only when sampled trajectories are sufficiently diverse, the reranker is aligned with real utility in the upper score tail, and denoising latency does not dominate the utility gain.

## Sections

1. Introduction: inference-time trajectory selection and diffusion tail over-selection.
2. Formal setup: finite tie-aware Best-of-N law for `(S, U)` action-trajectory pools.
3. Diagnostics: diversity, alignment, denoising/latency, deployment gate.
4. Experiments: controlled sampler, learned state/image Diffusion Policy-lite, scorer comparison with calibration success/failure map, `N` versus `K`.
5. Audit readiness: CI-backed claim gates, artifact inventory, CPU-only scope.
6. Limitations: synthetic evidence, toy learned model, no broad real-robot claim.

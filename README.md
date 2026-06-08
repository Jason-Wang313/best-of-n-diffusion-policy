# How Many Diffusion Trajectories Should a Robot Sample?

Best-of-N laws for Diffusion Policy-style action trajectory reranking.

This repository studies inference-time selection for stochastic diffusion action generators. Given an observation `o`, a sampler proposes `N` action trajectories, a reranker score `S(o, tau)` selects the top-scoring candidate, and the measured quantity is real task utility minus optional denoising latency cost.

The project is paper-first: the theorem is finite and tie-aware, the experiments are synthetic and CPU-light, and the claim audit only promotes claims backed by CSV/JSON artifacts with confidence-interval evidence. It does not claim real-robot validation or universal Diffusion Policy improvement.

## Scientific Object

The core failure mode is diffusion tail over-selection:

- low diversity means larger `N` has little marginal value;
- high diversity plus aligned scoring can improve selected real utility;
- high diversity plus misaligned scoring can select bad high-score outliers;
- expensive denoising can make a smaller `N` or `K` preferable after latency cost.

The project is scientifically distinct from WAM and JEPA Best-of-N projects. It centers stochastic action-trajectory diffusion, sample diversity, denoising steps, reranker alignment, and latency-adjusted deployment rules.

## Commands

```bash
bash scripts/run_smoke.sh
bash scripts/run_all.sh
bash scripts/run_claim_audit.sh
pytest
```

Artifacts are written under `results/` by default, or `results/smoke/` for the smoke script.

The claim audit also writes `results/ideal_metrics_status.json`; in this repo, `SUPPORTED` means the corresponding strong effect-size and CI gate in `docs/ideal_metrics.md` passed. The smoke run keeps the learned model to one seed, while the full run uses three learned state/image seeds.

See `docs/readiness.md` for the final audit inventory, strongest supported claim, weakest remaining claim, and CPU-only scope.

## Experiment Families

- Family A: controlled diffusion-like action sampler for 2D reaching/pushing.
- Family B: learned Diffusion Policy-lite with a state-conditioned MLP denoiser, a 32x32 image-conditioned tiny-CNN denoiser, action horizons, visual ID/OOD evaluation, seed-level summaries, CI tables, and receding-horizon execution.
- Family C: scorer/reranker comparison across random, diffusion proxy, behavior-cloning critic, pilot value critic, calibrated critic, misaligned tail scorer, oracle, and a calibration success/failure map.
- Family D: phase diagram over trajectory count `N` and denoising steps `K` at fixed budget pressure.

## Claim Boundary

Promoted claims must pass `scripts/run_claim_audit.sh`. Unsupported or partial claims remain research notes, not paper claims. In particular:

- do not claim real-robot validation;
- do not claim that Best-of-N always helps;
- do not claim that calibration always repairs every bad scorer;
- do not claim that a hand-designed sampler is a full Diffusion Policy model.

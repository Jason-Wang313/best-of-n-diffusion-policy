# Final V4 Audit

## Main Thesis

Diffusion action policies can sample many candidate trajectories at test time,
but maximum-score selection is only conditionally useful. Search helps when
candidate diversity, score-utility alignment, and latency-adjusted utility gates
pass; it can hurt, saturate, or waste compute when the score tail is misaligned,
the candidate pool is collapsed, or denoising latency dominates.

## Genuine Novelty

The paper does not claim that diffusion policies, trajectory reranking, or
calibration are new. The contribution is a finite, tie-aware audit law plus an
Audit-Then-Sample controller that decides when to increase `N`, audit rollouts,
repair the scorer, increase diversity, reduce `K`, stop early, or block high-`N`
selection. The v4 paper is specific to stochastic action-trajectory diffusion,
candidate diversity, score-tail alignment, and denoising latency.

## Duplicate-Risk Audit

The v4 manuscript is not a generic Best-of-N wrapper. It has a paper-specific
title, "Audit-Then-Sample: Certifying When Diffusion Policies Should Search More
Trajectories," and its evidence spine is diffusion-policy-specific: true
epsilon-prediction DDPM/DDIM action diffusion, PushT simulator rollouts,
FetchPush-v4 Gymnasium Robotics/MuJoCo rollouts, denoising-step latency, `N`
versus `K` budget tradeoffs, score-tail alignment, diversity collapse, and
controller claim gates.

## Evidence Stack

The final paper combines:

- controlled diffusion-like action samplers
- Audit-Then-Sample controller audit with false-admit negative controls
- scorer/reranker comparison and calibration repair map
- `N` versus `K` latency-adjusted budget sweep
- learned state and 32x32 image Diffusion Policy-lite diagnostics
- true epsilon-prediction action DDPM/DDIM with one-step and clean-target
  ablations
- PushT simulator benchmark over actual sampled action trajectories
- FetchPush-v4 Gymnasium Robotics/MuJoCo benchmark bridge
- 180-row sequential deployment stress suite
- claim audit requiring 20 supported claims, no full-run low-power warning, and
  no real-robot/full-visual-policy overclaim

## Strongest Empirical Results

The claim audit reports all 20 promoted claims as supported. It includes high-`N`
aligned gains, harmful misaligned-score cases, low-diversity saturation, runtime
evidence, zero false admits in harmful negative controls, true-DDPM survival,
PushT rollout metrics, FetchPush rollout metrics, and a reviewer-skepticism
check with no underpowered critical CI units.

The main-text headline is deliberately conditional rather than universal:
Audit-Then-Sample admits high `N` only under lower-bound gates and otherwise
abstains, audits, repairs, reduces compute, increases diversity, or blocks
high-`N` selection.

## Attack Summary

- **Synthetic-only rejection:** repaired by adding PushT and FetchPush-v4
  simulator rollout tiers.
- **Toy diffusion rejection:** repaired by true epsilon-prediction action
  DDPM/DDIM and one-step/clean-target ablations.
- **Universal high-N overclaim:** blocked by negative controls and scoped claims.
- **Controller safety overclaim:** blocked by explicit no hardware-safety or
  production-deployment claim gates.
- **Weak statistics:** full-run critical CI units meet the configured 12-unit
  gate across true-DDPM, PushT, and FetchPush tiers.
- **Latency ignored:** repaired by `N` versus `K` measured runtime and
  latency-adjusted utility.

## Biggest Remaining Scope Boundaries

- All acceptance evidence is CPU simulation evidence.
- PushT and FetchPush use low-dimensional simulator observations and heuristic
  demonstrations, not full-scale visual robot training.
- The image-conditioned diagnostic is 32x32 toy rendering with a tiny CNN.
- The paper does not establish real-robot performance, hardware safety
  certification, production deployment, or universal high-`N` improvement.

These limits are explicit in the abstract, limitations, claim audit, README, and
appendix.

## Paper-Readiness Judgment

Submission-ready as a bounded ICLR-style mechanism paper. The PDF is anonymous,
compiled in the ICLR 2026 template, 11 pages including appendix and references,
with a 6-page main-text label before references. It contains main-text figures
for controller, true diffusion, PushT, and FetchPush results; appendix proof
details; claim ledger; reviewer concern matrix; reproduction commands; and LLM
usage disclosure.

## Verification On 2026-06-16

- `python -m compileall src experiments scripts tests -q`: passed.
- `python -m pytest -q`: passed with 41 tests and only third-party deprecation
  warnings from `pygame/pkg_resources`.
- `bash scripts/run_claim_audit.sh`: passed; all 20 promoted claims are
  supported.
- `python scripts\build_iclr_paper.py --desktop-copy "C:\Users\wangz\OneDrive\Desktop\best of n diffusion policy-v4.pdf"`: passed.
- Final `paper/iclr/main.log` scan found no unresolved citation warnings,
  unresolved reference warnings, rerun warnings, or overfull boxes.
- Standard repo PDF, nested ICLR final PDF, and visible Desktop PDF have
  matching SHA256:
  `B8BFFABA567BE71AFDCAEFDB3339C72707AB24531970599DF6C6BDE41D3D44D0`.
- `pdfinfo` reports 11 letter-size pages.
- Visual QA rendered the final PDF with Poppler and inspected pages 1, 5, 7, 10,
  and 11 for title layout, result figures, references/proof transition, claim
  matrix, reproduction commands, and final path note.

## Exact PDF Paths

Standard local final artifact:
`paper\final\best of n diffusion policy-v4.pdf`

Nested ICLR final artifact:
`paper\iclr\final\best of n diffusion policy-v4.pdf`

Visible Desktop final artifact:
`C:\Users\wangz\OneDrive\Desktop\best of n diffusion policy-v4.pdf`

# Theory

## Finite Tie-Aware Best-of-N Law

For one observation `o`, a diffusion policy samples action trajectories `tau_i ~ pi_theta(tau | o)`. Each trajectory has a scalar reranker score `S(o, tau_i)` and real utility `U(o, tau_i)`.

Best-of-N selects `tau* = argmax_i S(o, tau_i)`. On a finite candidate pool sampled with replacement, the exact expected selected utility is determined by the finite joint distribution of `(S, U)`.

If score tie group `g` occupies sorted ranks `r_min(g)` through `r_max(g)` among `m` finite candidates, then its contribution at sample count `N` is:

`mean_U(g) * [(r_max(g) / m)^N - ((r_min(g) - 1) / m)^N]`.

The implementation in `src/diffusion_best_of_n/theory.py` supports real-valued utilities, binary success/failure utilities, exact selected-score curves, and Monte Carlo simulation with random tie handling.

## Diffusion-Specific Reading

The theorem alone does not say a diffusion policy improves with larger `N`. It says the value of larger `N` is a property of the joint distribution of sampled trajectory score and real utility.

For diffusion action generation, the relevant mechanisms are:

- sample diversity and mode coverage;
- denoising steps `K` and residual low-`K` noise;
- scorer/reranker alignment in the upper score tail;
- latency-adjusted utility `E[U(o, tau*) - lambda C(N, K)]`.

The central failure mode is diffusion tail over-selection: a misaligned scorer can make the selected score improve with `N` while selected real utility saturates or decreases.

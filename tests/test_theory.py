from __future__ import annotations

import numpy as np
import pytest

from diffusion_best_of_n.theory import (
    binary_best_of_n_finite,
    selected_score_best_of_n_finite,
    simulate_best_of_n,
    utility_best_of_n_finite,
)


def test_tie_aware_best_of_n_uses_tie_group_mean_utility():
    scores = np.array([0.0, 1.0, 1.0])
    utilities = np.array([0.0, 1.0, 3.0])
    curve = utility_best_of_n_finite(scores, utilities, [1, 2])
    assert curve[1] == pytest.approx(np.mean(utilities))
    assert curve[2] == pytest.approx(2.0 * (1.0 - (1.0 / 3.0) ** 2))


def test_binary_and_real_valued_utility_consistency():
    scores = [0.0, 0.5, 0.5, 1.0]
    success = [0, 1, 0, 1]
    assert binary_best_of_n_finite(scores, success, [1, 2, 8]) == utility_best_of_n_finite(
        scores, success, [1, 2, 8]
    )


def test_constant_utility_invariance():
    curve = utility_best_of_n_finite([0, 1, 2, 3], [7.0, 7.0, 7.0, 7.0], [1, 4, 16])
    assert all(value == pytest.approx(7.0) for value in curve.values())


def test_oracle_score_monotonicity():
    utilities = np.array([-2.0, -0.5, 0.0, 1.5, 3.0])
    curve = utility_best_of_n_finite(utilities, utilities, [1, 2, 4, 8])
    vals = [curve[n] for n in [1, 2, 4, 8]]
    assert vals == sorted(vals)
    assert vals[-1] > vals[0]


def test_anti_aligned_score_degrades_selected_real_utility():
    utilities = np.array([3.0, 2.0, 1.0, 0.0, -1.0])
    scores = -utilities
    real_curve = utility_best_of_n_finite(scores, utilities, [1, 64])
    score_curve = selected_score_best_of_n_finite(scores, [1, 64])
    assert score_curve[64] > score_curve[1]
    assert real_curve[64] < real_curve[1]


def test_monte_carlo_reproducibility_with_fixed_seed():
    scores = np.array([0.0, 1.0, 1.0, 2.0])
    utilities = np.array([0.0, 1.0, 3.0, 5.0])
    a = simulate_best_of_n(scores, utilities, n=3, trials=200, seed=123)
    b = simulate_best_of_n(scores, utilities, n=3, trials=200, seed=123)
    np.testing.assert_array_equal(a, b)


def test_input_validation():
    with pytest.raises(ValueError):
        utility_best_of_n_finite([], [], [1])
    with pytest.raises(ValueError):
        utility_best_of_n_finite([1.0], [1.0, 2.0], [1])
    with pytest.raises(ValueError):
        utility_best_of_n_finite([1.0], [1.0], [0])
    with pytest.raises(ValueError):
        utility_best_of_n_finite([float("nan")], [1.0], [1])
    with pytest.raises(ValueError):
        binary_best_of_n_finite([0.0, 1.0], [0, 0.5], [1])

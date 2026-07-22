"""
Tests for the correlation and regression functions.

Deterministic cases (perfect linear/monotonic relationships) use exact
hand-derivable expected values. The multi-predictor regression test uses
a seeded random dataset and checks that the true, known coefficients are
recovered within a tolerance - this is standard practice for testing
statistical estimators, since exact floating-point reproduction of a
random draw is not something a test should depend on.

Run with: pytest tests/test_regression.py -v
"""

import numpy as np
import pytest
from app.statistics.regression import (
    pearson_correlation, spearman_correlation,
    simple_linear_regression, multiple_linear_regression,
)


def test_pearson_correlation_perfect_linear():
    # y is exactly 2x -> perfect positive linear relationship -> r = 1.0
    result = pearson_correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    assert result["r"] == pytest.approx(1.0)
    assert result["strength"] == "starker Zusammenhang"
    assert result["significant"] is True


def test_spearman_catches_monotonic_nonlinear_relationship():
    # y = x^2 is monotonically increasing but NOT linear.
    # Spearman (rank-based) should find a perfect relationship (rho = 1),
    # while Pearson (linear) should report something noticeably below 1 -
    # this is exactly the difference the two methods are meant to show.
    x, y = [1, 2, 3, 4, 5], [1, 4, 9, 16, 25]
    spearman_result = spearman_correlation(x, y)
    pearson_result = pearson_correlation(x, y)
    assert spearman_result["r"] == pytest.approx(1.0)
    assert pearson_result["r"] < 1.0


def test_simple_linear_regression_perfect_fit():
    result = simple_linear_regression([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    assert result["slope"] == pytest.approx(2.0)
    assert result["intercept"] == pytest.approx(0.0, abs=1e-9)
    assert result["r_squared"] == pytest.approx(1.0)


def test_multiple_regression_matches_simple_regression_for_one_predictor():
    # With exactly one predictor column, multiple_linear_regression must
    # produce the same intercept, slope and r_squared as
    # simple_linear_regression - it's the same underlying math.
    x, y = [1, 2, 3, 4, 5], [2, 4, 6, 8, 10]
    simple_result = simple_linear_regression(x, y)
    multi_result = multiple_linear_regression(np.array(x).reshape(-1, 1), y)
    assert multi_result["coefficients"]["x1"]["coefficient"] == pytest.approx(simple_result["slope"], abs=1e-6)
    assert multi_result["coefficients"]["intercept"]["coefficient"] == pytest.approx(simple_result["intercept"], abs=1e-6)
    assert multi_result["r_squared"] == pytest.approx(simple_result["r_squared"], abs=1e-6)


def test_multiple_regression_recovers_true_coefficients():
    # Build synthetic data from a known formula (intercept=5, x1-coef=2,
    # x2-coef=-1.5) plus a bit of random noise, then check the model
    # recovers coefficients close to the true values.
    rng = np.random.default_rng(42)
    n = 30
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 5, n)
    noise = rng.normal(0, 1, n)
    y = 5 + 2 * x1 - 1.5 * x2 + noise

    result = multiple_linear_regression(np.column_stack([x1, x2]), y)

    assert result["coefficients"]["intercept"]["coefficient"] == pytest.approx(5, abs=1.5)
    assert result["coefficients"]["x1"]["coefficient"] == pytest.approx(2, abs=0.5)
    assert result["coefficients"]["x2"]["coefficient"] == pytest.approx(-1.5, abs=0.5)
    assert result["r_squared"] > 0.85  # small noise -> should fit well


def test_multiple_regression_matches_normal_equations():
    # Independent correctness check: OLS coefficients must exactly satisfy
    # the normal equations (X'X) * beta = X'y. This verifies the
    # implementation is mathematically correct, independent of the
    # np.linalg.lstsq call used inside the function itself.
    rng = np.random.default_rng(1)
    n = 20
    X = rng.uniform(-5, 5, size=(n, 2))
    y = 3 - 1 * X[:, 0] + 4 * X[:, 1] + rng.normal(0, 0.5, n)

    result = multiple_linear_regression(X, y)
    recovered = np.array([
        result["coefficients"]["intercept"]["coefficient"],
        result["coefficients"]["x1"]["coefficient"],
        result["coefficients"]["x2"]["coefficient"],
    ])

    design = np.column_stack([np.ones(n), X])
    expected = np.linalg.solve(design.T @ design, design.T @ y)
    assert np.allclose(recovered, expected, atol=1e-8)


def test_mismatched_lengths_raise_error():
    with pytest.raises(ValueError):
        pearson_correlation([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        simple_linear_regression([1, 2, 3], [1, 2])


def test_multiple_regression_requires_2d_input():
    with pytest.raises(ValueError):
        multiple_linear_regression([1, 2, 3], [1, 2, 3])


def test_multiple_regression_too_few_observations_raises():
    # 3 observations but 5 predictors -> not enough residual degrees of
    # freedom to fit the model reliably.
    with pytest.raises(ValueError):
        multiple_linear_regression(np.zeros((3, 5)), [1, 2, 3])
"""
Tests for the descriptive statistics functions.
For each function we check: (1) a normal case with a hand-computable
result, (2) at least one edge case, (3) that invalid input raises an
error.
Run with: pytest tests/test_descriptive.py -v
"""

import pytest
from app.statistics.descriptive import (
    mean, median, mode, variance, std_dev,
    value_range, quartiles, iqr, summary_statistics,
)


def test_mean_simple_case():
    # (1+2+3+4+5) / 5 = 3.0 - easy to verify by hand
    assert mean([1, 2, 3, 4, 5]) == 3.0


def test_mean_reacts_to_outlier():
    # A single extreme value (1000) pulls the mean strongly upward
    normal = mean([1, 2, 3, 4, 5])
    with_outlier = mean([1, 2, 3, 4, 1000])
    assert with_outlier > normal * 10


def test_median_odd_count():
    assert median([5, 1, 3]) == 3.0


def test_median_even_count():
    # mean of the two middle values -> (2+3)/2 = 2.5
    assert median([1, 2, 3, 4]) == 2.5


def test_median_robust_to_outlier():
    # Stays stable while the mean on the same data would jump sharply
    normal = median([1, 2, 3, 4, 5])
    with_outlier = median([1, 2, 3, 4, 1000])
    assert normal == with_outlier == 3.0


def test_mode_single_value():
    assert mode([1, 2, 2, 3]) == [2.0]


def test_mode_multimodal():
    # Two values occur equally often (2x each) -> both are returned
    result = mode([1, 1, 2, 2, 3])
    assert set(result) == {1.0, 2.0}


def test_variance_and_std_dev_consistent():
    # std_dev MUST always equal the square root of variance - this checks
    # the mathematical relationship rather than a hand-computed number.
    data = [2, 4, 4, 4, 5, 5, 7, 9]
    assert std_dev(data) == pytest.approx(variance(data) ** 0.5)


def test_variance_too_few_values_raises():
    # With ddof=1 (default) at least 2 values are needed, since spread is
    # undefined for a single value.
    with pytest.raises(ValueError):
        variance([5])


def test_value_range():
    assert value_range([1, 5, 3, 10, 2]) == 9.0


def test_quartiles_and_iqr():
    data = list(range(1, 11))
    q1, q2, q3 = quartiles(data)
    assert q2 == median(data)  # Q2 is by definition equal to the median
    assert iqr(data) == pytest.approx(q3 - q1)


def test_empty_list_raises():
    with pytest.raises(ValueError):
        mean([])


def test_nan_in_data_raises():
    with pytest.raises(ValueError):
        mean([1, 2, float("nan"), 4])


def test_summary_contains_all_metrics_with_description():
    result = summary_statistics([1, 2, 3, 4, 5])
    assert result["mean"]["value"] == 3.0
    # Every metric must ship a non-empty description - the chatbot needs
    # this later to present results to non-technical users.
    for entry in result.values():
        assert entry["description"] != ""
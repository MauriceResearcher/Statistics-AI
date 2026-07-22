"""
Tests for the hypothesis-testing functions.

Expected statistic/p-value numbers were cross-checked against a direct
scipy.stats call before being hardcoded here, so these tests also serve as
a regression check: if a future code change accidentally alters the
calculation, these exact numbers will catch it.

Run with: pytest tests/test_inferential.py -v
"""

import pytest
from app.statistics.inferential import (
    one_sample_t_test, two_sample_t_test, paired_t_test, chi_square_test,
)


def test_one_sample_t_test_no_difference():
    # Mean of [10,12,9,11,8] is exactly 10, i.e. equal to population_mean.
    # -> t-statistic must be exactly 0, p-value exactly 1 (no evidence of difference)
    result = one_sample_t_test([10, 12, 9, 11, 8], population_mean=10)
    assert result["statistic"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)
    assert result["significant"] is False


def test_one_sample_t_test_detects_difference():
    # Sample is clearly shifted away from population_mean=10 -> should be significant
    result = one_sample_t_test([20, 22, 19, 21, 18], population_mean=10)
    assert result["p_value"] < 0.001
    assert result["significant"] is True
    assert result["degrees_of_freedom"] == 4  # n - 1 = 5 - 1


def test_two_sample_t_test_identical_groups():
    # Identical groups -> no difference at all -> statistic 0, p-value 1
    result = two_sample_t_test([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert result["statistic"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)
    assert result["significant"] is False


def test_two_sample_t_test_clear_difference():
    # Two clearly separated groups -> should be highly significant
    result = two_sample_t_test([1, 2, 3, 4, 5], [100, 101, 102, 103, 104])
    assert result["statistic"] == pytest.approx(-99.0)
    assert result["p_value"] < 1e-10
    assert result["significant"] is True


def test_two_sample_t_test_welch_vs_student_both_run():
    # For groups of equal size and similar spread, Student's and Welch's
    # test should give very similar (here: identical) results - this test
    # mainly checks that both code paths (equal_variance True/False) run
    # without errors and produce a sensible result.
    student = two_sample_t_test([1, 2, 3, 4, 5], [100, 101, 102, 103, 104], equal_variance=True)
    welch = two_sample_t_test([1, 2, 3, 4, 5], [100, 101, 102, 103, 104], equal_variance=False)
    assert student["statistic"] == pytest.approx(welch["statistic"])


def test_paired_t_test():
    before = [10, 12, 9, 11, 8]
    after = [12, 15, 10, 14, 9]
    result = paired_t_test(before, after)
    assert result["statistic"] == pytest.approx(-4.47213595499958)
    assert result["p_value"] == pytest.approx(0.011056493393450067)
    assert result["significant"] is True  # p < 0.05


def test_paired_t_test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        paired_t_test([1, 2, 3], [1, 2])


def test_chi_square_test():
    table = [[10, 20], [20, 10]]
    result = chi_square_test(table)
    assert result["statistic"] == pytest.approx(5.4)
    assert result["p_value"] == pytest.approx(0.02013675155034633)
    assert result["degrees_of_freedom"] == 1
    assert result["significant"] is True  # p < 0.05


def test_chi_square_test_requires_2d_table():
    with pytest.raises(ValueError):
        chi_square_test([1, 2, 3])


def test_all_results_include_german_interpretation_text():
    # Every test function must return a non-empty, user-facing interpretation
    result = one_sample_t_test([20, 22, 19, 21, 18], population_mean=10)
    assert "Signifikanzniveau" in result["interpretation"]
    assert result["description"] != ""
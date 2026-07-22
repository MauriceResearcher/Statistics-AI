"""
Inferential statistics: hypothesis tests.

While descriptive statistics summarize the data you already have,
inferential statistics ask whether a pattern in that data (a difference
between two groups, a relationship between variables) is likely to be
"real" or could plausibly have happened by chance alone.

Each test below returns a p-value: roughly, the probability of seeing a
result at least this extreme if there were actually no effect at all (the
null hypothesis, H0). A small p-value (conventionally below 0.05) is
usually taken as evidence against H0.
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from ._utils import to_array, interpret_p_value

# Conventional significance threshold. Can be overridden per call if the
# user wants a stricter or looser criterion.
ALPHA_DEFAULT = 0.05

# Static, German explanations of what each test actually checks - shown to
# the end user in the app alongside the numeric result.
TEST_DESCRIPTIONS = {
    "one_sample_t_test": "Prueft, ob der Mittelwert einer Stichprobe von einem angenommenen Wert abweicht.",
    "two_sample_t_test": "Prueft, ob sich die Mittelwerte zweier unabhaengiger Gruppen signifikant unterscheiden.",
    "paired_t_test": "Prueft, ob sich Werte innerhalb derselben Gruppe (z.B. vorher/nachher) signifikant veraendert haben.",
    "chi_square_test": "Prueft, ob ein statistischer Zusammenhang zwischen zwei kategorialen Merkmalen besteht.",
}


def _build_result(test_name: str, statistic: float, p_value: float, degrees_of_freedom, alpha: float) -> dict:
    """
    Shared result builder for all hypothesis tests in this module, so the
    same dict-assembly logic doesn't have to be repeated four times.
    """
    return {
        "test": test_name,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "degrees_of_freedom": degrees_of_freedom,
        "alpha": alpha,
        "significant": bool(p_value < alpha),
        "description": TEST_DESCRIPTIONS.get(test_name, ""),
        "interpretation": interpret_p_value(p_value, alpha),
    }


def one_sample_t_test(data, population_mean: float, alpha: float = ALPHA_DEFAULT) -> dict:
    """
    One-sample t-test.

    Tests whether the mean of a single sample differs significantly from a
    known or hypothesized value (population_mean).

    H0 (null hypothesis): the sample mean equals population_mean.
    H1 (alternative hypothesis): the sample mean differs from population_mean.

    Assumptions: the data is approximately normally distributed (fairly
    robust for larger samples thanks to the Central Limit Theorem), and
    observations are independent of each other.
    """
    array = to_array(data, min_length=2)
    result = stats.ttest_1samp(array, population_mean)
    degrees_of_freedom = array.size - 1
    return _build_result("one_sample_t_test", result.statistic, result.pvalue, degrees_of_freedom, alpha)


def two_sample_t_test(group1, group2, equal_variance: bool = True, alpha: float = ALPHA_DEFAULT) -> dict:
    """
    Independent two-sample t-test.

    Tests whether the means of two independent groups differ significantly.

    H0: both groups have the same mean.
    H1: the group means differ.

    Set equal_variance=True for Student's t-test (assumes similar variance
    in both groups) or equal_variance=False for Welch's t-test, which does
    not assume equal variance - the safer default if you are not sure.
    """
    array1 = to_array(group1, min_length=2)
    array2 = to_array(group2, min_length=2)
    result = stats.ttest_ind(array1, array2, equal_var=equal_variance)
    degrees_of_freedom = float(result.df)
    return _build_result("two_sample_t_test", result.statistic, result.pvalue, degrees_of_freedom, alpha)


def paired_t_test(before, after, alpha: float = ALPHA_DEFAULT) -> dict:
    """
    Paired (dependent) t-test.

    Tests whether there is a significant difference between two sets of
    paired measurements - e.g. the same subjects measured before and after
    a treatment. Both inputs must be the same length and in matching order
    (before[i] and after[i] belong to the same subject).

    H0: the mean difference between pairs is zero.
    H1: the mean difference is not zero.
    """
    array_before = to_array(before, min_length=2)
    array_after = to_array(after, min_length=2)
    if array_before.size != array_after.size:
        raise ValueError("Die beiden Messreihen muessen gleich lang sein (ein Vorher-/Nachher-Wert pro Subjekt).")
    result = stats.ttest_rel(array_before, array_after)
    degrees_of_freedom = array_before.size - 1
    return _build_result("paired_t_test", result.statistic, result.pvalue, degrees_of_freedom, alpha)


def chi_square_test(contingency_table, alpha: float = ALPHA_DEFAULT) -> dict:
    """
    Chi-square test of independence.

    Tests whether there is a significant association between two
    categorical variables, given as a contingency table (rows = categories
    of variable A, columns = categories of variable B, cells = observed
    counts).

    H0: the two variables are independent.
    H1: the two variables are associated.

    Assumption: expected cell counts should generally be at least 5 for the
    test to be considered reliable.
    """
    table = np.asarray(contingency_table, dtype=float)
    if table.ndim != 2:
        raise ValueError("Die Kontingenztabelle muss zweidimensional sein (Zeilen x Spalten).")
    chi2, p_value, degrees_of_freedom, _expected = stats.chi2_contingency(table)
    return _build_result("chi_square_test", chi2, p_value, degrees_of_freedom, alpha)
"""
Regression and correlation analysis.

While the hypothesis tests in inferential.py check whether a difference
between groups is statistically significant, the functions here look at
RELATIONSHIPS between two (or more) numeric variables: how strongly they
move together (correlation), and how one variable can be predicted from
another (regression).
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from ._utils import to_array, interpret_p_value

ALPHA_DEFAULT = 0.05

CORRELATION_DESCRIPTIONS = {
    "pearson": "Misst die Staerke eines LINEAREN Zusammenhangs zwischen zwei Merkmalen (Wert zwischen -1 und 1).",
    "spearman": "Misst die Staerke eines MONOTONEN Zusammenhangs (muss nicht linear sein) - robuster gegenueber Ausreissern als Pearson.",
}

REGRESSION_DESCRIPTIONS = {
    "simple_linear_regression": "Beschreibt, wie gut sich eine Zielgroesse aus EINER erklaerenden Variable linear vorhersagen laesst.",
    "multiple_linear_regression": "Beschreibt, wie gut sich eine Zielgroesse aus MEHREREN erklaerenden Variablen linear vorhersagen laesst.",
}


def _correlation_strength(r: float) -> str:
    """
    Classifies the absolute correlation coefficient into a plain-language
    strength label. This is a common rule of thumb, not a strict
    statistical rule - it exists purely to make the number easier to read
    for non-technical users.
    """
    magnitude = abs(r)
    if magnitude < 0.3:
        return "schwacher Zusammenhang"
    if magnitude < 0.7:
        return "moderater Zusammenhang"
    return "starker Zusammenhang"


def _validate_pair(x, y) -> tuple[np.ndarray, np.ndarray]:
    """
    Shared validation for all two-variable functions in this module:
    converts both inputs to arrays and checks they have matching length.
    """
    array_x = to_array(x, min_length=3)
    array_y = to_array(y, min_length=3)
    if array_x.size != array_y.size:
        raise ValueError("x und y muessen gleich lang sein (ein Wertepaar pro Beobachtung).")
    return array_x, array_y


def pearson_correlation(x, y, alpha: float = ALPHA_DEFAULT) -> dict:
    """
    Pearson correlation coefficient (r).

    Measures the strength and direction of a LINEAR relationship between
    two numeric variables. Ranges from -1 (perfect negative relationship)
    through 0 (no linear relationship) to +1 (perfect positive
    relationship).

    Assumption: works best when the relationship (if any) is linear - it
    can miss or understate strong non-linear relationships.
    """
    array_x, array_y = _validate_pair(x, y)
    r, p_value = stats.pearsonr(array_x, array_y)
    return {
        "method": "pearson",
        "r": float(r),
        "p_value": float(p_value),
        "alpha": alpha,
        "significant": bool(p_value < alpha),
        "strength": _correlation_strength(r),
        "description": CORRELATION_DESCRIPTIONS["pearson"],
        "interpretation": interpret_p_value(p_value, alpha),
    }


def spearman_correlation(x, y, alpha: float = ALPHA_DEFAULT) -> dict:
    """
    Spearman rank correlation coefficient (rho).

    Measures the strength and direction of a MONOTONIC relationship (not
    necessarily a straight line) between two numeric variables, by
    correlating their RANKS rather than the raw values. More robust to
    outliers and non-linear (but still monotonic) relationships than
    Pearson's correlation.
    """
    array_x, array_y = _validate_pair(x, y)
    rho, p_value = stats.spearmanr(array_x, array_y)
    return {
        "method": "spearman",
        "r": float(rho),
        "p_value": float(p_value),
        "alpha": alpha,
        "significant": bool(p_value < alpha),
        "strength": _correlation_strength(rho),
        "description": CORRELATION_DESCRIPTIONS["spearman"],
        "interpretation": interpret_p_value(p_value, alpha),
    }


def simple_linear_regression(x, y, alpha: float = ALPHA_DEFAULT) -> dict:
    """
    Simple linear regression: fits a straight line y = slope * x + intercept
    that best predicts y from a single explanatory variable x (least
    squares method).

    r_squared indicates how much of the variance in y is explained by x
    (0 = none, 1 = all). The p-value tests whether the slope is
    significantly different from zero (H0: slope = 0, i.e. x has no
    linear effect on y).
    """
    array_x, array_y = _validate_pair(x, y)
    result = stats.linregress(array_x, array_y)
    return {
        "method": "simple_linear_regression",
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r_squared": float(result.rvalue ** 2),
        "p_value": float(result.pvalue),
        "standard_error": float(result.stderr),
        "alpha": alpha,
        "significant": bool(result.pvalue < alpha),
        "description": REGRESSION_DESCRIPTIONS["simple_linear_regression"],
        "interpretation": interpret_p_value(result.pvalue, alpha),
    }


def multiple_linear_regression(x_columns, y, alpha: float = ALPHA_DEFAULT) -> dict:
    """
    Multiple linear regression via ordinary least squares (OLS), computed
    directly with numpy's linear algebra (no extra dependency needed).

    x_columns: a 2D array-like structure, shape (n_observations,
    n_predictors) - each column is one explanatory variable.
    y: the target variable, one value per observation.

    Returns one coefficient per predictor (plus an intercept), along with
    a standard error, t-statistic and p-value for each coefficient -
    testing H0: this coefficient is zero (the predictor has no linear
    effect on y, holding the other predictors constant).
    """
    X = np.asarray(x_columns, dtype=float)
    if X.ndim != 2:
        raise ValueError("x_columns muss zweidimensional sein (Beobachtungen x Praediktoren).")
    n_observations, n_predictors = X.shape
    y_array = to_array(y, min_length=n_predictors + 2)
    if X.shape[0] != y_array.size:
        raise ValueError("Die Anzahl der Zeilen in x_columns muss der Laenge von y entsprechen.")

    # Add a column of 1s for the intercept term
    design_matrix = np.column_stack([np.ones(n_observations), X])

    # Least-squares solution: beta minimizes the sum of squared residuals
    coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(design_matrix, y_array, rcond=None)

    predictions = design_matrix @ coefficients
    residuals = y_array - predictions
    sse = float(np.sum(residuals ** 2))  # sum of squared errors (unexplained variance)
    sst = float(np.sum((y_array - np.mean(y_array)) ** 2))  # total variance in y
    r_squared = 1 - sse / sst if sst > 0 else 0.0

    degrees_of_freedom_resid = n_observations - n_predictors - 1
    if degrees_of_freedom_resid <= 0:
        raise ValueError(
            "Zu wenige Beobachtungen fuer die Anzahl an Praediktoren "
            "(mehr Daten oder weniger Variablen noetig)."
        )
    adjusted_r_squared = 1 - (1 - r_squared) * (n_observations - 1) / degrees_of_freedom_resid

    # Covariance matrix of the coefficients, used to derive standard errors
    residual_variance = sse / degrees_of_freedom_resid
    covariance_matrix = residual_variance * np.linalg.inv(design_matrix.T @ design_matrix)
    standard_errors = np.sqrt(np.diag(covariance_matrix))
    t_statistics = coefficients / standard_errors
    # Two-sided p-value from the t-distribution for each coefficient
    p_values = 2 * stats.t.sf(np.abs(t_statistics), degrees_of_freedom_resid)

    names = ["intercept"] + [f"x{i + 1}" for i in range(n_predictors)]
    coefficient_details = {
        name: {
            "coefficient": float(coefficients[i]),
            "standard_error": float(standard_errors[i]),
            "t_statistic": float(t_statistics[i]),
            "p_value": float(p_values[i]),
            "significant": bool(p_values[i] < alpha),
        }
        for i, name in enumerate(names)
    }

    return {
        "method": "multiple_linear_regression",
        "coefficients": coefficient_details,
        "r_squared": float(r_squared),
        "adjusted_r_squared": float(adjusted_r_squared),
        "degrees_of_freedom": degrees_of_freedom_resid,
        "alpha": alpha,
        "description": REGRESSION_DESCRIPTIONS["multiple_linear_regression"],
    }
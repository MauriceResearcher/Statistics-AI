"""
Tests for chatbot.py.

Only tests build_tools() directly - calling the returned functions like
normal Python functions, without going through Gemini. This checks our
own wiring logic (column lookup, NaN handling, error dicts); no network
or API key needed. The real end-to-end API call is checked manually via
chatbot_manual_check.py instead (see that file for why).

Run with: pytest tests/test_chatbot.py -v
"""

import pandas as pd
from app.chatbot import build_tools


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [23, 45, 31, 29, 52, 38, 41, 27, 33, 48],
        "city": ["Berlin", "Hamburg", "Berlin", "Muenchen", "Berlin",
                  "Hamburg", "Berlin", "Hamburg", "Muenchen", "Berlin"],
        "income": [2200, 3100, 2500, 2000, 3300, 2900, 2600, 2800, 2100, 3400],
        "experience": [1, 10, 5, 3, 15, 8, 9, 2, 4, 12],
        "satisfied": ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "no", "yes"],
    })


def _tools_by_name() -> dict:
    # Lookup by function name instead of list position, so the tests
    # don't silently break if the tool order in build_tools() changes.
    return {tool.__name__: tool for tool in build_tools(_sample_dataframe())}


def test_get_descriptive_summary():
    tools = _tools_by_name()
    result = tools["get_descriptive_summary"]("age")
    assert "error" not in result
    assert "description" in result["mean"]


def test_run_one_sample_t_test():
    tools = _tools_by_name()
    result = tools["run_one_sample_t_test"]("age", population_mean=30)
    assert "error" not in result
    assert "p_value" in result


def test_run_two_sample_t_test_by_group():
    tools = _tools_by_name()
    result = tools["run_two_sample_t_test"]("income", "city", "Berlin", "Hamburg")
    assert "error" not in result
    assert "p_value" in result


def test_run_two_sample_t_test_unknown_group_value():
    # "Kiel" does not occur in the city column -> should return a
    # friendly error dict, not crash or silently compare empty groups.
    tools = _tools_by_name()
    result = tools["run_two_sample_t_test"]("income", "city", "Berlin", "Kiel")
    assert "error" in result


def test_run_paired_t_test():
    tools = _tools_by_name()
    # Using age/experience just to check the wiring - not claiming these
    # are meaningfully "paired" in a real-world sense.
    result = tools["run_paired_t_test"]("age", "experience")
    assert "error" not in result
    assert "p_value" in result


def test_run_chi_square_test():
    tools = _tools_by_name()
    result = tools["run_chi_square_test"]("city", "satisfied")
    assert "error" not in result
    assert "p_value" in result


def test_run_correlation_pearson_and_spearman():
    tools = _tools_by_name()
    pearson_result = tools["run_correlation"]("age", "income", method="pearson")
    spearman_result = tools["run_correlation"]("age", "income", method="spearman")
    assert "error" not in pearson_result
    assert "error" not in spearman_result
    assert pearson_result["method"] == "pearson"
    assert spearman_result["method"] == "spearman"


def test_run_simple_regression():
    tools = _tools_by_name()
    result = tools["run_simple_regression"]("age", "income")
    assert "error" not in result
    assert "slope" in result


def test_run_multiple_regression():
    tools = _tools_by_name()
    result = tools["run_multiple_regression"]("income", ["age", "experience"])
    assert "error" not in result
    assert "x1" in result["coefficients"]
    assert "x2" in result["coefficients"]


def test_unknown_column_returns_error_dict_not_exception():
    tools = _tools_by_name()
    result = tools["run_simple_regression"]("does_not_exist", "income")
    assert "error" in result


def test_tools_normalize_messy_column_names():
    # Models sometimes wrap arguments in stray quotes/whitespace - the
    # tools should clean that up automatically rather than failing.
    tools = _tools_by_name()
    result = tools["get_descriptive_summary"]("  'age' ")
    assert "error" not in result


def test_run_two_sample_t_test_with_messy_input():
    tools = _tools_by_name()
    result = tools["run_two_sample_t_test"](' income ', '"city"', ' Berlin ', 'Hamburg')
    assert "error" not in result
    assert "p_value" in result
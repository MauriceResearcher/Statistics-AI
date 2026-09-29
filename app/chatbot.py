"""
Chatbot with Gemini function calling.

The core idea: we hand the Gemini SDK plain Python functions as "tools".
The SDK inspects each function's type hints and docstring to build the
schema Gemini sees, and - when Gemini decides a function should be
called - the SDK calls the REAL Python function itself and sends the
result back automatically ("automatic function calling"). We never send
Gemini the actual dataset, only column names as short strings; the real
calculation always happens locally, inside our already-tested functions
from descriptive.py / inferential.py / regression.py.
"""

import difflib
import logging
import pandas as pd
from google import genai
from google.genai import types

from app.statistics.descriptive import summary_statistics
from app.statistics.inferential import (
    one_sample_t_test, two_sample_t_test, paired_t_test, chi_square_test,
)
from app.statistics.regression import (
    pearson_correlation, spearman_correlation,
    simple_linear_regression, multiple_linear_regression,
)

# Server-seitiges Logging: sichtbar in der Konsole/den Server-Logs, aber
# NIE direkt dem Nutzer angezeigt (siehe streamlit_app.py) - so bleiben
# interne Details (Stacktraces, Bibliotheks-Fehlermeldungen) intern.
logger = logging.getLogger(__name__)

BASE_SYSTEM_INSTRUCTION = (
    "Du bist ein Statistik-Assistent. Du hilfst dabei, Kennzahlen zu einem "
    "hochgeladenen Datensatz zu berechnen, indem du ausschliesslich die "
    "bereitgestellten Funktionen (Tools) aufrufst - führe niemals eigene "
    "Berechnungen im Kopf durch, auch wenn du sie für einfach hältst. "
    "Wähle den passenden Test anhand der Fragestellung und der "
    "Spaltentypen (z.B. Chi-Quadrat-Test für zwei kategoriale Spalten, "
    "t-Test für den Vergleich numerischer Mittelwerte). Antworte auf "
    "Deutsch, in einfacher, für Laien verständlicher Sprache. Wenn ein "
    "Spaltenname unklar ist oder nicht existiert, frage nach, anstatt zu "
    "raten."
)

DEFAULT_MODEL = "gemini-3.5-flash-lite"


def _normalize(value):
    """
    Cleans up a single argument the model provided: strips stray
    whitespace and quote characters models sometimes add around values
    (e.g. "'age'" instead of "age"). Non-string values are returned
    unchanged.
    """
    if isinstance(value, str):
        return value.strip().strip("'\"")
    return value


def _resolve_column(requested_name: str, available_columns: list[str]) -> str:
    """
    Resolves a requested column name to an actual column name in the DataFrame.
    1. Exact match
    2. Case-insensitive match (e.g. 'age' -> 'Age')
    3. Partial match (e.g. 'alter' -> 'Alter_Jahre')
    4. Fuzzy match using difflib
    """
    if not requested_name or not available_columns:
        return requested_name

    # 1. Exakter Match
    if requested_name in available_columns:
        return requested_name

    cleaned_req = requested_name.strip().lower()

    # 2. Case-insensitive Match
    col_map = {col.lower(): col for col in available_columns}
    if cleaned_req in col_map:
        return col_map[cleaned_req]

    # 3. Substring/Teil-Match (z.B. 'gehalt' in 'brutto_gehalt')
    partial_matches = [col for col in available_columns if cleaned_req in col.lower()]
    if len(partial_matches) == 1:
        return partial_matches[0]

    # 4. Fuzzy Match (Toleranz für Tippfehler)
    matches = difflib.get_close_matches(requested_name, available_columns, n=1, cutoff=0.6)
    if matches:
        return matches[0]

    return requested_name


def build_tools(dataframe: pd.DataFrame) -> list:
    """
    Builds the list of tool functions bound to one specific DataFrame via
    closure. Gemini only ever sees column names (short strings) and
    simple values that it passes as arguments - never the DataFrame
    itself. The actual data access happens here, entirely on our side.
    """
    available_cols = list(dataframe.columns)

    def _resolve_and_validate_columns(*columns: str):
        """
        Helper that resolves and checks any number of requested column names.
        Returns:
            (error_dict, resolved_columns_list)
            If all columns are resolved: (None, ['col1', 'col2', ...])
            If any column fails: ({'error': '...'}, None)
        """
        resolved_cols = []
        missing = []

        for col in columns:
            col_norm = _normalize(col)
            resolved = _resolve_column(col_norm, available_cols)
            if resolved in dataframe.columns:
                resolved_cols.append(resolved)
            else:
                missing.append(col)

        if missing:
            return {
                "error": f"Spalte(n) nicht gefunden: {', '.join(missing)}. "
                         f"Verfügbare Spalten sind: {available_cols}"
            }, None

        return None, resolved_cols

    def get_descriptive_summary(column: str) -> dict:
        """
        Computes descriptive statistics for one numeric column of the
        loaded dataset: mean, median, mode, standard deviation, range,
        and interquartile range, each with a short explanation.

        Args:
            column: Name of the numeric column to summarize.
        """
        error, resolved = _resolve_and_validate_columns(column)
        if error:
            return error

        target_col = resolved[0]
        values = dataframe[target_col].dropna().tolist()
        try:
            return summary_statistics(values)
        except ValueError as error:
            return {"error": str(error)}
        except Exception:
            logger.exception("Unerwarteter Fehler in get_descriptive_summary")
            return {"error": "Bei der Berechnung ist ein unerwarteter Fehler aufgetreten."}

    def run_one_sample_t_test(column: str, population_mean: float) -> dict:
        """
        Tests whether the mean of a numeric column differs significantly
        from a given reference value.

        Args:
            column: Name of the numeric column to test.
            population_mean: The reference value to compare the column's mean against.
        """
        error, resolved = _resolve_and_validate_columns(column)
        if error:
            return error

        target_col = resolved[0]
        values = dataframe[target_col].dropna().tolist()
        try:
            return one_sample_t_test(values, population_mean)
        except ValueError as error:
            return {"error": str(error)}
        except Exception:
            logger.exception("Unerwarteter Fehler in run_one_sample_t_test")
            return {"error": "Bei der Berechnung ist ein unerwarteter Fehler aufgetreten."}

    def run_two_sample_t_test(
        value_column: str, group_column: str, group_a: str, group_b: str,
        equal_variance: bool = True,
    ) -> dict:
        """
        Compares the mean of a numeric column between two groups defined
        by a categorical column - e.g. compare 'income' between city
        'Berlin' and 'Hamburg'.

        Args:
            value_column: Name of the numeric column to compare.
            group_column: Name of the categorical column defining the groups.
            group_a: Value in group_column identifying the first group.
            group_b: Value in group_column identifying the second group.
            equal_variance: True for Student's t-test, False for Welch's t-test.
        """
        group_a, group_b = _normalize(group_a), _normalize(group_b)

        error, resolved = _resolve_and_validate_columns(value_column, group_column)
        if error:
            return error

        val_col, grp_col = resolved[0], resolved[1]

        group1 = dataframe.loc[dataframe[grp_col] == group_a, val_col].dropna().tolist()
        group2 = dataframe.loc[dataframe[grp_col] == group_b, val_col].dropna().tolist()

        if not group1:
            return {"error": f"Keine Werte für '{grp_col} == {group_a}' gefunden."}
        if not group2:
            return {"error": f"Keine Werte für '{grp_col} == {group_b}' gefunden."}

        try:
            return two_sample_t_test(group1, group2, equal_variance=equal_variance)
        except ValueError as error:
            return {"error": str(error)}
        except Exception:
            logger.exception("Unerwarteter Fehler in run_two_sample_t_test")
            return {"error": "Bei der Berechnung ist ein unerwarteter Fehler aufgetreten."}

    def run_paired_t_test(column_before: str, column_after: str) -> dict:
        """
        Tests whether there is a significant difference between two
        matched measurements of the same rows - e.g. 'weight_before' vs
        'weight_after'.

        Args:
            column_before: Name of the "before" numeric column.
            column_after: Name of the "after" numeric column.
        """
        error, resolved = _resolve_and_validate_columns(column_before, column_after)
        if error:
            return error

        col_before, col_after = resolved[0], resolved[1]
        paired = dataframe[[col_before, col_after]].dropna()
        try:
            return paired_t_test(paired[col_before].tolist(), paired[col_after].tolist())
        except ValueError as error:
            return {"error": str(error)}
        except Exception:
            logger.exception("Unerwarteter Fehler in run_paired_t_test")
            return {"error": "Bei der Berechnung ist ein unerwarteter Fehler aufgetreten."}

    def run_chi_square_test(column_a: str, column_b: str) -> dict:
        """
        Tests whether there is a significant association between two
        categorical columns (e.g. 'city' and 'satisfied').

        Args:
            column_a: Name of the first categorical column.
            column_b: Name of the second categorical column.
        """
        error, resolved = _resolve_and_validate_columns(column_a, column_b)
        if error:
            return error

        col_a, col_b = resolved[0], resolved[1]
        subset = dataframe[[col_a, col_b]].dropna()
        contingency_table = pd.crosstab(subset[col_a], subset[col_b])
        try:
            return chi_square_test(contingency_table.values)
        except ValueError as error:
            return {"error": str(error)}
        except Exception:
            logger.exception("Unerwarteter Fehler in run_chi_square_test")
            return {"error": "Bei der Berechnung ist ein unerwarteter Fehler aufgetreten."}

    def run_correlation(column_x: str, column_y: str, method: str = "pearson") -> dict:
        """
        Measures the correlation between two numeric columns.

        Args:
            column_x: Name of the first numeric column.
            column_y: Name of the second numeric column.
            method: Either "pearson" (linear relationships) or "spearman"
                (monotonic relationships, more robust to outliers).
        """
        method = _normalize(method)
        error, resolved = _resolve_and_validate_columns(column_x, column_y)
        if error:
            return error

        col_x, col_y = resolved[0], resolved[1]
        paired = dataframe[[col_x, col_y]].dropna()
        correlation_function = pearson_correlation if method == "pearson" else spearman_correlation
        try:
            return correlation_function(paired[col_x].tolist(), paired[col_y].tolist())
        except ValueError as error:
            return {"error": str(error)}
        except Exception:
            logger.exception("Unerwarteter Fehler in run_correlation")
            return {"error": "Bei der Berechnung ist ein unerwarteter Fehler aufgetreten."}

    def run_simple_regression(column_x: str, column_y: str) -> dict:
        """
        Fits a simple linear regression predicting column_y from column_x.

        Args:
            column_x: Name of the explanatory (predictor) numeric column.
            column_y: Name of the target numeric column.
        """
        error, resolved = _resolve_and_validate_columns(column_x, column_y)
        if error:
            return error

        col_x, col_y = resolved[0], resolved[1]
        paired = dataframe[[col_x, col_y]].dropna()
        try:
            return simple_linear_regression(paired[col_x].tolist(), paired[col_y].tolist())
        except ValueError as error:
            return {"error": str(error)}
        except Exception:
            logger.exception("Unerwarteter Fehler in run_simple_regression")
            return {"error": "Bei der Berechnung ist ein unerwarteter Fehler aufgetreten."}

    def run_multiple_regression(target_column: str, predictor_columns: list[str]) -> dict:
        """
        Fits a multiple linear regression predicting target_column from
        several predictor columns at once.

        Args:
            target_column: Name of the numeric column to predict.
            predictor_columns: Names of the numeric predictor columns.
        """
        all_requested = [target_column] + predictor_columns
        error, resolved = _resolve_and_validate_columns(*all_requested)
        if error:
            return error

        resolved_target = resolved[0]
        resolved_predictors = resolved[1:]

        subset = dataframe[[resolved_target] + resolved_predictors].dropna()
        try:
            return multiple_linear_regression(
                subset[resolved_predictors].values, subset[resolved_target].tolist()
            )
        except ValueError as error:
            return {"error": str(error)}
        except Exception:
            logger.exception("Unerwarteter Fehler in run_multiple_regression")
            return {"error": "Bei der Berechnung ist ein unerwarteter Fehler aufgetreten."}

    return [
        get_descriptive_summary,
        run_one_sample_t_test,
        run_two_sample_t_test,
        run_paired_t_test,
        run_chi_square_test,
        run_correlation,
        run_simple_regression,
        run_multiple_regression,
    ]


class StatisticsChatbot:
    """
    Wraps Gemini chat calls with our statistics functions available as
    tools. Create one instance per uploaded dataset, since the tools are
    bound to that specific DataFrame.
    """

    def __init__(self, dataframe: pd.DataFrame, api_key: str, model: str = DEFAULT_MODEL):
        self.dataframe = dataframe
        self.api_key = api_key
        self.model = model
        self.tools = build_tools(dataframe)

        # Füge die verfuegbaren Spalten direkt in die System-Instruction ein,
        # damit Gemini von Beginn an die genauen Namen kennt.
        cols_str = ", ".join(f"'{c}'" for c in dataframe.columns)
        self.system_instruction = (
            f"{BASE_SYSTEM_INSTRUCTION}\n\n"
            f"Die verfügbaren Spalten im geladenen Datensatz lauten: [{cols_str}]."
        )
        self.history = []

    def ask(self, message: str) -> str:
        """
        Sends a user message to the chatbot and returns its final,
        plain-text reply. Any tool calls the model decides to make are
        handled automatically by the SDK behind the scenes.
        """
        client = genai.Client(api_key=self.api_key)
        chat = client.chats.create(
            model=self.model,
            config=types.GenerateContentConfig(
                tools=self.tools,
                system_instruction=self.system_instruction,
            ),
            history=self.history,
        )
        response = chat.send_message(message)
        self.history = chat.get_history()
        return response.text
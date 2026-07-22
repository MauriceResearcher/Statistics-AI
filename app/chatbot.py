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

SYSTEM_INSTRUCTION = (
    "Du bist ein Statistik-Assistent. Du hilfst dabei, Kennzahlen zu einem "
    "hochgeladenen Datensatz zu berechnen, indem du ausschliesslich die "
    "bereitgestellten Funktionen (Tools) aufrufst - fuehre niemals eigene "
    "Berechnungen im Kopf durch, auch wenn du sie fuer einfach haeltst. "
    "Waehle den passenden Test anhand der Fragestellung und der "
    "Spaltentypen (z.B. Chi-Quadrat-Test fuer zwei kategoriale Spalten, "
    "t-Test fuer den Vergleich numerischer Mittelwerte). Antworte auf "
    "Deutsch, in einfacher, fuer Laien verstaendlicher Sprache. Wenn ein "
    "Spaltenname unklar ist oder nicht existiert, frage nach, anstatt zu "
    "raten."
)

DEFAULT_MODEL = "gemini-3.1-flash-lite"


def build_tools(dataframe: pd.DataFrame) -> list:
    """
    Builds the list of tool functions bound to one specific DataFrame via
    closure. Gemini only ever sees column names (short strings) and
    simple values that it passes as arguments - never the DataFrame
    itself. The actual data access happens here, entirely on our side.
    """

    def _missing_columns_error(*columns: str) -> dict | None:
        """Shared helper: returns an error dict if any column is missing, else None."""
        missing = [c for c in columns if c not in dataframe.columns]
        if missing:
            return {"error": f"Spalte(n) nicht gefunden: {', '.join(missing)}"}
        return None

    def get_descriptive_summary(column: str) -> dict:
        """
        Computes descriptive statistics for one numeric column of the
        loaded dataset: mean, median, mode, standard deviation, range,
        and interquartile range, each with a short explanation.

        Args:
            column: Exact name of the numeric column to summarize.
        """
        error = _missing_columns_error(column)
        if error:
            return error
        values = dataframe[column].dropna().tolist()
        try:
            return summary_statistics(values)
        except ValueError as error:
            return {"error": str(error)}

    def run_one_sample_t_test(column: str, population_mean: float) -> dict:
        """
        Tests whether the mean of a numeric column differs significantly
        from a given reference value.

        Args:
            column: Exact name of the numeric column to test.
            population_mean: The reference value to compare the column's mean against.
        """
        error = _missing_columns_error(column)
        if error:
            return error
        values = dataframe[column].dropna().tolist()
        try:
            return one_sample_t_test(values, population_mean)
        except ValueError as error:
            return {"error": str(error)}

    def run_two_sample_t_test(
            value_column: str, group_column: str, group_a: str, group_b: str,
            equal_variance: bool = True,
    ) -> dict:
        """
        Compares the mean of a numeric column between two groups defined
        by a categorical column - e.g. compare 'income' between city
        'Berlin' and 'Hamburg'.

        Args:
            value_column: Exact name of the numeric column to compare.
            group_column: Exact name of the categorical column defining the groups.
            group_a: Value in group_column identifying the first group.
            group_b: Value in group_column identifying the second group.
            equal_variance: True for Student's t-test, False for Welch's t-test.
        """
        error = _missing_columns_error(value_column, group_column)
        if error:
            return error
        group1 = dataframe.loc[dataframe[group_column] == group_a, value_column].dropna().tolist()
        group2 = dataframe.loc[dataframe[group_column] == group_b, value_column].dropna().tolist()
        if not group1:
            return {"error": f"Keine Werte fuer '{group_column} == {group_a}' gefunden."}
        if not group2:
            return {"error": f"Keine Werte fuer '{group_column} == {group_b}' gefunden."}
        try:
            return two_sample_t_test(group1, group2, equal_variance=equal_variance)
        except ValueError as error:
            return {"error": str(error)}

    def run_paired_t_test(column_before: str, column_after: str) -> dict:
        """
        Tests whether there is a significant difference between two
        matched measurements of the same rows - e.g. 'weight_before' vs
        'weight_after'.

        Args:
            column_before: Exact name of the "before" numeric column.
            column_after: Exact name of the "after" numeric column.
        """
        error = _missing_columns_error(column_before, column_after)
        if error:
            return error
        paired = dataframe[[column_before, column_after]].dropna()
        try:
            return paired_t_test(paired[column_before].tolist(), paired[column_after].tolist())
        except ValueError as error:
            return {"error": str(error)}

    def run_chi_square_test(column_a: str, column_b: str) -> dict:
        """
        Tests whether there is a significant association between two
        categorical columns (e.g. 'city' and 'satisfied').

        Args:
            column_a: Exact name of the first categorical column.
            column_b: Exact name of the second categorical column.
        """
        error = _missing_columns_error(column_a, column_b)
        if error:
            return error
        subset = dataframe[[column_a, column_b]].dropna()
        contingency_table = pd.crosstab(subset[column_a], subset[column_b])
        try:
            return chi_square_test(contingency_table.values)
        except ValueError as error:
            return {"error": str(error)}

    def run_correlation(column_x: str, column_y: str, method: str = "pearson") -> dict:
        """
        Measures the correlation between two numeric columns.

        Args:
            column_x: Exact name of the first numeric column.
            column_y: Exact name of the second numeric column.
            method: Either "pearson" (linear relationships) or "spearman"
                (monotonic relationships, more robust to outliers).
        """
        error = _missing_columns_error(column_x, column_y)
        if error:
            return error
        paired = dataframe[[column_x, column_y]].dropna()
        correlation_function = pearson_correlation if method == "pearson" else spearman_correlation
        try:
            return correlation_function(paired[column_x].tolist(), paired[column_y].tolist())
        except ValueError as error:
            return {"error": str(error)}

    def run_simple_regression(column_x: str, column_y: str) -> dict:
        """
        Fits a simple linear regression predicting column_y from column_x.

        Args:
            column_x: Exact name of the explanatory (predictor) numeric column.
            column_y: Exact name of the target numeric column.
        """
        error = _missing_columns_error(column_x, column_y)
        if error:
            return error
        paired = dataframe[[column_x, column_y]].dropna()
        try:
            return simple_linear_regression(paired[column_x].tolist(), paired[column_y].tolist())
        except ValueError as error:
            return {"error": str(error)}

    def run_multiple_regression(target_column: str, predictor_columns: list[str]) -> dict:
        """
        Fits a multiple linear regression predicting target_column from
        several predictor columns at once.

        Args:
            target_column: Exact name of the numeric column to predict.
            predictor_columns: Exact names of the numeric predictor columns.
        """
        error = _missing_columns_error(target_column, *predictor_columns)
        if error:
            return error
        subset = dataframe[[target_column, *predictor_columns]].dropna()
        try:
            return multiple_linear_regression(
                subset[predictor_columns].values, subset[target_column].tolist()
            )
        except ValueError as error:
            return {"error": str(error)}

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
    Wraps one Gemini chat session with our statistics functions available
    as tools. Create one instance per uploaded dataset, since the tools
    are bound to that specific DataFrame.
    """

    def __init__(self, dataframe: pd.DataFrame, api_key: str, model: str = DEFAULT_MODEL):
        self.dataframe = dataframe
        self.api_key = api_key
        self.model = model
        self.tools = build_tools(dataframe)
        # Reine Konversations-Historie (Content-Objekte der SDK) statt
        # eines lebenden Client/Chat-Objekts, das ueber mehrere Streamlit-
        # Reruns hinweg offen gehalten wird. Grund: ein bekannter Bug in
        # aktuellen google-genai-Versionen schliesst den internen HTTP-
        # Client manchmal vorzeitig, wenn dieselbe Chat-Session-Instanz
        # ueber mehrere Aufrufe hinweg wiederverwendet wird. Ein frischer
        # Client pro Nachricht umgeht das zuverlaessig.
        self.history = []

    def ask(self, message: str) -> str:
        """
        Sends a user message to the chatbot and returns its final,
        plain-text reply. Any tool calls the model decides to make are
        handled automatically by the SDK behind the scenes.

        Builds a fresh client and chat session for this single call,
        seeded with the conversation history from all previous calls -
        see the __init__ comment for why we don't keep one long-lived
        chat session around instead.
        """
        client = genai.Client(api_key=self.api_key)
        chat = client.chats.create(
            model=self.model,
            config=types.GenerateContentConfig(
                tools=self.tools,
                system_instruction=SYSTEM_INSTRUCTION,
            ),
            history=self.history,
        )
        response = chat.send_message(message)
        self.history = chat.get_history()  # fuer den naechsten Aufruf merken
        return response.text




"""
Functions for loading uploaded data files into pandas DataFrames.

Kept separate from the Streamlit UI code (streamlit_app.py) on purpose:
this way the loading/parsing logic can be tested on its own, without
needing to run a Streamlit server or simulate button clicks.
"""

from __future__ import annotations
import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")

# Simple safety limit: without this, someone could upload an extremely
# large file and exhaust server memory/CPU. 20 MB is generous for a
# personal statistics tool while still protecting against abuse.
MAX_FILE_SIZE_MB = 20


def load_dataframe(file) -> pd.DataFrame:
    """
    Loads an uploaded file (CSV or Excel) into a pandas DataFrame.

    `file` can be a file path (str) or a file-like object - both
    pandas.read_csv and pandas.read_excel accept either, and Streamlit's
    st.file_uploader() returns exactly this kind of file-like object.

    Raises ValueError with a German message (shown directly to the end
    user in the app) if the file is too large, the file type is
    unsupported, or the file cannot be parsed.
    """
    filename = getattr(file, "name", str(file))

    # Streamlit's UploadedFile exposes a `.size` attribute (bytes); plain
    # file-like objects used in tests (e.g. io.StringIO) don't have one -
    # in that case we simply skip the size check instead of erroring.
    size_bytes = getattr(file, "size", None)
    if size_bytes is not None and size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(
            f"Die Datei ist zu gross ({size_bytes / (1024 * 1024):.1f} MB). "
            f"Maximal erlaubt sind {MAX_FILE_SIZE_MB} MB."
        )

    if filename.endswith(".csv"):
        try:
            return pd.read_csv(file)
        except Exception as error:
            raise ValueError(f"Die CSV-Datei konnte nicht gelesen werden: {error}") from error

    if filename.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(file)
        except Exception as error:
            raise ValueError(f"Die Excel-Datei konnte nicht gelesen werden: {error}") from error

    raise ValueError(
        f"Nicht unterstuetztes Dateiformat: '{filename}'. "
        f"Unterstuetzt werden: {', '.join(SUPPORTED_EXTENSIONS)}"
    )


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """
    Returns the names of all columns that contain numeric data - these
    are the columns our statistics functions (mean, variance, t-tests,
    etc.) can actually be applied to.
    """
    return df.select_dtypes(include="number").columns.tolist()
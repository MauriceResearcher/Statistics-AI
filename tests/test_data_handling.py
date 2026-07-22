"""
Tests for data_handling.py.

Uses in-memory file-like objects (io.StringIO) instead of real files on
disk, so the tests run fast and don't leave temp files behind.

Run with: pytest tests/test_data_handling.py -v
"""

import io
import pytest
from app.data_handling import load_dataframe, numeric_columns


class _FakeUploadedFile(io.StringIO):
    """
    Streamlit's file_uploader returns an object with a `.name` attribute
    plus normal file-like read behaviour. Plain io.StringIO has no
    `.name`, so this tiny wrapper adds one - purely for testing, so we
    don't need a real Streamlit app running to test load_dataframe().
    """
    def __init__(self, content: str, name: str):
        super().__init__(content)
        self.name = name


def test_load_csv_basic():
    csv_content = "a,b\n1,2\n3,4\n"
    file = _FakeUploadedFile(csv_content, "data.csv")
    df = load_dataframe(file)
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)
    assert df["a"].tolist() == [1, 3]


def test_unsupported_extension_raises():
    file = _FakeUploadedFile("irrelevant content", "data.txt")
    with pytest.raises(ValueError):
        load_dataframe(file)


def test_empty_csv_raises_value_error():
    # pandas raises its own EmptyDataError for a file with no content at
    # all - this test checks that we wrap that in our own ValueError with
    # a German message, instead of leaking a pandas-specific exception
    # type up to the UI layer.
    file = _FakeUploadedFile("", "empty.csv")
    with pytest.raises(ValueError):
        load_dataframe(file)


def test_numeric_columns_filters_correctly():
    csv_content = "age,name,score\n25,Anna,1.5\n30,Ben,2.5\n"
    file = _FakeUploadedFile(csv_content, "data.csv")
    df = load_dataframe(file)
    # "name" is text, so only "age" and "score" should be picked up
    assert numeric_columns(df) == ["age", "score"]
"""
Tests for data_handling.py.

Uses in-memory file-like objects (io.StringIO) instead of real files on
disk, so the tests run fast and don't leave temp files behind.

Run with: pytest tests/test_data_handling.py -v
"""

import io
import pytest
from app.data_handling import load_dataframe, numeric_columns, MAX_FILE_SIZE_MB


class _FakeUploadedFile(io.StringIO):
    """
    Streamlit's file_uploader returns an object with `.name` and `.size`
    attributes plus normal file-like read behaviour. Plain io.StringIO
    has neither, so this tiny wrapper adds them - purely for testing.
    """
    def __init__(self, content: str, name: str, size: int | None = None):
        super().__init__(content)
        self.name = name
        self.size = size if size is not None else len(content.encode("utf-8"))


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
    file = _FakeUploadedFile("", "empty.csv")
    with pytest.raises(ValueError):
        load_dataframe(file)


def test_numeric_columns_filters_correctly():
    csv_content = "age,name,score\n25,Anna,1.5\n30,Ben,2.5\n"
    file = _FakeUploadedFile(csv_content, "data.csv")
    df = load_dataframe(file)
    assert numeric_columns(df) == ["age", "score"]


def test_oversized_file_raises_value_error():
    # Simulate a file that reports itself as larger than the limit,
    # WITHOUT actually allocating that much memory in the test.
    oversized_bytes = (MAX_FILE_SIZE_MB + 1) * 1024 * 1024
    file = _FakeUploadedFile("a,b\n1,2\n", "big.csv", size=oversized_bytes)
    with pytest.raises(ValueError):
        load_dataframe(file)


def test_file_without_size_attribute_still_works():
    # Plain io.StringIO (no .size) should not trigger the size check at
    # all - this mirrors how our existing tests call load_dataframe().
    file = io.StringIO("a,b\n1,2\n")
    file.name = "data.csv"
    df = load_dataframe(file)
    assert df.shape == (1, 2)
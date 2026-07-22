"""
Descriptive statistics functions.

Descriptive statistics summarize an existing dataset with individual
metrics (e.g. "where is the center of the data?", "how spread out are the
values?"). They make NO claim about whether these patterns would also
appear in other, new data - that's what the functions in inferential.py
are for.

Each function takes a list (or other iterable) of numbers and returns a
single metric. At the end of the file there is a summary function that
bundles all metrics together with a short, plain-language explanation -
exactly what the chatbot will later present to a user.
"""

from __future__ import annotations
import numpy as np
from ._utils import to_array


def mean(data) -> float:
    """
    Arithmetic mean (colloquially "the average"): sum of all values
    divided by the count. Sensitive to outliers - for skewed distributions
    (e.g. income, house prices), the median is often more informative.
    """
    array = to_array(data)
    return float(np.mean(array))


def median(data) -> float:
    """
    Median (central value): the value that sits exactly in the middle once
    all values are sorted. For an even number of values, the mean of the
    two middle values is used.

    Advantage over the mean: robust to outliers, because only the position
    in the sorted order matters, not the absolute size of the value.
    """
    array = to_array(data)
    return float(np.median(array))


def mode(data) -> list[float]:
    """
    Mode: the most frequently occurring value(s). Returns a list because
    there can be multiple equally frequent values (a multimodal
    distribution).
    """
    array = to_array(data)
    values, counts = np.unique(array, return_counts=True)
    max_count = counts.max()
    return [float(v) for v in values[counts == max_count]]


def variance(data, ddof: int = 1) -> float:
    """
    Variance: the average squared deviation from the mean - a measure of
    how much the values spread out overall.

    ddof (delta degrees of freedom) = 1 is the standard for a SAMPLE
    (Bessel's correction: dividing by n-1 instead of n). This is the
    normal case whenever your data is only a subset of a larger
    population - true in practice almost always. ddof=0 would compute the
    variance of the ENTIRE population (only appropriate if your data
    really covers every existing case).
    """
    array = to_array(data)
    if array.size <= ddof:
        raise ValueError(f"Fuer die Varianz mit ddof={ddof} werden mehr als {ddof} Werte benoetigt.")
    return float(np.var(array, ddof=ddof))


def std_dev(data, ddof: int = 1) -> float:
    """
    Standard deviation = square root of the variance. Advantage over the
    variance: it has the same unit as the original data (e.g. "euros"
    instead of "euros squared") and is therefore easier to interpret.

    Rule of thumb for approximately normally distributed data: about 68%
    of values fall within one standard deviation of the mean.
    """
    return float(np.sqrt(variance(data, ddof=ddof)))


def value_range(data) -> float:
    """
    Range: difference between the largest and smallest value. The
    simplest measure of spread, but very sensitive to individual
    outliers, since only the two extreme values are considered.
    """
    array = to_array(data)
    return float(np.max(array) - np.min(array))


def quartiles(data) -> tuple[float, float, float]:
    """
    Quartiles split the sorted data into four equal parts:
    Q1 (25th percentile), Q2 (50th percentile = median), Q3 (75th
    percentile). Useful for getting a rough sense of the distribution
    (e.g. the basis for boxplots).
    """
    array = to_array(data)
    q1, q2, q3 = np.percentile(array, [25, 50, 75])
    return float(q1), float(q2), float(q3)


def iqr(data) -> float:
    """
    Interquartile range (IQR) = Q3 - Q1: the span in which the "middle"
    50% of the data lies. Commonly used for outlier detection: values
    outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are usually considered outliers.
    More robust than the range, since extreme individual values barely
    affect it.
    """
    q1, _, q3 = quartiles(data)
    return float(q3 - q1)


# These description texts are deliberately kept in German and separate
# from the calculations above: they are the text ultimately shown to the
# (German-speaking) end user inside the app, whereas the code itself
# (names, docstrings, comments) is in English for development purposes.
DESCRIPTIONS = {
    "mean": "Der Durchschnitt aller Werte. Empfindlich gegenueber Ausreissern.",
    "median": "Der mittlere Wert der sortierten Daten. Robust gegenueber Ausreissern.",
    "mode": "Der/die haeufigste(n) Wert(e) im Datensatz.",
    "std_dev": "Wie stark die Werte im Schnitt um den Mittelwert streuen.",
    "range": "Abstand zwischen groesstem und kleinstem Wert.",
    "iqr": "Spanne der mittleren 50% der Daten - robustes Streuungsmass.",
    "q1": "25%-Quantil: ein Viertel der Werte liegt darunter.",
    "q3": "75%-Quantil: drei Viertel der Werte liegen darunter.",
}


def summary_statistics(data) -> dict:
    """
    Bundles all descriptive metrics together with an explanation for each.
    This is the function the chatbot will call later, e.g. when a user
    asks "give me an overview of the 'age' column".
    """
    q1, q2, q3 = quartiles(data)
    values = {
        "mean": mean(data),
        "median": median(data),
        "mode": mode(data),
        "std_dev": std_dev(data),
        "range": value_range(data),
        "iqr": iqr(data),
        "q1": q1,
        "q3": q3,
    }
    return {
        name: {"value": value, "description": DESCRIPTIONS.get(name, "")}
        for name, value in values.items()
    }
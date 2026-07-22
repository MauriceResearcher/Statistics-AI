"""
Shared helper functions used across the statistics modules (descriptive,
inferential, regression). Centralizing this avoids repeating the same
input-validation and p-value-interpretation logic in every single function.
"""

from __future__ import annotations
import numpy as np


def to_array(data, min_length: int = 1) -> np.ndarray:
    """
    Converts input data into a 1D numpy array and validates it.

    Raises ValueError (with a German message, since this can bubble up
    directly to the end user in the app) if the data is empty, shorter
    than min_length, or contains missing values (NaN).
    """
    array = np.asarray(data, dtype=float)
    if array.size == 0:
        raise ValueError("Die Datenliste ist leer - es kann keine Statistik berechnet werden.")
    if array.size < min_length:
        raise ValueError(
            f"Es werden mindestens {min_length} Werte benoetigt, aber nur {array.size} wurden uebergeben."
        )
    if np.isnan(array).any():
        raise ValueError("Die Daten enthalten fehlende Werte (NaN). Bitte vorher bereinigen.")
    return array


def interpret_p_value(p_value: float, alpha: float = 0.05) -> str:
    """
    Turns a raw p-value into a plain-language interpretation. This text is
    meant to be shown directly to the end user in the app, so it stays
    German regardless of the language used for the surrounding code.
    """
    if p_value < alpha:
        return (
            f"p-Wert ({p_value:.4f}) liegt unter dem Signifikanzniveau ({alpha}) - "
            f"die Nullhypothese wird verworfen, das Ergebnis ist statistisch signifikant."
        )
    return (
        f"p-Wert ({p_value:.4f}) liegt nicht unter dem Signifikanzniveau ({alpha}) - "
        f"die Nullhypothese kann nicht verworfen werden, es gibt keinen ausreichenden Beleg fuer einen Effekt."
    )
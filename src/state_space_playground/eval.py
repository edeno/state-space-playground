"""Evaluation helpers for state-space models.

Deliberately minimal — most model-specific evaluation lives in the
notebooks themselves. This module only contains small shared helpers
that multiple notebooks would otherwise reimplement.
"""

from __future__ import annotations

import numpy as np


def tracking_error_cm(
    true_xy: np.ndarray,
    decoded_xy: np.ndarray,
) -> np.ndarray:
    """Per-bin Euclidean tracking error in cm.

    Parameters
    ----------
    true_xy : np.ndarray, shape (n_time, 2)
    decoded_xy : np.ndarray, shape (n_time, 2)

    Returns
    -------
    np.ndarray, shape (n_time,)
        ``‖true - decoded‖`` per time bin.
    """
    true_xy = np.asarray(true_xy)
    decoded_xy = np.asarray(decoded_xy)
    if true_xy.shape != decoded_xy.shape:
        raise ValueError(
            f"true_xy {true_xy.shape} and decoded_xy {decoded_xy.shape} "
            "must have the same shape."
        )
    if true_xy.ndim != 2 or true_xy.shape[1] != 2:
        raise ValueError(
            f"expected (n_time, 2) arrays, got {true_xy.shape}."
        )
    return np.linalg.norm(true_xy - decoded_xy, axis=1)


def tracking_error_summary(error_cm: np.ndarray) -> dict[str, float]:
    """Summary statistics for a tracking-error time series.

    Returns the median, 90th percentile, and fraction-within-20cm /
    fraction-within-10cm — the numbers we quote in decoder-comparison
    tables. Also returns ``n_finite`` / ``n_total`` so callers can see
    how much data was discarded (e.g., NaN bins from a Kalman filter
    before the first observation).

    Non-finite values are excluded from all statistics. If no finite
    values remain, all numeric stats are NaN.
    """
    raw = np.asarray(error_cm)
    finite = raw[np.isfinite(raw)]
    n_finite = int(finite.size)
    n_total = int(raw.size)
    if n_finite == 0:
        return {
            "median_cm": float("nan"),
            "p90_cm": float("nan"),
            "frac_within_20cm": float("nan"),
            "frac_within_10cm": float("nan"),
            "n_finite": n_finite,
            "n_total": n_total,
        }
    return {
        "median_cm": float(np.median(finite)),
        "p90_cm": float(np.percentile(finite, 90)),
        "frac_within_20cm": float(np.mean(finite < 20.0)),
        "frac_within_10cm": float(np.mean(finite < 10.0)),
        "n_finite": n_finite,
        "n_total": n_total,
    }

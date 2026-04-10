"""Shared plotting utilities for the state-space playground.

The centerpiece is ``plot_with_task_context``, which overlays
trial boundaries, reward events, and ripple spans on any time-series
axis. Every notebook uses it to make model outputs directly comparable
to behavioral/neural events.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_with_task_context(
    ax: plt.Axes,
    t: np.ndarray,
    y: np.ndarray,
    *,
    trials: Optional[pd.DataFrame] = None,
    ripple_times: Optional[pd.DataFrame] = None,
    pump_events: Optional[pd.DataFrame] = None,
    label: Optional[str] = None,
    color: str = "C0",
    lw: float = 1.0,
) -> None:
    """Plot ``y`` vs ``t`` with task-event context overlay.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes to draw on. Modified in place.
    t : np.ndarray, shape (n,)
        Time in seconds. Can be absolute or epoch-relative, but the
        overlaid event times (``trials``, ``ripple_times``, ``pump_events``)
        must be in the same frame.
    y : np.ndarray, shape (n,)
        Values to plot.
    trials : pd.DataFrame or None, optional
        If given, draws thin vertical lines at each trial's ``start_time``.
    ripple_times : pd.DataFrame or None, optional
        If given, shades the ``[start_time, end_time]`` span for each
        ripple in translucent orange.
    pump_events : pd.DataFrame or None, optional
        If given, draws a green downward triangle near the top of the
        axes at each pump ``start_time``.
    label : str or None, optional
    color : matplotlib color spec, default ``"C0"``
    lw : float, default 1.0

    Notes
    -----
    Overlay order is bottom-to-top: ripple spans (z=1) → trial lines
    (z=2) → main line (z=3) → reward markers (z=4). The main line is
    drawn *over* ripple spans so it remains readable.
    """
    # Ripple spans first so they sit behind everything else.
    if ripple_times is not None and len(ripple_times) > 0:
        for _, row in ripple_times.iterrows():
            ax.axvspan(
                row["start_time"],
                row["end_time"],
                color="#d95f02",
                alpha=0.15,
                zorder=1,
            )

    # Trial start lines.
    if trials is not None and len(trials) > 0:
        for tt in trials["start_time"]:
            ax.axvline(tt, color="k", lw=0.5, alpha=0.3, zorder=2)

    # Main series.
    ax.plot(t, y, color=color, lw=lw, label=label, zorder=3)

    # Reward markers via a blended transform (data-x, axes-fraction-y)
    # so the markers stay near the top of the axes even if a later
    # caller adds artists that expand the ylim.
    if pump_events is not None and len(pump_events) > 0:
        reward_times = pump_events["start_time"].to_numpy()
        ax.plot(
            reward_times,
            np.full_like(reward_times, 0.98, dtype=float),
            marker="v",
            color="g",
            markersize=4,
            linestyle="none",
            transform=ax.get_xaxis_transform(),
            zorder=4,
        )


def covariance_ellipse_points(
    mean: np.ndarray,
    cov: np.ndarray,
    n_std: float = 2.0,
    n_points: int = 64,
) -> np.ndarray:
    """Return ``(n_points, 2)`` points tracing an ``n_std`` covariance ellipse.

    Useful for drawing the posterior uncertainty of a 2D Gaussian
    position decoder. The ellipse is the level set
    ``(x - mean)^T Sigma^{-1} (x - mean) = n_std^2``.

    Parameters
    ----------
    mean : np.ndarray, shape (2,)
    cov : np.ndarray, shape (2, 2)
    n_std : float, default 2.0
        Number of standard deviations (Mahalanobis). 2.0 corresponds
        to a ~86% confidence region for a 2D Gaussian.
    n_points : int, default 64

    Returns
    -------
    np.ndarray, shape (n_points, 2)
    """
    # Eigendecomposition of the 2x2 covariance.
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Clip tiny negatives from numerical issues.
    eigvals = np.clip(eigvals, 0.0, None)
    radii = n_std * np.sqrt(eigvals)

    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (n, 2)
    # Scale by radii, then rotate by eigenvectors, then translate.
    scaled = circle * radii[None, :]
    rotated = scaled @ eigvecs.T
    return rotated + mean[None, :]

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook 2 — Time-varying place fields via `PlaceFieldModel`
#
# Fit `state_space_practice.PlaceFieldModel` on sorted CA1 pyramidal cells
# from **`j1620210710_.nwb / 02_r1`**. The model treats each unit's place
# field as a random walk on the weights of a 2D B-spline basis:
#
# $$ x_t = x_{t-1} + w_t, \quad y_t \sim \text{Poisson}(\exp(Z_t x_t) \Delta t) $$
#
# where the learned weights can drift across the session. Unlike classical
# occupancy-normalized rate maps, this gives both a mean place field and a
# time-resolved posterior with credible intervals on drift.
#
# **Precision.** The playground runs in float64 by default (via
# `pick_free_gpu()`). This is load-bearing — running in f32 produces
# silent NaN outputs from Kalman covariance losing PSD after enough bins.
# See `notebooks/02_STATUS.md` for the full investigation.
#
# **Fit architecture.** Each unit is fit as its own single-neuron model.
# `PlaceFieldModel` technically supports multi-neuron fitting via a
# block-diagonal design matrix, but that makes the Kalman covariance
# `(T, n_basis * n_neurons, n_basis * n_neurons)`, which for 305 units
# × 64 basis would be ~20k × 20k per time bin — terabytes of memory.
# Per-neuron fits are mathematically equivalent under block-diagonal
# independence and use ~1 GB each.
#
# **Pyramidal filter.** Interneurons fire at ~10–50 Hz and are not
# spatially tuned. Pyramidal / place cells fire at < 5 Hz with sharp
# spatial peaks. We filter to `rate < 5 Hz` and require ≥ 200 spikes
# in the session for usable fit quality.
#
# **No cross-validation** per the playground plan — fit on the full epoch,
# inspect results directly.

# %% [markdown]
# ## Setup — GPU pinning (which also enables f64) must come first

# %%
from state_space_playground.gpu import pick_free_gpu

pick_free_gpu(min_free_mb=20_000)

# %%
import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from state_space_practice.place_field_model import PlaceFieldModel

from state_space_playground.session import load_session

assert jax.config.jax_enable_x64, "f64 must be enabled — call pick_free_gpu() first"
print(f"jax dtype: {jnp.zeros(1).dtype}, devices: {jax.devices()}")

# %% [markdown]
# ## Configuration
#
# The notebook reads `NB2_N_CELLS` and `NB2_N_STEPS` from the environment
# so a smoke test can be run with smaller values without editing the
# notebook. Default values (the committed configuration) are 5 cells × 20
# SGD steps.

# %%
N_CELLS: int = int(os.environ.get("NB2_N_CELLS", "5"))
N_SGD_STEPS: int = int(os.environ.get("NB2_N_STEPS", "20"))
DT: float = 0.050  # 50 ms time bins -> T ≈ 28k over the 23.6 min epoch
N_INTERIOR_KNOTS: int = 5  # library default; 64 basis functions per unit
print(f"config: N_CELLS={N_CELLS}, N_SGD_STEPS={N_SGD_STEPS}, dt={DT}s, knots={N_INTERIOR_KNOTS}")

# %% [markdown]
# ## Load session and pick pyramidal candidates

# %%
data = load_session("j1620210710_.nwb", "02_r1", use_sorted_hpc=True)
position_info = data["position_info"]
spike_times_hpc: list[np.ndarray] = data["spike_times"]["HPC"]

duration_sec = float(position_info.index.max() - position_info.index.min())
rates_hz = np.array([len(u) / duration_sec for u in spike_times_hpc])
spike_counts = np.array([len(u) for u in spike_times_hpc])

# Pyramidal candidates: rate < 5 Hz AND at least 200 spikes for fit quality
is_pyr = (rates_hz < 5.0) & (spike_counts >= 200)
pyr_indices = np.where(is_pyr)[0]

print(f"Total HPC units:       {len(spike_times_hpc)}")
print(f"Pyramidal candidates:  {len(pyr_indices)} (rate<5 Hz, n>=200)")
print(f"  rate range:          {rates_hz[pyr_indices].min():.2f}"
      f" – {rates_hz[pyr_indices].max():.2f} Hz")
print(f"  spike count range:   {spike_counts[pyr_indices].min()}"
      f" – {spike_counts[pyr_indices].max()}")

# Rank pyramidal candidates by spike count (more spikes = better SNR)
rank_order = pyr_indices[np.argsort(-spike_counts[pyr_indices])]
selected_units = rank_order[:N_CELLS]
print(f"\nSelected top-{N_CELLS} pyramidal units (by spike count):")
for u in selected_units:
    print(f"  unit {u:3d}: {spike_counts[u]:5d} spikes, {rates_hz[u]:.2f} Hz")

# %% [markdown]
# ## Time binning
#
# Downsample position from its native rate (typically 500 Hz) to
# `1/DT` Hz. The animal moves at most ~0.5 cm per 50 ms bin at peak
# speed, much smaller than any place field, so start-of-bin position is
# an accurate approximation.
#
# The stride is computed from the actual native sampling rate of
# `position_info.index` rather than hard-coding a 500 Hz assumption,
# so the notebook doesn't silently misbin if a future `load_session`
# returns position at a different rate.

# %%
native_dt = float(np.median(np.diff(position_info.index.to_numpy()[:100])))
POSITION_STRIDE: int = int(round(DT / native_dt))
assert POSITION_STRIDE >= 1, (
    f"POSITION_STRIDE must be >=1; got {POSITION_STRIDE} "
    f"(native_dt={native_dt}, DT={DT})"
)
print(f"native position dt = {native_dt*1000:.2f} ms, stride = {POSITION_STRIDE}")

position_ds = position_info.iloc[::POSITION_STRIDE]
time_bins = position_ds.index.to_numpy()
xy = position_ds[["head_position_x", "head_position_y"]].to_numpy()
n_time = len(time_bins)

# Uniformity check: Spyglass position DataFrames can have sub-frame
# jitter at epoch boundaries. `PlaceFieldModel.bin_spike_times` and
# downstream filter code assume uniform bin spacing; verify that
# explicitly so a quietly non-uniform time_bins can't introduce subtle
# bias in the binning.
bin_diffs = np.diff(time_bins)
assert np.allclose(bin_diffs, DT, rtol=1e-3), (
    f"time_bins not uniform at dt={DT}: "
    f"min diff={bin_diffs.min():.6f}, max diff={bin_diffs.max():.6f}"
)

print(f"T = {n_time:,} time bins @ dt={DT*1000:.0f} ms "
      f"({n_time * DT / 60:.1f} min)")
print(f"arena: x ∈ [{xy[:, 0].min():.1f}, {xy[:, 0].max():.1f}],"
      f"        y ∈ [{xy[:, 1].min():.1f}, {xy[:, 1].max():.1f}]")

# %% [markdown]
# ## Occupancy map
#
# Context plot: where did the animal spend time? Place-field fits will
# only be meaningful where occupancy is nontrivial.

# %%
fig, ax = plt.subplots(figsize=(5, 4.5))
H, xe, ye = np.histogram2d(
    xy[:, 0],
    xy[:, 1],
    bins=60,
    range=[
        [xy[:, 0].min(), xy[:, 0].max()],
        [xy[:, 1].min(), xy[:, 1].max()],
    ],
)
occupancy_sec = H.T * DT
nonzero = occupancy_sec[occupancy_sec > 0]
im = ax.imshow(
    occupancy_sec,
    origin="lower",
    extent=(xe[0], xe[-1], ye[0], ye[-1]),
    aspect="equal",
    cmap="magma",
    vmax=float(np.percentile(nonzero, 99)) if len(nonzero) else None,
)
plt.colorbar(im, ax=ax, label="occupancy (s)")
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_title(f"Occupancy — {n_time * DT / 60:.1f} min, dt={int(DT * 1000)} ms")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Fit `PlaceFieldModel.fit_sgd` per unit
#
# Each unit is fit independently using library defaults:
#
# - `n_interior_knots=5` → 64 B-spline basis functions per unit
# - `init_cov_scale=0.01`, `init_process_noise=1e-6`, `max_firing_rate_hz=500`
# - `warm_start=True` runs a stationary Poisson GLM Laplace approximation
#   before SGD, giving a well-conditioned starting point
# - f64 precision throughout (enabled by `pick_free_gpu()`)
#
# We let SGD update all learnable parameters (the library defaults:
# `update_init_state=True`, `update_process_cov=True`,
# `update_transition_matrix=False`).

# %%
models: dict[int, PlaceFieldModel] = {}
ll_histories: dict[int, list[float]] = {}
fit_seconds: dict[int, float] = {}

for i, u in enumerate(selected_units):
    spikes_u = PlaceFieldModel.bin_spike_times(
        spike_times_hpc[u], time_bins, warn_on_drops=False
    )
    # Defensive contract assertion: bin_spike_times should return a 1D
    # array of length n_time for a single-neuron spike-time input. If
    # the upstream API ever changes to return (n_time, 1), the filter
    # would still run but the LL scale would be wrong; catch that here.
    assert spikes_u.ndim == 1 and len(spikes_u) == n_time, (
        f"unexpected spikes_u shape {spikes_u.shape}, expected ({n_time},)"
    )
    n_spk_binned = int(spikes_u.sum())  # count after downsampled binning

    model = PlaceFieldModel(
        dt=DT,
        n_interior_knots=N_INTERIOR_KNOTS,
    )
    t0 = time.time()
    lls = model.fit_sgd(
        position=xy,
        spikes=spikes_u,
        num_steps=N_SGD_STEPS,
        verbose=False,
    )
    elapsed = time.time() - t0

    models[int(u)] = model
    ll_histories[int(u)] = lls
    fit_seconds[int(u)] = elapsed

    ll_first = lls[0]
    ll_last = lls[-1]
    n_finite = int(np.sum(np.isfinite(np.asarray(lls))))
    print(
        f"[{i + 1:2d}/{N_CELLS}] unit {u:3d}  "
        f"{n_spk_binned:5d} binned spikes  "
        f"fit={elapsed:5.1f}s  "
        f"LL {ll_first:10.1f} -> {ll_last:10.1f}  "
        f"finite {n_finite}/{len(lls)}"
    )

print(f"\nTotal wall-clock: {sum(fit_seconds.values()):.1f}s for {N_CELLS} units")
print(f"Mean per-unit: {np.mean(list(fit_seconds.values())):.1f}s")

# %% [markdown]
# ## Diagnostic plots

# %% [markdown]
# ### SGD convergence curves
# Each unit's LL is normalized to [0, 1] within its own range so the
# shape of convergence can be compared across units regardless of
# absolute magnitude.

# %%
fig, ax = plt.subplots(figsize=(8, 4))
for u, lls in ll_histories.items():
    arr = np.asarray(lls, dtype=float)
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        norm = (arr - lo) / (hi - lo)
    else:
        norm = np.zeros_like(arr)
    ax.plot(norm, lw=1.0, alpha=0.7, label=f"u{u}")
ax.set_xlabel("SGD step")
ax.set_ylabel("normalized LL (each unit scaled to [0, 1])")
ax.set_title(f"SGD convergence — {N_CELLS} top-firing pyramidal units")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7, ncol=min(N_CELLS, 5), loc="lower right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Rate maps (time-averaged over the full session)
#
# For each unit we evaluate the smoothed place field on a 60×60 grid
# and average the smoothed weights across the full session. Color scale
# is per-panel since peak rates vary substantially between units.

# %%
N_GRID = 60
n_cols = min(N_CELLS, 5)
n_rows = int(np.ceil(N_CELLS / n_cols))
fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(n_cols * 2.6, n_rows * 2.4),
    sharex=True,
    sharey=True,
    squeeze=False,
)
axes_flat = axes.ravel()

for i, u in enumerate(selected_units):
    ax = axes_flat[i]
    model = models[int(u)]
    grid, xe_g, ye_g = model.make_grid(n_grid=N_GRID)
    rate, _ = model.predict_rate_map(grid)
    rate_2d = np.asarray(rate).reshape(len(ye_g), len(xe_g))
    im = ax.imshow(
        rate_2d,
        origin="lower",
        extent=(xe_g[0], xe_g[-1], ye_g[0], ye_g[-1]),
        aspect="equal",
        cmap="viridis",
    )
    n_spk = int(spike_counts[u])
    ax.set_title(
        f"u{u}  n={n_spk}  peak {float(rate.max()):.1f} Hz",
        fontsize=8,
    )
    ax.tick_params(labelsize=6)
    plt.colorbar(im, ax=ax, shrink=0.75)

for j in range(N_CELLS, len(axes_flat)):
    axes_flat[j].axis("off")

fig.suptitle(
    "CA1 place fields via PlaceFieldModel.fit_sgd — time-averaged",
    y=1.02,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Place field center drift across the session
#
# For each unit compute the weighted centroid of the rate map in 20
# temporal blocks. A tight cluster of centers means a stable field;
# a monotonic trajectory means drift; a wandering trajectory means
# the field is remapping or the signal is weak.

# %%
drift_summaries: dict[int, dict] = {}
for u in selected_units:
    drift_summaries[int(u)] = models[int(u)].drift_summary(
        n_grid=60, n_blocks=20
    )

fig, ax = plt.subplots(figsize=(6, 5.5))
# Background: light scatter of actual visited positions to show the
# track shape (this session is a plus-maze, not a rectangle).
# Subsample heavily so the background doesn't visually dominate.
bg_sample = xy[::50]
ax.scatter(
    bg_sample[:, 0],
    bg_sample[:, 1],
    color="lightgray",
    s=1,
    alpha=0.4,
    zorder=0,
)

colors = plt.cm.tab10(np.arange(N_CELLS))
for i, u in enumerate(selected_units):
    centers = drift_summaries[int(u)]["centers"]
    ax.plot(
        centers[:, 0],
        centers[:, 1],
        "-",
        color=colors[i],
        lw=1.0,
        alpha=0.7,
        label=f"u{u}",
    )
    ax.scatter(
        centers[0, 0],
        centers[0, 1],
        color=colors[i],
        s=40,
        marker="o",
        edgecolors="k",
        linewidths=0.5,
        zorder=5,
    )
    ax.scatter(
        centers[-1, 0],
        centers[-1, 1],
        color=colors[i],
        s=40,
        marker="X",
        edgecolors="k",
        linewidths=0.5,
        zorder=5,
    )

ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_title("Place field center drift — ○ first block, × last block")
ax.set_aspect("equal")
ax.legend(fontsize=7, loc="upper right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Drift summary table
#
# - `total_drift_cm`: straight-line distance from the first-block center
#   to the last-block center — net displacement
# - `cumulative_drift_cm`: path length of the full center trajectory —
#   captures back-and-forth wandering that net displacement misses
#
# `total_drift ≪ cumulative_drift` means the center wobbles but comes
# back; `total_drift ≈ cumulative_drift` means monotonic drift. Very small
# values (< ~5 cm) are within numerical noise.

# %%
drift_rows = []
for u in selected_units:
    ds = drift_summaries[int(u)]
    drift_rows.append(
        {
            "unit": int(u),
            "n_spikes": int(spike_counts[u]),
            "final_LL": float(ll_histories[int(u)][-1]),
            "total_drift_cm": float(ds["total_drift"]),
            "cumulative_drift_cm": float(ds["cumulative_drift"]),
            "peak_rate_Hz": float(ds["peak_rate_per_block"].max()),
            "fit_seconds": float(fit_seconds[int(u)]),
        }
    )
drift_df = pd.DataFrame(drift_rows).round(2)
drift_df

# %% [markdown]
# ### Detail plot for a single example unit
#
# Three panels: mean rate, upper 95% credible bound, and the **credible
# interval width** (`ci_hi − ci_lo`) — a direct "where is the field
# uncertain?" map. The posterior uncertainty is what distinguishes this
# model from a classical occupancy-normalized rate map. The CI-width
# panel concentrates high values at points with low occupancy (the model
# hasn't seen those positions much) and at field edges.

# %%
example_unit = int(selected_units[0])
model = models[example_unit]
grid, xe_g, ye_g = model.make_grid(n_grid=N_GRID)
rate, ci = model.predict_rate_map(grid, alpha=0.05)
rate_2d = np.asarray(rate).reshape(len(ye_g), len(xe_g))
ci_lo = np.asarray(ci[:, 0]).reshape(rate_2d.shape)
ci_hi = np.asarray(ci[:, 1]).reshape(rate_2d.shape)
ci_width = ci_hi - ci_lo

fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharex=True, sharey=True)

# Panel 1: mean rate
im0 = axes[0].imshow(
    rate_2d,
    origin="lower",
    extent=(xe_g[0], xe_g[-1], ye_g[0], ye_g[-1]),
    aspect="equal",
    cmap="viridis",
)
axes[0].set_title("mean rate (Hz)", fontsize=9)
plt.colorbar(im0, ax=axes[0], shrink=0.75)

# Panel 2: 97.5% upper bound — always >= mean, shares cmap scale for
# visual comparability with panel 1
im1 = axes[1].imshow(
    ci_hi,
    origin="lower",
    extent=(xe_g[0], xe_g[-1], ye_g[0], ye_g[-1]),
    aspect="equal",
    cmap="viridis",
    vmax=float(np.percentile(ci_hi, 99)),
)
axes[1].set_title("97.5% upper bound (Hz)", fontsize=9)
plt.colorbar(im1, ax=axes[1], shrink=0.75)

# Panel 3: credible-interval width — "uncertainty map"
im2 = axes[2].imshow(
    ci_width,
    origin="lower",
    extent=(xe_g[0], xe_g[-1], ye_g[0], ye_g[-1]),
    aspect="equal",
    cmap="magma",
    vmax=float(np.percentile(ci_width, 99)),
)
axes[2].set_title("95% CI width = hi − lo (Hz)", fontsize=9)
plt.colorbar(im2, ax=axes[2], shrink=0.75)

for ax in axes:
    ax.tick_params(labelsize=7)
fig.suptitle(
    f"unit {example_unit} — rate map with 95% credible interval",
    y=1.04,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Notes
#
# - **f64 is load-bearing.** Running this notebook under f32 produces
#   silent NaN outputs from the Kalman covariance losing PSD after ~245
#   bins at the default `diagonal_boost`. `pick_free_gpu()` enables f64
#   by default; the assert at the top of the imports cell will fail
#   loudly if someone accidentally disables it.
#
# - **Per-neuron fits vs multi-neuron block-diagonal.** For 305 units ×
#   64 basis functions, the multi-neuron state would be ~20,000
#   dimensional and the Kalman covariance `(T, 20000, 20000)` is
#   infeasible. Per-neuron fits are block-diagonally equivalent and
#   each costs ~1 GB of GPU memory in f64.
#
# - **Why top-N by spike count within pyramidal candidates?** Higher
#   spike count = higher SNR for place field estimation. A pyramidal
#   cell with 200 spikes will have a noisier rate map than one with
#   5,000. We sort within the pyramidal filter (`rate < 5 Hz`), not
#   across all units, so interneurons never enter the candidate pool.
#
# - **Scaling to all 155 pyramidal candidates.** This notebook's for-loop
#   trivially scales. At ~28 s per fit in f64, 155 cells is ~72 minutes.
#   The "real scaling win" would be `jax.vmap` over units to fit them in
#   parallel rather than sequentially, but that's an upstream change to
#   `state_space_practice` — out of scope here.
#
# - **Drift interpretation**: `total_drift ≪ cumulative_drift` indicates
#   the center wobbles but returns; `total_drift ≈ cumulative_drift`
#   indicates monotonic drift (possible remapping or tracking artifact).
#   Values below ~5 cm are within block-to-block noise of the centroid
#   estimate. Values ≫ typical place field width (30–40 cm in CA1) are
#   genuine signal.
#
# - **What this model buys you over classical rate maps.** Classical
#   occupancy-normalized rate maps give you "the best single rate map
#   averaged over the session". This model gives you (1) a principled
#   posterior over the weights, so `predict_rate_map` returns credible
#   intervals; (2) a time-resolved trajectory of the field via
#   `drift_summary` that can detect within-session drift; (3) a
#   log-likelihood you can compare across models. The cost is an order
#   of magnitude more compute than a classical rate map fit.
#
# Next: notebook 3 — position decoder comparison
# (`non_local_detector.SortedSpikesDecoder` vs
# `state_space_practice.PositionDecoder`) with a side-by-side 2D
# posterior movie.

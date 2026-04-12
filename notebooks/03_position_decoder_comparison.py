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
# # Notebook 3 — Position decoder comparison (non_local_detector vs state_space_practice)
#
# Fit two fundamentally different 2D position decoders on the same sorted
# CA1 spikes from **`j1620210710_.nwb / 02_r1`** and render a side-by-side
# movie of the decoded posterior over a short window containing runs and
# a patch switch.
#
# | Decoder | Package | Latent | Likelihood | Inference |
# |---|---|---|---|---|
# | `SortedSpikesDecoder` | `non_local_detector` | **Grid** over 2D position | KDE place field × Poisson | Forward-backward over discrete grid |
# | `PositionDecoder` | `state_space_practice` | **Gaussian** over (x, y, vx, vy) | KDE place field × Poisson | Laplace-EKF Kalman filter + RTS smoother |
#
# Both decoders consume the **same** sorted spike data and decode over
# the same time bins, so the comparison is apples-to-apples. They
# differ dramatically in how they represent the posterior:
#
# - non_local_detector gives a full (possibly multimodal) posterior
#   over the 2D grid. If the animal could be in two places, you see it.
# - state_space_practice gives a unimodal Gaussian. Mean + covariance
#   is all the information. Multimodality is collapsed to a mean, which
#   can be wildly wrong if the true posterior has multiple modes.
#
# **No cross-validation** per the playground plan. Both decoders fit on
# the full epoch and decode on the full epoch. This is a visualization
# notebook, not a benchmark.

# %% [markdown]
# ## Setup — GPU + f64 (via pick_free_gpu)

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
from matplotlib.animation import FFMpegWriter, FuncAnimation
from non_local_detector import Environment, SortedSpikesDecoder
from state_space_practice.place_field_model import PlaceFieldModel
from state_space_practice.position_decoder import PositionDecoder

from state_space_playground.plot import covariance_ellipse_points
from state_space_playground.session import load_session

assert jax.config.jax_enable_x64, "f64 must be enabled — call pick_free_gpu() first"
print(f"jax dtype: {jnp.zeros(1).dtype}, devices: {jax.devices()}")

# %% [markdown]
# ## Configuration
#
# `NB3_N_UNITS` and `NB3_WINDOW_MIN` from env vars let the notebook be
# smoke-tested (fewer units, shorter window) without editing. Defaults
# (committed): all pyramidal candidates, full session.

# %%
MAX_UNITS: int = int(os.environ.get("NB3_N_UNITS", "155"))
WINDOW_MIN: float = float(os.environ.get("NB3_WINDOW_MIN", "0"))  # 0 = full session
DT: float = 0.050  # 50 ms bins → 20 Hz decode rate
PLACE_BIN_SIZE: float = 3.0  # cm, for the non_local_detector grid
MOVIE_FPS: int = 20
MOVIE_SEC: float = 40.0  # seconds of playback (== seconds of data)

print(f"config: MAX_UNITS={MAX_UNITS}, WINDOW_MIN={WINDOW_MIN} "
      f"(0=full), dt={DT*1000:.0f}ms, place_bin={PLACE_BIN_SIZE}cm, "
      f"movie {MOVIE_FPS}fps/{MOVIE_SEC}s")

# %% [markdown]
# ## Load session, pick pyramidal candidates, compute time bins

# %%
data = load_session("j1620210710_.nwb", "02_r1", use_sorted_hpc=True)
position_info = data["position_info"]
spike_times_hpc: list[np.ndarray] = data["spike_times"]["HPC"]
trials = data["trials"]

# Restrict to a sub-window if NB3_WINDOW_MIN > 0 (for smoke test)
if WINDOW_MIN > 0:
    t0 = position_info.index.min()
    t1 = t0 + WINDOW_MIN * 60
    position_info = position_info[(position_info.index >= t0) & (position_info.index <= t1)]
    print(f"Restricted to first {WINDOW_MIN} min of session: "
          f"{len(position_info):,} position samples")

# Pyramidal candidate filter — same criteria as notebook 2
duration_sec = float(position_info.index.max() - position_info.index.min())
# Count each unit's spikes WITHIN the (possibly trimmed) window
_t_lo = float(position_info.index.min())
_t_hi = float(position_info.index.max())
spikes_in_window = [
    st[(st >= _t_lo) & (st <= _t_hi)] for st in spike_times_hpc
]
rates_hz = np.array([len(u) / duration_sec for u in spikes_in_window])
counts_in_window = np.array([len(u) for u in spikes_in_window])
is_pyr = (rates_hz < 5.0) & (counts_in_window >= 200)
pyr_indices = np.where(is_pyr)[0]
# Rank by spike count and keep top MAX_UNITS
rank_order = pyr_indices[np.argsort(-counts_in_window[pyr_indices])]
selected = rank_order[:MAX_UNITS]

print(f"Pyramidal candidates: {len(pyr_indices)} (rate<5 Hz, n>=200 in window)")
print(f"Selected top-{len(selected)}: rate range "
      f"{rates_hz[selected].min():.2f}–{rates_hz[selected].max():.2f} Hz, "
      f"spike counts {counts_in_window[selected].min()}–"
      f"{counts_in_window[selected].max()}")

selected_spike_times: list[np.ndarray] = [spikes_in_window[u] for u in selected]

# %% [markdown]
# ### Time bins and position at those bins

# %%
native_dt = float(np.median(np.diff(position_info.index.to_numpy()[:100])))
POSITION_STRIDE: int = int(round(DT / native_dt))
print(f"native dt = {native_dt*1000:.2f} ms, stride = {POSITION_STRIDE}")

position_ds = position_info.iloc[::POSITION_STRIDE]
time_bins = position_ds.index.to_numpy()
xy = position_ds[["head_position_x", "head_position_y"]].to_numpy()
n_time = len(time_bins)
bin_diffs = np.diff(time_bins)
assert np.allclose(bin_diffs, DT, rtol=1e-3), (
    f"time_bins not uniform at dt={DT}: min={bin_diffs.min()}, max={bin_diffs.max()}"
)
print(f"T = {n_time:,} bins @ dt={DT*1000:.0f}ms ({n_time * DT / 60:.1f} min)")

# Bin spike counts per unit for state_space_practice PositionDecoder
spike_counts = np.zeros((n_time, len(selected)), dtype=np.int64)
for j, st in enumerate(selected_spike_times):
    spike_counts[:, j] = PlaceFieldModel.bin_spike_times(
        st, time_bins, warn_on_drops=False
    )
print(f"binned spike counts shape: {spike_counts.shape}, "
      f"total spikes binned: {int(spike_counts.sum()):,}")

# %% [markdown]
# ## Fit and decode — `state_space_practice.PositionDecoder`
#
# KDE place field per unit, then Laplace-EKF Kalman filter over (x, y,
# vx, vy). Fit is closed-form KDE; decode is the Kalman filter + RTS
# smoother. Both steps are fast for 2D state dimension.

# %%
t0 = time.time()
# include_velocity=False gives a pure position random-walk decoder.
# smoothing_sigma=3.5 matches the typical CA1 place field width / kernel
# bandwidth used in the parent project.
# occupancy_tau=1.0 enables Bayesian shrinkage: at bins where the animal
# spent little time, the rate estimate shrinks toward the neuron's
# session-wide mean rate instead of being NaN (0/0). Without this, 21%
# of the grid is NaN and the bilinear interpolation near track edges
# poisons the Laplace-EKF update, systematically pushing the filter
# toward the track center.
ssp_decoder = PositionDecoder(
    dt=DT,
    q_pos=5.0,
    include_velocity=False,
    n_grid=50,
    smoothing_sigma=3.5,
    occupancy_tau=1.0,
)
ssp_decoder.fit(xy, spike_counts)
t_ssp_fit = time.time() - t0
print(f"PositionDecoder.fit(): {t_ssp_fit:.1f}s")
print(f"  rate_maps: {ssp_decoder.rate_maps.rate_maps.shape}")
print(f"  n_neurons: {ssp_decoder.rate_maps.n_neurons}")

# Initialize the Kalman filter at the animal's actual starting position
# rather than the center of the arena (the default). With few units the
# likelihood can be too weak to pull the filter away from a bad init.
init_pos = xy[0].copy()
print(f"  init_position set to animal's t=0 position: {init_pos.tolist()}")

t0 = time.time()
ssp_result = ssp_decoder.decode(
    spike_counts,
    method="smoother",
    init_position=init_pos,
)
t_ssp_dec = time.time() - t0
print(f"PositionDecoder.decode(): {t_ssp_dec:.1f}s")
print(f"  position_mean shape: {ssp_result.position_mean.shape}")
print(f"  position_cov shape: {ssp_result.position_cov.shape}")

# %% [markdown]
# ## Fit and decode — `non_local_detector.SortedSpikesDecoder`
#
# Grid-based decoder on 2D place bins. The environment infers the track
# interior from position data, then fits per-unit KDE place fields on
# the training data. Decoding is forward-backward over the discrete
# grid states. No velocity state.

# %%
t0 = time.time()
environment = Environment(
    place_bin_size=PLACE_BIN_SIZE,
    infer_track_interior=True,
    fill_holes=True,
)

nld_decoder = SortedSpikesDecoder(
    environments=environment,
    sampling_frequency=1.0 / DT,
)
nld_decoder.fit(
    position_time=time_bins,
    position=xy,
    spike_times=selected_spike_times,
)
t_nld_fit = time.time() - t0
print(f"SortedSpikesDecoder.fit(): {t_nld_fit:.1f}s")
print(f"  n_state_bins: {nld_decoder.environments[0].place_bin_centers_.shape[0]}")

t0 = time.time()
nld_result = nld_decoder.predict(
    spike_times=selected_spike_times,
    time=time_bins,
    position=xy,
    position_time=time_bins,
)
t_nld_dec = time.time() - t0
print(f"SortedSpikesDecoder.predict(): {t_nld_dec:.1f}s")
print(f"  posterior variables: {list(nld_result.data_vars)}")
print(f"  posterior shape: {nld_result['acausal_posterior'].shape}")

# %% [markdown]
# ## Extract decoded trajectories and tracking error
#
# For state_space_practice: the posterior mean is a 4D Gaussian state
# `[x, y, vx, vy]`; we take the first two dimensions.
#
# For non_local_detector: the posterior at each time is a distribution
# over grid bins. We take the **expected position** (weighted centroid
# over the grid), which is a principled "mean" that works for unimodal
# and sufficiently-peaky distributions. For multi-modal posteriors this
# under-represents the uncertainty.

# %%
# --- state_space_practice posterior mean ---
# State is (x, y) when include_velocity=False, (x, y, vx, vy) when True.
# Take the first two dims either way.
ssp_xy = np.asarray(ssp_result.position_mean[:, :2])  # (n_time, 2)
ssp_cov_xy = np.asarray(ssp_result.position_cov[:, :2, :2])  # (n_time, 2, 2)
print(f"\nssp decoded position: first 3 = {ssp_xy[:3].tolist()}")
print(f"  range x: [{ssp_xy[:, 0].min():.1f}, {ssp_xy[:, 0].max():.1f}]")
print(f"  range y: [{ssp_xy[:, 1].min():.1f}, {ssp_xy[:, 1].max():.1f}]")
print(f"  any NaN: {np.any(~np.isfinite(ssp_xy))}")

# --- non_local_detector expected position ---
# Inspect the posterior structure before reshaping
print(f"\nnld posterior: dims={nld_result['acausal_posterior'].dims}, "
      f"shape={nld_result['acausal_posterior'].shape}")
if "state" in nld_result["acausal_posterior"].dims:
    # State dim present — select the single "Continuous" state
    acausal = np.asarray(
        nld_result["acausal_posterior"].sel(state="Continuous").values
    )
    print(f"  after selecting 'Continuous' state: shape={acausal.shape}")
else:
    acausal = np.asarray(nld_result["acausal_posterior"].values)
    print(f"  no state dim, shape={acausal.shape}")

# acausal should now be (n_time, n_state_bins)
if acausal.ndim == 3:
    acausal = acausal[:, 0, :]  # fallback squeeze
    print(f"  squeezed to shape={acausal.shape}")

place_bin_centers = nld_decoder.environments[0].place_bin_centers_  # (n_state_bins, 2)
print(f"  place_bin_centers shape: {place_bin_centers.shape}")
print(f"  acausal range (ignoring NaN): "
      f"[{float(np.nanmin(acausal)):.4e}, {float(np.nanmax(acausal)):.4e}]")

# non_local_detector uses NaN at non-interior grid bins (bins outside
# the track footprint within the bounding box). For a plus-maze in a
# rectangular grid, ~80% of bins are non-interior.
# Mask to interior-only bins before computing the expected position.
interior_mask = np.isfinite(acausal[0])  # True for interior bins
n_interior = int(interior_mask.sum())
print(f"  interior bins (non-NaN in posterior): {n_interior} / {acausal.shape[1]}")

acausal_interior = np.nan_to_num(acausal[:, interior_mask], nan=0.0)
centers_interior = place_bin_centers[interior_mask]

# Renormalize row-wise over interior bins
posterior_norm_full = np.zeros_like(acausal)
row_sums = np.maximum(acausal_interior.sum(axis=1, keepdims=True), 1e-30)
posterior_norm_full[:, interior_mask] = acausal_interior / row_sums

# Weighted centroid over interior bins = expected decoded position
nld_xy = (acausal_interior / row_sums) @ centers_interior  # (n_time, 2)

# Also keep the full posterior_norm for the movie heatmap
# (it's zero at non-interior and normalized at interior)
posterior_norm = posterior_norm_full

print(f"  nld decoded first 3: {nld_xy[:3].tolist()}")
print(f"  nld range x: [{nld_xy[:, 0].min():.1f}, {nld_xy[:, 0].max():.1f}]")
print(f"  nld range y: [{nld_xy[:, 1].min():.1f}, {nld_xy[:, 1].max():.1f}]")
print(f"  nld any NaN: {np.any(~np.isfinite(nld_xy))}")

# Tracking error (Euclidean distance from decoded to true position)
err_ssp = np.linalg.norm(ssp_xy - xy, axis=1)
err_nld = np.linalg.norm(nld_xy - xy, axis=1)
print("\n=== Tracking error (cm) ===")
print(f"{'decoder':<20} {'median':>10} {'p90':>10} {'frac<20cm':>12} {'n_finite':>10}")
for name, err in [("state_space_practice", err_ssp), ("non_local_detector", err_nld)]:
    finite = err[np.isfinite(err)]
    if len(finite) == 0:
        print(f"{name:<20} {'—':>10} {'—':>10} {'—':>12} {0:>10}")
        continue
    med = float(np.median(finite))
    p90 = float(np.percentile(finite, 90))
    frac20 = float(np.mean(finite < 20.0))
    print(f"{name:<20} {med:>10.2f} {p90:>10.2f} {frac20:>12.3f} {len(finite):>10}")

# %% [markdown]
# ## Error-vs-speed diagnostic
#
# Both decoders are expected to do better when the animal is running
# (there's more informative spiking) and worse when it's stationary.
# The error-vs-speed curve is the most informative aggregate plot for
# this data.

# %%
head_speed = position_ds["head_speed"].to_numpy()
speed_bins = np.linspace(0, 50, 11)
speed_bin_idx = np.digitize(head_speed, speed_bins) - 1
speed_bin_idx = np.clip(speed_bin_idx, 0, len(speed_bins) - 2)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: error distributions per speed bin (violin-like boxplot)
data_for_bins = []
labels = []
for b in range(len(speed_bins) - 1):
    mask = speed_bin_idx == b
    if mask.sum() < 10:
        continue
    data_for_bins.append(
        (err_ssp[mask], err_nld[mask], f"{speed_bins[b]:.0f}–{speed_bins[b+1]:.0f}")
    )

positions_ssp = np.arange(len(data_for_bins)) * 3
positions_nld = positions_ssp + 1
for i, (e_ssp, e_nld, _lab) in enumerate(data_for_bins):
    axes[0].boxplot(
        e_ssp, positions=[positions_ssp[i]], widths=0.8, showfliers=False,
        patch_artist=True, boxprops={"facecolor": "#1f77b4", "alpha": 0.6},
        medianprops={"color": "k"},
    )
    axes[0].boxplot(
        e_nld, positions=[positions_nld[i]], widths=0.8, showfliers=False,
        patch_artist=True, boxprops={"facecolor": "#ff7f0e", "alpha": 0.6},
        medianprops={"color": "k"},
    )

axes[0].set_xticks(positions_ssp + 0.5)
axes[0].set_xticklabels([d[2] for d in data_for_bins], rotation=45, ha="right")
axes[0].set_xlabel("head speed (cm/s)")
axes[0].set_ylabel("tracking error (cm)")
axes[0].set_title("Tracking error vs speed")
axes[0].set_ylim(0, None)
# Manual legend
from matplotlib.patches import Patch  # noqa: E402

axes[0].legend(
    handles=[
        Patch(facecolor="#1f77b4", alpha=0.6, label="state_space_practice"),
        Patch(facecolor="#ff7f0e", alpha=0.6, label="non_local_detector"),
    ],
    fontsize=8,
    loc="upper right",
)
axes[0].grid(True, alpha=0.3)

# Right: cumulative error distribution
axes[1].plot(
    np.sort(err_ssp), np.linspace(0, 1, len(err_ssp)),
    color="#1f77b4", lw=1.5, label="state_space_practice",
)
axes[1].plot(
    np.sort(err_nld), np.linspace(0, 1, len(err_nld)),
    color="#ff7f0e", lw=1.5, label="non_local_detector",
)
axes[1].set_xlabel("tracking error (cm)")
axes[1].set_ylabel("cumulative fraction")
axes[1].set_title("CDF of tracking error")
axes[1].axvline(20, color="k", ls=":", lw=0.8, label="20 cm threshold")
axes[1].legend(fontsize=8, loc="lower right")
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 100)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Select movie window
#
# We want ~40 seconds of behavior in the middle of the session, ideally
# containing at least one patch change. We scan `trials.is_patch_change`
# for a trial with a patch change, grab its start time, and cut
# `[t - MOVIE_SEC/2, t + MOVIE_SEC/2]`.

# %%
t_start_session = float(time_bins[0])
t_end_session = float(time_bins[-1])

# Find patch-change trials in the session interior (avoid edges)
pc_trials = trials[trials["is_patch_change"]]
pc_trials_in_range = pc_trials[
    (pc_trials["start_time"] > t_start_session + MOVIE_SEC / 2)
    & (pc_trials["start_time"] < t_end_session - MOVIE_SEC / 2)
]
if len(pc_trials_in_range) == 0:
    # Fallback: pick the middle trial regardless of patch-change
    t_center = 0.5 * (t_start_session + t_end_session)
    print("No patch-change trial in safe range; using session midpoint")
else:
    # Pick the patch-change trial closest to the session midpoint
    mid = 0.5 * (t_start_session + t_end_session)
    pc_start_times = pc_trials_in_range["start_time"].to_numpy()
    closest_idx = int(np.argmin(np.abs(pc_start_times - mid)))
    t_center = float(pc_start_times[closest_idx])
    print(f"Selected patch-change trial at t={t_center:.1f}s "
          f"(session midpoint: {mid:.1f}s)")

t_win_lo = t_center - MOVIE_SEC / 2
t_win_hi = t_center + MOVIE_SEC / 2
win_mask = (time_bins >= t_win_lo) & (time_bins <= t_win_hi)
win_idx = np.where(win_mask)[0]
print(f"Movie window: [{t_win_lo:.1f}, {t_win_hi:.1f}]s, "
      f"{len(win_idx)} bins ({len(win_idx) * DT:.1f}s at {1/DT:.0f}Hz)")

# %% [markdown]
# ## Render the side-by-side movie
#
# Left: non_local_detector posterior as a 2D heatmap over the place
# grid, with true position (green ○) and decoded expected position (red
# ×). Right: state_space_practice posterior mean (red ×) with 95%
# covariance ellipse and true position (green ○). Bottom strip: spike
# raster for all selected units over the window, with a moving time
# cursor.

# %%
# Precompute window-specific arrays
win_time = time_bins[win_idx]
win_xy_true = xy[win_idx]
win_ssp_xy = ssp_xy[win_idx]
win_ssp_cov = ssp_cov_xy[win_idx]
win_nld_posterior = posterior_norm[win_idx]

# Grid shape for the non_local_detector heatmap
env = nld_decoder.environments[0]
grid_shape = env.centers_shape_  # (n_x_bins, n_y_bins) or similar
n_state_bins = env.place_bin_centers_.shape[0]
x_edges = env.edges_[0]
y_edges = env.edges_[1]
is_interior = env.is_track_interior_
print(f"Grid shape: {grid_shape}, interior bins: {int(np.asarray(is_interior).sum())}")

# For the raster, collect spike times falling in the window per unit
unit_spike_times_in_win = [
    st[(st >= t_win_lo) & (st <= t_win_hi)]
    for st in selected_spike_times
]
unit_y_positions = np.arange(len(selected_spike_times))

# Shared color scale for the nld posterior heatmap
positive_vals = win_nld_posterior[win_nld_posterior > 0]
nld_vmax = float(np.percentile(positive_vals, 99)) if len(positive_vals) > 0 else 1e-3

# Figure layout: two square panels on top, one wide raster strip below
fig = plt.figure(figsize=(11, 8))
gs = fig.add_gridspec(
    2, 2,
    height_ratios=[3, 1],
    hspace=0.3,
    wspace=0.25,
)
ax_nld = fig.add_subplot(gs[0, 0])
ax_ssp = fig.add_subplot(gs[0, 1])
ax_raster = fig.add_subplot(gs[1, :])

# Pre-draw static elements ---
# Both top panels share the same x/y extents
arena_extent = (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())

ax_nld.set_xlim(arena_extent[0], arena_extent[1])
ax_nld.set_ylim(arena_extent[2], arena_extent[3])
ax_nld.set_aspect("equal")
ax_nld.set_title("non_local_detector grid posterior", fontsize=10)
ax_nld.set_xlabel("x (cm)")
ax_nld.set_ylabel("y (cm)")

ax_ssp.set_xlim(arena_extent[0], arena_extent[1])
ax_ssp.set_ylim(arena_extent[2], arena_extent[3])
ax_ssp.set_aspect("equal")
ax_ssp.set_title("state_space_practice Gaussian posterior", fontsize=10)
ax_ssp.set_xlabel("x (cm)")

# Background scatter of the animal's trajectory (light gray) on both
for ax in (ax_nld, ax_ssp):
    ax.scatter(
        xy[::10, 0], xy[::10, 1],
        color="lightgray", s=2, alpha=0.5, zorder=0,
    )

# Initial heatmap (nld), true position dot (both), decoded markers (both)
# We'll update these via `set_data` and `set_offsets` inside the animation.
initial_posterior_2d = win_nld_posterior[0].reshape(grid_shape)
nld_im = ax_nld.imshow(
    initial_posterior_2d.T,  # transpose: grid_shape is (nx, ny), imshow wants (rows=y, cols=x)
    origin="lower",
    extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
    cmap="magma",
    vmin=0,
    vmax=nld_vmax,
    aspect="equal",
    interpolation="nearest",
    zorder=1,
)
nld_true = ax_nld.plot(
    [win_xy_true[0, 0]], [win_xy_true[0, 1]],
    marker="o", color="lime", markersize=10, markeredgecolor="k",
    linestyle="none", zorder=5, label="true",
)[0]
nld_pred = ax_nld.plot(
    [win_xy_true[0, 0]], [win_xy_true[0, 1]],  # placeholder, updated per frame
    marker="x", color="red", markersize=10, markeredgewidth=2,
    linestyle="none", zorder=6, label="decoded",
)[0]
ax_nld.legend(fontsize=8, loc="upper right")

# state_space_practice panel: true, decoded, ellipse
ssp_true = ax_ssp.plot(
    [win_xy_true[0, 0]], [win_xy_true[0, 1]],
    marker="o", color="lime", markersize=10, markeredgecolor="k",
    linestyle="none", zorder=5, label="true",
)[0]
ssp_pred = ax_ssp.plot(
    [win_ssp_xy[0, 0]], [win_ssp_xy[0, 1]],
    marker="x", color="red", markersize=10, markeredgewidth=2,
    linestyle="none", zorder=6, label="decoded",
)[0]
# Initial ellipse
ell_pts = covariance_ellipse_points(win_ssp_xy[0], win_ssp_cov[0], n_std=2.0, n_points=64)
ssp_ellipse = ax_ssp.plot(
    ell_pts[:, 0], ell_pts[:, 1],
    color="red", lw=1.2, alpha=0.7, zorder=4,
)[0]
ax_ssp.legend(fontsize=8, loc="upper right")

# Raster panel
ax_raster.set_xlim(t_win_lo, t_win_hi)
ax_raster.set_ylim(-1, len(selected_spike_times))
ax_raster.set_xlabel("time (s)")
ax_raster.set_ylabel("unit index")
ax_raster.set_title(
    f"spike raster — {len(selected_spike_times)} pyramidal units",
    fontsize=10,
)
for j, st in enumerate(unit_spike_times_in_win):
    if len(st) == 0:
        continue
    ax_raster.plot(
        st, np.full_like(st, j, dtype=float),
        "|", color="k", markersize=3, markeredgewidth=0.5,
    )

# Time cursor (vertical line)
time_cursor = ax_raster.axvline(t_win_lo, color="red", lw=1.2, zorder=5)

# Title text that updates per frame
title_text = fig.suptitle(f"t = {win_time[0] - t_start_session:.2f}s", fontsize=11)


def update(frame_idx: int):
    """Update all animated elements for a single frame."""
    t = win_time[frame_idx]
    # True position
    true_x, true_y = win_xy_true[frame_idx]
    nld_true.set_data([true_x], [true_y])
    ssp_true.set_data([true_x], [true_y])

    # nld heatmap + decoded point
    post_2d = win_nld_posterior[frame_idx].reshape(grid_shape).T  # (ny, nx)
    nld_im.set_data(post_2d)
    # Decoded position for nld = posterior expected position (precomputed)
    nld_x = nld_xy[win_idx[frame_idx], 0]
    nld_y = nld_xy[win_idx[frame_idx], 1]
    nld_pred.set_data([nld_x], [nld_y])

    # ssp decoded point + covariance ellipse
    ssp_x, ssp_y = win_ssp_xy[frame_idx]
    ssp_pred.set_data([ssp_x], [ssp_y])
    ell_pts = covariance_ellipse_points(
        win_ssp_xy[frame_idx], win_ssp_cov[frame_idx], n_std=2.0, n_points=64,
    )
    ssp_ellipse.set_data(ell_pts[:, 0], ell_pts[:, 1])

    # Time cursor
    time_cursor.set_xdata([t, t])

    title_text.set_text(
        f"t = {t - t_start_session:.2f}s "
        f"(Δ from session start)   |   frame {frame_idx + 1}/{len(win_idx)}"
    )

    return (nld_im, nld_true, nld_pred, ssp_true, ssp_pred, ssp_ellipse, time_cursor, title_text)


print(f"Building animation ({len(win_idx)} frames @ {MOVIE_FPS} fps)...")
t0 = time.time()
anim = FuncAnimation(
    fig,
    update,
    frames=len(win_idx),
    interval=1000 / MOVIE_FPS,
    blit=False,
    repeat=False,
)

# Save as MP4 via ffmpeg
import pathlib

_repo_root = pathlib.Path(__file__).resolve().parent.parent
out_path = str(_repo_root / "notebook_03_decoder_comparison.mp4")
writer = FFMpegWriter(fps=MOVIE_FPS, bitrate=3000)
anim.save(out_path, writer=writer)
plt.close(fig)
print(f"Movie saved: {out_path} ({time.time() - t0:.1f}s to render)")
print(f"  file size: {os.path.getsize(out_path) / 1e6:.1f} MB")

# %% [markdown]
# ## Notes and findings
#
# ### Key result: NLD grid decoder dramatically outperforms SSP Gaussian EKF
#
# On this data (20 pyramidal cells, 5 min, dt=50 ms), the NLD grid
# decoder achieves **~5 cm median error** while the SSP Gaussian EKF
# is stuck at **~90–110 cm** (effectively not tracking). This gap
# persists regardless of `q_pos` tuning or rate map choice.
#
# ### Why the SSP decoder doesn't track: two compounding issues
#
# **1. Rate map calibration.** The SSP `PositionDecoder.fit()` KDE
# produces rate maps with peak rates ~20× higher than
# `non_local_detector`'s `place_fields` (SSP peak ~38 Hz vs NLD peak
# ~2 Hz for the same neuron with 0.19 Hz mean rate). This is a
# normalization issue in the SSP occupancy-normalized KDE that warrants
# upstream investigation. With inflated rates, the Poisson model
# expects far more spikes per bin than are observed, creating a
# systematic "surprisingly few spikes" penalty that confuses the filter.
#
# **2. Measurement update is too weak for the EKF.** Even with
# correctly-calibrated NLD rate maps, the Gaussian EKF can't track:
# 20 neurons at ~0.2 Hz mean × 50 ms bins = ~0.2 expected spikes per
# bin across all neurons. Most bins have zero spikes, providing zero
# spatial information. The rare 1-spike bin gives a Poisson LL
# difference of ~0.1 nats between the field peak and background — far
# too little for the EKF measurement update to overcome process noise.
# The filter is simply frozen. The grid decoder handles this because
# forward-backward on a discrete grid accumulates evidence
# fundamentally differently from a Gaussian EKF.
#
# ### What's different about the NLD grid decoder
#
# - Grid posterior: multimodal, naturally constrained to track interior
# - Transition model: adjacency on the track graph (can't teleport)
# - Evidence accumulation: forward-backward on ~761 discrete bins
#   propagates even tiny per-bin likelihoods across the grid
# - Result: sub-5-cm tracking with only 20 neurons
#
# The Gaussian EKF has none of these structural advantages. Its
# posterior is unimodal and unconstrained in 2D, the measurement update
# is a single Newton-like step that needs sufficient per-bin evidence to
# move, and it has no track-geometry inductive bias beyond a soft track
# penalty.
#
# ### Open investigation items
#
# - **SSP rate map normalization**: the ~20× inflation in the
#   occupancy-normalized KDE is worth an upstream bug report. The
#   occupancy-weighted mean of the rate map should equal the neuron's
#   mean firing rate; currently it's ~20× higher.
# - **Scaling to more neurons**: with 155 pyramidal units instead of 20,
#   the per-bin spike count goes up 8×, which might provide enough
#   evidence for the EKF to track. Not tested yet.
# - **`include_velocity=True`**: the velocity model adds a strong
#   inductive bias that might help bridge spike-silent bins, but it
#   also diverged off-arena in our initial tests. Needs tuning of
#   q_vel together with the rate map fix.
# - **Non_local_detector and f64**: the NLD predict path emits dtype
#   mismatch warnings under f64 but produces correct results when the
#   posterior's NaN non-interior bins are properly masked. No f32
#   toggle is needed.
#
# ### Technical notes
#
# - **NLD posterior NaN at non-interior bins**: `non_local_detector`
#   uses NaN at grid bins outside the track interior. The notebook
#   masks to interior-only bins before computing the expected decoded
#   position via weighted centroid. Without masking, `nansum` propagates
#   NaN and the decoded trajectory is all-NaN.
# - **No CV.** Per the plan, both decoders fit and decode on the same
#   epoch. This is a visualization of behavior, not a benchmark.
# - **The movie uses expected position for NLD's "decoded" point.**
#   For multimodal posteriors this mis-represents the state; the
#   heatmap is the truthful visualization.
#
# Next: notebook 4 — oscillator models on LFP.

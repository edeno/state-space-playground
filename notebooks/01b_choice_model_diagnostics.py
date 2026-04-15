# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook 1b — Choice-model diagnostics
#
# Three follow-ups to notebook 01, all on the same 180-trial session
# `j1620210710_.nwb / 02_r1`:
#
# | § | Topic | Question |
# |---|---|---|
# | 1 | **Empirical anti-win-stay sanity check** | Notebook 01 model 3 found *negative* per-well win-stay weights (after a rewarded visit to well *w*, the animal is *less* likely to return). Is that a real data pattern or a model artifact? |
# | 2 | **Spatial covariates** | Notebook 01 model 3 used per-well one-hots, leaving 36 free entries in the perseveration matrix without an interpretable story. Does the off-diagonal structure collapse to a few interpretable spatial scalars (`is_same_stem`, `is_same_patch`, `is_alternate_leaf`)? |
# | 3 | **Switching + Frank-lab covariates** | The 1c switching model split the session into "explore (β≈0.5)" vs "exploit (β≈5)" regimes. Once we control for perseveration + win-stay, does the regime split survive — i.e., is there a real strategy switch, or was it just coding for switch cost? |
#
# Each section visualizes the raw data alongside the model fit and walks
# through the interpretation. We refit notebook 01's model 3 here so this
# notebook stands alone.

# %% [markdown]
# ## Setup

# %%
from state_space_playground.gpu import pick_free_gpu

pick_free_gpu(min_free_mb=20_000)

# %%
import logging

import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from state_space_practice.covariate_choice import CovariateChoiceModel
from state_space_practice.switching_choice import SwitchingChoiceModel

from state_space_playground.session import load_session

logging.basicConfig(level=logging.WARNING)

# %% [markdown]
# Load session and build the same per-trial arrays as notebook 01.

# %%
data = load_session("j1620210710_.nwb", "02_r1", use_sorted_hpc=True)
trials = data["trials"]
chosen_well: np.ndarray = trials["to_well"].to_numpy().astype(np.int32)
is_reward: np.ndarray = trials["is_reward"].to_numpy().astype(np.int32)
is_patch_change: np.ndarray = trials["is_patch_change"].to_numpy()
trial_num: np.ndarray = trials.index.to_numpy()
prev_reward: np.ndarray = np.concatenate([[0], is_reward[:-1]]).astype(np.float32)
prev_well_int = np.concatenate([[0], chosen_well[:-1]]).astype(np.int32)

n_trials = len(trials)
n_wells = 6

WELL_COLORS = {
    0: "#1b7837", 1: "#7fbf7b",  # patch 1 — green
    2: "#c51b7d", 3: "#de77ae",  # patch 2 — magenta
    4: "#2166ac", 5: "#92c5de",  # patch 3 — blue
}
WELL_PATCH = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3}
WELL_LEAF = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1}  # leaf within patch
patch_change_trials = trial_num[is_patch_change]


def _overlay_patch_changes(ax: plt.Axes) -> None:
    for tc in patch_change_trials:
        ax.axvline(tc, color="k", lw=0.5, alpha=0.3, ls="--", zorder=1)


print(f"n_trials = {n_trials}, n_rewards = {is_reward.sum()}, "
      f"n_patch_changes = {is_patch_change.sum()}")
print(f"well visits: {dict(enumerate(np.bincount(chosen_well, minlength=n_wells).tolist()))}")

# %% [markdown]
# ---
#
# # § 1. Empirical sanity check on anti-win-stay
#
# **Claim under test (from notebook 01 model 3):** the animal is *less*
# likely to return to a well after a rewarded visit than after an
# unrewarded one. The model said this with `winstay_diag` ≈ -1 to -1.4
# (logit units). Before reading anything into that, let's check the raw
# data.
#
# For each well *w*, count:
#
# - `n_visit[w]` = trials where well *w* was visited
# - `n_return[w | r]` = trials where (well *w* visited at *t*) AND
#   (well *w* visited at *t+1*) AND (reward at *t* = r)
# - empirical `P(return | r=1)` and `P(return | r=0)` at well *w*
#
# We also report the same broken down by "next trial is in the same patch
# vs. across patches" to see whether the apparent anti-win-stay is really
# about leaving the well or about leaving the *patch*.

# %%
def empirical_return_table(chosen_well: np.ndarray, is_reward: np.ndarray) -> pd.DataFrame:
    rows = []
    for w in range(n_wells):
        for r in (0, 1):
            mask_t = (chosen_well[:-1] == w) & (is_reward[:-1] == r)
            n_eligible = int(mask_t.sum())
            if n_eligible == 0:
                rows.append({"well": w, "patch": WELL_PATCH[w], "prev_reward": r,
                             "n_eligible": 0, "n_return_same_well": 0,
                             "n_return_same_patch": 0, "p_same_well": np.nan,
                             "p_same_patch": np.nan})
                continue
            next_well = chosen_well[1:][mask_t]
            n_return_same_well = int((next_well == w).sum())
            n_return_same_patch = int(((next_well // 2) == (w // 2)).sum())
            rows.append({
                "well": w, "patch": WELL_PATCH[w], "prev_reward": r,
                "n_eligible": n_eligible,
                "n_return_same_well": n_return_same_well,
                "n_return_same_patch": n_return_same_patch,
                "p_same_well": n_return_same_well / n_eligible,
                "p_same_patch": n_return_same_patch / n_eligible,
            })
    return pd.DataFrame(rows)


empirical_table = empirical_return_table(chosen_well, is_reward)
empirical_table.round(3)

# %% [markdown]
# **How to read the table.** Each row is "trials where well *w* was the
# *previous* choice and `prev_reward` was r". `p_same_well` is the
# fraction that returned to the *same well* immediately. `p_same_patch`
# includes returns to either well in the same patch.
#
# The bar plot below shows `p_same_well` for `prev_reward = 0` (gray)
# vs `prev_reward = 1` (colored) per well. **If model 3's negative
# win-stay is real, the rewarded bar should be lower than the
# unrewarded bar at every well.**

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
ax_well, ax_patch = axes

x = np.arange(n_wells)
width = 0.38
for ax, col, title in [
    (ax_well, "p_same_well", "P(return to same well | prev_reward)"),
    (ax_patch, "p_same_patch", "P(stay in same patch | prev_reward)"),
]:
    p_un = empirical_table.set_index(["well", "prev_reward"])[col].unstack().reindex(range(n_wells))
    ax.bar(x - width / 2, p_un[0].values, width, color="0.6",
           edgecolor="k", linewidth=0.5, label="prev_reward = 0")
    ax.bar(x + width / 2, p_un[1].values, width,
           color=[WELL_COLORS[w] for w in range(n_wells)],
           edgecolor="k", linewidth=0.5, label="prev_reward = 1")
    # annotate sample sizes
    eligible = empirical_table.set_index(["well", "prev_reward"])["n_eligible"].unstack().reindex(range(n_wells))
    for i in range(n_wells):
        ax.text(i - width / 2, -0.04, f"n={int(eligible[0].iloc[i])}", ha="center", fontsize=7, color="0.4")
        ax.text(i + width / 2, -0.04, f"n={int(eligible[1].iloc[i])}", ha="center", fontsize=7, color="0.4")
    ax.set_xticks(x)
    ax.set_xticklabels([f"w{w}\n(p{WELL_PATCH[w]})" for w in range(n_wells)])
    ax.set_ylim(-0.08, 1.0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("empirical probability")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
ax_well.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Reading the empirical bars.**
#
# - Look first at wells 4 and 5 — these have the largest `n_eligible`
#   (66 and 63 visits respectively) so the rates are statistically
#   informative.
# - Wells 0, 2, 3 have very few rewarded visits (≤3), so the
#   `prev_reward = 1` bars at those wells are essentially noise.
# - The model 3 claim is that **rewarded bar < unrewarded bar** for
#   `p_same_well` at every well. If you see the rewarded bar lower at
#   wells 4 and 5 (and weakly elsewhere), model 3's negative win-stay
#   reflects real data structure. If the rewarded bar is *higher* at
#   wells 4/5, the model is doing something funky and the negative
#   win-stay is absorbing structure from a missing covariate.
# - Compare `p_same_well` to `p_same_patch`. If the same-patch
#   probability is much higher than same-well even after reward, the
#   animal is alternating leaves *within* the patch — and a flat
#   per-well perseveration covariate can't express that. § 2 tests
#   this directly.

# %% [markdown]
# ## Trials-since-last-visit — a missing covariate?
#
# A depletion-driven anti-win-stay should *also* depend on how recently
# the animal visited the well. If a well has been left alone for 20
# trials, it should be replenished, so the anti-win-stay penalty should
# fade. Let's check.
#
# For each trial *t* where the animal chose well *w*, compute the gap
# in trials since the animal last visited *w* (∞ if never). Then plot
# the empirical visit rate (over all wells) as a function of this gap.

# %%
def trials_since_last_visit(chosen_well: np.ndarray, n_wells: int) -> np.ndarray:
    """For each (trial t, well w), how many trials since well w was last visited (inf if never)."""
    n_trials = chosen_well.shape[0]
    last_visit = np.full(n_wells, -np.inf)
    out = np.full((n_trials, n_wells), np.inf)
    for t in range(n_trials):
        out[t] = t - last_visit  # gap; will be inf for never-visited
        last_visit[chosen_well[t]] = t
    return out


tsv = trials_since_last_visit(chosen_well, n_wells)  # (T, W)
# For each *chosen* trial t, get tsv at the chosen well
chosen_tsv = np.array([tsv[t, chosen_well[t]] for t in range(n_trials)])
# Empirical CDF of "gap to revisit" — i.e., P(animal returns within k trials | left)
finite_gaps = chosen_tsv[np.isfinite(chosen_tsv)]
# Histogram of gap-since-last-visit at the moment of choice
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ax_hist, ax_rate = axes
ax_hist.hist(finite_gaps, bins=np.arange(0, 30) - 0.5, color="C0", edgecolor="k")
ax_hist.set_xlabel("trials since last visit (when re-chosen)")
ax_hist.set_ylabel("count of choices")
ax_hist.set_title(
    f"Distribution of revisit gaps\n"
    f"(median = {np.median(finite_gaps):.1f}, "
    f"frac immediate = {(finite_gaps == 1).mean():.2%})"
)
ax_hist.grid(True, alpha=0.3)

# Empirical P(visit | gap) — for each gap k, what fraction of (well, trial) pairs
# with last-visit gap = k actually got chosen?
gap_bins = np.arange(0, 30)
chosen_mask = np.zeros((n_trials, n_wells), dtype=bool)
chosen_mask[np.arange(n_trials), chosen_well] = True
p_visit_by_gap = np.zeros(len(gap_bins) - 1)
for i, lo in enumerate(gap_bins[:-1]):
    hi = gap_bins[i + 1]
    mask = (tsv >= lo) & (tsv < hi)
    if mask.sum() == 0:
        p_visit_by_gap[i] = np.nan
        continue
    p_visit_by_gap[i] = (mask & chosen_mask).sum() / mask.sum()

ax_rate.bar(gap_bins[:-1], p_visit_by_gap, color="C2", edgecolor="k")
ax_rate.axhline(1 / n_wells, color="k", lw=0.8, ls="--", label="chance (1/6)")
ax_rate.set_xlabel("trials since last visit")
ax_rate.set_ylabel("P(this well chosen)")
ax_rate.set_title("Visit rate by recency — depletion / refractoriness")
ax_rate.legend(fontsize=8)
ax_rate.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpretation.** If P(visit | gap=1) is below chance and P(visit |
# gap=k) climbs toward chance as k grows, that's classic depletion: the
# animal avoids just-visited wells and returns only after a recovery
# window. If P(visit | gap=1) is *near* chance, the apparent
# anti-perseveration in model 3 is actually a static spatial bias, not
# a depletion-time effect.
#
# This plot is the closest model-free analog of the next-step covariate
# we'd add to the model: a "trials-since-last-visit at this well"
# covariate would absorb depletion-time structure that the static per-
# well perseveration covariate currently bundles together.

# %% [markdown]
# ---
#
# # § 2. Spatial covariates — adding patch-aware features
#
# Notebook 01 model 3 fit a `(K=6, d=12)` `obs_weights_` matrix: 6
# columns of per-well perseveration + 6 columns of per-well win-stay.
# That's 72 obs weights total. The per-well baseline gives the model
# a separate scalar for every (this-well, last-well) pair — including
# off-diagonal entries like "after well 4, well 5 becomes more likely",
# which would be the smoking gun for **within-patch alternation**.
#
# But the per-well parameterization can't tell us *why* — is the off-
# diagonal at (5, 4) a property of "well 5 specifically loves it
# when well 4 was just visited", or is it the general task-level
# pattern "after any visit, the alternate leaf in the same patch is
# attractive"? To distinguish these we add **task-aware patch-level
# features** that name the spatial pattern explicitly:
#
# - `is_same_well`: 1 if the candidate well *equals* the previous well
# - `is_same_patch`: 1 if the candidate well is in the same patch as
#   the previous well (includes `is_same_well`)
# - `is_alternate_leaf_in_patch`: 1 if candidate is the *other* leaf
#   in the previous patch (i.e., same patch but different well)
# - `is_same_well × prev_reward`: per-trial win-stay
# - `is_same_patch × prev_reward`: per-trial "win-stay-in-patch"
#
# **Important parameter-count caveat.** The SSP API expects
# `obs_covariates` shape `(T, d_obs)` and learns `obs_weights_` shape
# `(K, d_obs)`. A truly scalar feature "alternate leaf in patch was
# just visited" depends on *which option you're scoring*, so it can't
# fit in a single column. We encode each spatial feature as a
# `(T, K)` indicator block where column *k* is 1 iff option *k*
# satisfies the feature on trial *t*. The model then learns
# `(K, K) = 36` weights per feature, of which only the **diagonal**
# (option == lit-for-option) carries the intended interpretation;
# the off-diagonals are extra capacity. Total: 5 features × 36 =
# **180 obs weights, more than the per-well baseline's 72**. So this
# isn't a parsimony exercise — it's a **structure-naming** exercise.
# The diagonals of each feature block answer the question "does this
# spatial pattern matter, and how strongly per-option?". A LL gain
# alone could just reflect the larger parameter budget.

# %%
def build_spatial_obs_covariates(
    chosen_well: np.ndarray, prev_reward: np.ndarray, n_wells: int
) -> tuple[np.ndarray, list[str]]:
    """Build per-trial spatial features as one-hot blocks over candidate wells.

    Each feature is a (T, n_wells) block: column `k` is 1 iff candidate
    option `k` satisfies the feature on trial `t` (given the previous
    well). Concatenated to (T, 5 * n_wells).

    Features:
      0. is_same_well       (1 if candidate == prev_well)
      1. is_same_patch      (1 if patch(candidate) == patch(prev_well))
      2. is_alternate_leaf  (same_patch AND NOT same_well)
      3. is_same_well * prev_reward
      4. is_same_patch * prev_reward
    """
    n_trials = chosen_well.shape[0]
    prev_well = np.concatenate([[0], chosen_well[:-1]])
    prev_well_oh = np.eye(n_wells, dtype=np.float32)[prev_well]  # (T, K)
    prev_patch = prev_well // 2
    same_patch = (np.arange(n_wells)[None, :] // 2 == prev_patch[:, None]).astype(np.float32)
    alternate_leaf = same_patch * (1 - prev_well_oh)
    pr = prev_reward[:, None].astype(np.float32)
    win_well = prev_well_oh * pr
    win_patch = same_patch * pr

    # Zero out trial 0 (no previous)
    has_prev = np.concatenate([[0.0], np.ones(n_trials - 1)])[:, None].astype(np.float32)
    blocks = [
        ("same_well", prev_well_oh * has_prev),
        ("same_patch", same_patch * has_prev),
        ("alternate_leaf", alternate_leaf * has_prev),
        ("win_same_well", win_well * has_prev),
        ("win_same_patch", win_patch * has_prev),
    ]
    obs = np.hstack([b for _, b in blocks])
    names = [n for n, _ in blocks]
    return obs.astype(np.float32), names


obs_spatial, spatial_names = build_spatial_obs_covariates(chosen_well, prev_reward, n_wells)
print(f"spatial obs shape: {obs_spatial.shape}  "
      f"(features: {spatial_names})")

# %% [markdown]
# ## Refit notebook 01's model 3 (per-well one-hot baseline)
#
# We need this for an apples-to-apples LL comparison. Same fit as in
# notebook 01 §3.

# %%
prev_well_oh_full = np.eye(n_wells, dtype=np.float32)[prev_well_int]
win_stay_oh = (prev_reward[:, None] * prev_well_oh_full).astype(np.float32)
obs_perwell = np.hstack([prev_well_oh_full, win_stay_oh]).astype(np.float32)

model_perwell = CovariateChoiceModel(
    n_options=n_wells, n_covariates=1, n_obs_covariates=obs_perwell.shape[1],
)
lls_perwell = model_perwell.fit_sgd(
    jnp.asarray(chosen_well),
    covariates=jnp.asarray(prev_reward[:, None]),
    obs_covariates=jnp.asarray(obs_perwell),
    num_steps=500,
    verbose=False,
)
print(f"per-well baseline   final LL: {lls_perwell[-1]:.2f}  "
      f"({obs_perwell.shape[1]} obs features → {obs_perwell.shape[1] * n_wells} obs params)")

# %% [markdown]
# ## Fit the spatial-covariate model

# %%
model_spatial = CovariateChoiceModel(
    n_options=n_wells, n_covariates=1, n_obs_covariates=obs_spatial.shape[1],
)
lls_spatial = model_spatial.fit_sgd(
    jnp.asarray(chosen_well),
    covariates=jnp.asarray(prev_reward[:, None]),
    obs_covariates=jnp.asarray(obs_spatial),
    num_steps=500,
    verbose=False,
)
print(f"spatial-covariate   final LL: {lls_spatial[-1]:.2f}  "
      f"({obs_spatial.shape[1]} obs features → {obs_spatial.shape[1] * n_wells} obs params)")
print(f"Δ = {lls_spatial[-1] - lls_perwell[-1]:+.2f} "
      f"({'spatial' if lls_spatial[-1] > lls_perwell[-1] else 'per-well'} wins)")

# %% [markdown]
# **Reading the LL gap.** Because the spatial parameterization has
# *more* obs weights (180 vs 72), a small positive `Δ` is expected
# from extra capacity alone. The interesting question is whether the
# **diagonal** weights (the interpretable per-feature, per-option
# scalars) match the per-well diagonals from notebook 01 model 3, and
# whether `alternate_leaf` and `win_same_patch` come out non-zero —
# those are spatial patterns the per-well baseline can encode only
# implicitly via off-diagonal entries.

# %% [markdown]
# ## Visualize the learned spatial weights
#
# `obs_weights_` for the spatial model is `(K=6, d_obs=30)` —
# 5 features × 6 columns each. Within each feature block, each row
# of `obs_weights_` is a per-option scalar weight on that feature.
# We plot one panel per feature: the per-option logit shift when the
# feature is 1 for that option.

# %%
W_sp = np.asarray(model_spatial.obs_weights_)  # (K=6, 30)
W_blocks = W_sp.reshape(n_wells, len(spatial_names), n_wells)  # (option, feature, lit_for_option)
# Per-feature, per-option weight: take the diagonal (option == lit_for_option)
# since each column lights up exactly one option per row.
diag_W = np.array([W_sp[:, i * n_wells:(i + 1) * n_wells].diagonal() for i in range(len(spatial_names))])
# Shape (n_features, n_options): row = feature, col = option

fig, axes = plt.subplots(1, len(spatial_names), figsize=(15, 3.5), sharey=True)
for i, (name, ax) in enumerate(zip(spatial_names, axes)):
    bars = ax.bar(
        range(n_wells), diag_W[i],
        color=[WELL_COLORS[w] for w in range(n_wells)],
        edgecolor="k", linewidth=0.5,
    )
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(n_wells))
    ax.set_xticklabels([f"w{w}" for w in range(n_wells)], fontsize=8)
    ax.set_title(name, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
axes[0].set_ylabel("logit shift")
fig.suptitle("Spatial-feature obs weights (per-option diag)", y=1.02, fontsize=11)
plt.tight_layout()
plt.show()

# %% [markdown]
# **How to interpret each panel.**
#
# - **`same_well`**: the perseveration effect on each candidate well
#   *when it equals the previously-chosen well*. Negative = depletion.
#   Should look like a flatter version of model 3's per-well
#   perseveration diagonal.
# - **`same_patch`**: the bias toward each well *when its patch was
#   just visited*. Combines stay-on-stem and within-patch alternation.
# - **`alternate_leaf`**: the bias toward visiting the *other* leaf of
#   the previous patch. **Positive values here are the smoking gun for
#   within-patch alternation** — the animal stays on the same stem but
#   flips to the other well. This is structure the per-well baseline
#   could express only by encoding 6 separate off-diagonal entries.
# - **`win_same_well`**: extra logit shift when the previous well was
#   *both* the same as the candidate AND rewarded. Negative = the
#   anti-win-stay finding from notebook 01 model 3, now in a single
#   per-well scalar instead of 6 entries.
# - **`win_same_patch`**: extra logit shift when the candidate is in
#   the patch the animal was just rewarded in. If positive, "rewards
#   pull the animal toward the patch but not the well" (animal leaves
#   the depleted well but stays on the stem to try the alternate
#   leaf).

# %% [markdown]
# ## Compare to the per-well baseline diagonals

# %%
W_pw = np.asarray(model_perwell.obs_weights_)  # (K=6, 12)
persev_diag = np.diag(W_pw[:, :n_wells])
winstay_diag = np.diag(W_pw[:, n_wells:])

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(n_wells)
width = 0.18
ax.bar(x - 1.5 * width, persev_diag, width, label="per-well: same_well diag",
       color="0.5", edgecolor="k", linewidth=0.4)
ax.bar(x - 0.5 * width, diag_W[spatial_names.index("same_well")], width,
       label="spatial: same_well", color="0.7", edgecolor="k", linewidth=0.4)
ax.bar(x + 0.5 * width, winstay_diag, width, label="per-well: win_same_well diag",
       color="C3", edgecolor="k", linewidth=0.4, alpha=0.7)
ax.bar(x + 1.5 * width, diag_W[spatial_names.index("win_same_well")], width,
       label="spatial: win_same_well", color="C0", edgecolor="k", linewidth=0.4, alpha=0.7)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels([f"w{w}\n(p{WELL_PATCH[w]})" for w in range(n_wells)])
ax.set_ylabel("logit shift")
ax.set_title("Per-well baseline (notebook 01 model 3) vs spatial parameterization")
ax.legend(fontsize=7, loc="best", ncol=2)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Reading.** If the spatial bars closely track the per-well bars at
# every well, the spatial parameterization explains the per-well
# coefficients via the shared spatial structure. Differences between
# them point at well-specific anomalies — typically wells with very
# few visits (0, 2, 3) where the per-well estimate is noisy and the
# spatial estimate borrows from neighbors.

# %% [markdown]
# ---
#
# # § 3. Switching + Frank-lab covariates
#
# Notebook 01 model 1c found two regimes with very different inverse
# temperatures (β ≈ 0.5 vs β ≈ 5.3) and a sticky transition matrix
# (>0.99 self-transition for the high-β state). The interpretation
# was "explore vs exploit". But model 3 then showed that most of the
# raw-LL signal is anti-perseveration + slow latent drift, *not*
# value-based exploration vs exploitation.
#
# This section asks: **once we partial out perseveration and
# win-stay via obs covariates, does the regime split survive?**
#
# `SwitchingChoiceModel` accepts `obs_covariates` with shared weights
# across regimes (per the SSP API — the `Θ` matrix is `(K, d_obs)`,
# not `(S, K, d_obs)`). So the covariates can absorb the shared
# perseveration signal, leaving the regime-specific β / process-noise
# to fit whatever residual variation remains.

# %%
model_1c_plain = SwitchingChoiceModel(
    n_options=n_wells, n_discrete_states=2,
    init_inverse_temperatures=jnp.array([0.5, 5.0]),
)
lls_1c_plain = model_1c_plain.fit_sgd(jnp.asarray(chosen_well), num_steps=500, verbose=False)
post_1c_plain = np.asarray(model_1c_plain.smoothed_discrete_probs_)
print(
    f"1c plain  final LL: {lls_1c_plain[-1]:.2f}  "
    f"βs: {np.asarray(model_1c_plain.inverse_temperatures_).round(2)}, "
    f"process noise: {np.asarray(model_1c_plain.process_noises_).round(4)}"
)

# %%
model_1c_cov = SwitchingChoiceModel(
    n_options=n_wells, n_discrete_states=2, n_covariates=1, n_obs_covariates=12,
    init_inverse_temperatures=jnp.array([0.5, 5.0]),
)
lls_1c_cov = model_1c_cov.fit_sgd(
    jnp.asarray(chosen_well),
    covariates=jnp.asarray(prev_reward[:, None]),
    obs_covariates=jnp.asarray(obs_perwell),
    num_steps=500,
    verbose=False,
)
post_1c_cov = np.asarray(model_1c_cov.smoothed_discrete_probs_)
print(
    f"1c + cov  final LL: {lls_1c_cov[-1]:.2f}  "
    f"βs: {np.asarray(model_1c_cov.inverse_temperatures_).round(2)}, "
    f"process noise: {np.asarray(model_1c_cov.process_noises_).round(4)}"
)
print(
    f"1c + cov  transition matrix:\n"
    f"{np.asarray(model_1c_cov.discrete_transition_matrix_).round(3)}"
)

# %% [markdown]
# ## Visualize regime posteriors before vs after adding covariates
#
# Top: 1c plain (no covariates) — the original "explore vs exploit"
# result. Bottom: 1c + Frank-lab covariates. The reward raster sits
# below for context.
#
# **What to look for.**
#
# - If the bottom panel is **mostly one color** (one regime dominates),
#   the original regime split was a coding for the per-trial
#   perseveration signal that the covariates now express directly.
#   The "two strategies" interpretation collapses.
# - If the bottom panel **still shows two distinct regimes** with
#   transitions roughly aligned to the top panel, there's residual
#   strategy-like variation *beyond* perseveration. That would
#   support the original cognitive story (explore vs exploit) but
#   with the per-trial perseveration noise removed.
# - If the bottom panel shows **different transitions** than the top,
#   the covariates have "rotated" what the regime axis is coding for.
#   Read off `inverse_temperatures_` to see whether the new regimes
#   still differ in β-magnitude or have collapsed to similar values.

# %%
fig = plt.figure(figsize=(13, 8))
gs = gridspec.GridSpec(3, 1, height_ratios=[3, 3, 1.5], hspace=0.12, figure=fig)
ax_plain = fig.add_subplot(gs[0])
ax_cov = fig.add_subplot(gs[1], sharex=ax_plain)
ax_raster = fig.add_subplot(gs[2], sharex=ax_plain)

inv_plain = np.asarray(model_1c_plain.inverse_temperatures_)
inv_cov = np.asarray(model_1c_cov.inverse_temperatures_)

for ax, post, inv_temps, title in [
    (ax_plain, post_1c_plain, inv_plain, "1c plain — smoothed regime posterior"),
    (ax_cov, post_1c_cov, inv_cov, "1c + Frank-lab covariates — smoothed regime posterior"),
]:
    labels = [f"state {s} (β={inv_temps[s]:.2f})" for s in range(post.shape[1])]
    ax.stackplot(trial_num, post.T, labels=labels, alpha=0.8)
    _overlay_patch_changes(ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(state)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    plt.setp(ax.get_xticklabels(), visible=False)

# Reward raster
for w in range(n_wells):
    mask = chosen_well == w
    t_w = trial_num[mask]
    r_w = is_reward[mask]
    ax_raster.scatter(
        t_w[r_w == 0], np.full((r_w == 0).sum(), w, dtype=float),
        marker="o", s=10, facecolors="none", edgecolors=WELL_COLORS[w], linewidths=0.6,
    )
    ax_raster.scatter(
        t_w[r_w == 1], np.full((r_w == 1).sum(), w, dtype=float),
        marker="o", s=14, color=WELL_COLORS[w],
    )
_overlay_patch_changes(ax_raster)
ax_raster.set_yticks(range(n_wells))
ax_raster.set_yticklabels([f"w{w}" for w in range(n_wells)], fontsize=7)
ax_raster.set_ylim(-0.5, n_wells - 0.5)
ax_raster.invert_yaxis()
ax_raster.set_xlabel("trial number")
ax_raster.set_title("Reward raster (filled = rewarded, open = unrewarded)", fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Compare the learned obs covariate weights between models 3 and 1c+cov
#
# If the switching model is doing something genuinely new beyond what
# model 3 captured, its obs covariates should *differ* from model 3's.
# If they look identical, the regime structure is layered on top of
# the same covariate fit — either way, useful to see.

# %%
W_1c_cov = np.asarray(model_1c_cov.obs_weights_)
persev_diag_1c = np.diag(W_1c_cov[:, :n_wells])
winstay_diag_1c = np.diag(W_1c_cov[:, n_wells:])

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(n_wells)
width = 0.2
ax.bar(x - 1.5 * width, persev_diag, width,
       label="model 3 — perseveration diag", color="C0", edgecolor="k", linewidth=0.4)
ax.bar(x - 0.5 * width, persev_diag_1c, width,
       label="1c+cov — perseveration diag", color="C0", edgecolor="k", linewidth=0.4, alpha=0.5, hatch="//")
ax.bar(x + 0.5 * width, winstay_diag, width,
       label="model 3 — win-stay diag", color="C3", edgecolor="k", linewidth=0.4)
ax.bar(x + 1.5 * width, winstay_diag_1c, width,
       label="1c+cov — win-stay diag", color="C3", edgecolor="k", linewidth=0.4, alpha=0.5, hatch="//")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels([f"w{w}\n(p{WELL_PATCH[w]})" for w in range(n_wells)])
ax.set_ylabel("logit shift")
ax.set_title("Per-well covariate weights: model 3 (no switching) vs 1c+cov (with switching)")
ax.legend(fontsize=7, loc="best", ncol=2)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
#
# # Summary

# %%
summary_rows = [
    {"section": "0", "model": "(uniform 6-way baseline)",
     "final_ll": n_trials * float(np.log(1 / n_wells)), "notes": ""},
    {"section": "1 (no fit)", "model": "empirical anti-win-stay check",
     "final_ll": np.nan, "notes": "see § 1 plots"},
    {"section": "2", "model": "per-well one-hot baseline (=notebook 01 model 3)",
     "final_ll": float(lls_perwell[-1]),
     "notes": f"{obs_perwell.shape[1]} obs features"},
    {"section": "2", "model": "spatial-covariate parameterization",
     "final_ll": float(lls_spatial[-1]),
     "notes": f"{obs_spatial.shape[1]} obs features ({len(spatial_names)} scalars × K)"},
    {"section": "3", "model": "1c switching plain",
     "final_ll": float(lls_1c_plain[-1]),
     "notes": f"βs={np.asarray(model_1c_plain.inverse_temperatures_).round(2).tolist()}"},
    {"section": "3", "model": "1c switching + Frank-lab covariates",
     "final_ll": float(lls_1c_cov[-1]),
     "notes": f"βs={np.asarray(model_1c_cov.inverse_temperatures_).round(2).tolist()}"},
]
pd.DataFrame(summary_rows).round(2)

# %% [markdown]
# ## Take-homes (to be confirmed by the plots above)
#
# - **§ 1**: If the empirical win-stay/lose-stay bars confirm rewarded
#   visits are followed by *less* return, the model 3 anti-win-stay
#   finding is real, not an artifact. Trials-since-last-visit should
#   show a recovery curve consistent with depletion if depletion is
#   the right story.
# - **§ 2**: The spatial parameterization should fit nearly as well as
#   the per-well baseline with **half the parameters**. If
#   `alternate_leaf` and `win_same_patch` come out positive, the
#   animal is doing **patch-stay-but-leaf-shift** — a structured
#   pattern the published Frank-lab `stay_bias` (a single scalar)
#   cannot express.
# - **§ 3**: If the switching posterior collapses to one regime once
#   covariates are added, the original "explore vs exploit" split was
#   coding for switch cost / perseveration. If it survives, there's a
#   genuine strategy axis to investigate further (notebook 01c will
#   look at how it varies across the day).

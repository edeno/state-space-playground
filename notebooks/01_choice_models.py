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
# # Notebook 1 — Behavioral models on well choices
#
# Five state-space behavioral models on 180 trials from
# **`j1620210710_.nwb / 02_r1`**, all fit via SGD through
# `state_space_practice.sgd_fitting.SGDFittableMixin.fit_sgd`:
#
# | # | Model | Obs data | Notes |
# |---|---|---|---|
# | 1a | `MultinomialChoiceModel` | choices | no-covariate baseline |
# | 1b | `CovariateChoiceModel` | choices + prev-reward | tests whether recent reward helps predict next choice |
# | 1c | `SwitchingChoiceModel` (K=2) | choices | discrete strategy (explore/exploit) regime switching |
# | 1d | `ContingencyBeliefModel` (K=2) | choices + rewards | hidden rule-state inference |
# | 1e | `SmithLearningModel` × 6 | per-well reward outcomes | descriptive reward-rate smoother, one fit per well |
#
# **Comparisons:** 1a vs 1b tests whether reward covariates improve choice
# prediction. 1a vs 1c tests whether allowing strategy switching helps.
# 1d is *not* directly comparable to 1a–1c because it conditions on
# rewards (observation model has more information) — we report its
# log-likelihood separately. Smith 1e is a descriptive smoother and uses
# a different per-well dataset for each fit; its log-likelihoods are not
# on a shared scale either.

# %% [markdown]
# ## Setup — GPU pinning must come first
#
# `pick_free_gpu()` **must** be called before any other
# `state_space_playground` import. Importing `session` pulls in `jax`
# transitively via the vendored data loaders, and once jax is loaded its
# CUDA backend is initialized and `CUDA_VISIBLE_DEVICES` has no effect.

# %%
from state_space_playground.gpu import pick_free_gpu

pick_free_gpu(min_free_mb=20_000)

# %%
import logging

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from state_space_practice.contingency_belief import ContingencyBeliefModel
from state_space_practice.covariate_choice import CovariateChoiceModel
from state_space_practice.multinomial_choice import MultinomialChoiceModel
from state_space_practice.smith_learning_algorithm import SmithLearningModel
from state_space_practice.switching_choice import SwitchingChoiceModel

from state_space_playground.session import load_session

logging.basicConfig(level=logging.WARNING)

# %% [markdown]
# ## Load session and extract per-trial arrays

# %%
data = load_session("j1620210710_.nwb", "02_r1", use_sorted_hpc=True)
trials = data["trials"]
print(f"Loaded {len(trials)} trials.")
trials[["is_reward", "from_well", "to_well", "from_patch", "to_patch", "is_patch_change"]].head()

# %%
# The "choice" on each trial is to_well (where the animal decided to go).
# The "reward" is whether that trial ended in reward delivery.
chosen_well: np.ndarray = trials["to_well"].to_numpy().astype(np.int32)
is_reward: np.ndarray = trials["is_reward"].to_numpy().astype(np.int32)
is_patch_change: np.ndarray = trials["is_patch_change"].to_numpy()
trial_num: np.ndarray = trials.index.to_numpy()  # 1-indexed trial numbers

# Previous-trial reward covariate for 1b. First trial has no history so its
# covariate is 0.
prev_reward: np.ndarray = np.concatenate([[0], is_reward[:-1]]).astype(np.float32)

n_trials = len(trials)
n_wells = 6
well_visit_counts = np.bincount(chosen_well, minlength=n_wells)

print(f"n_trials        = {n_trials}")
print(f"n_wells         = {n_wells}")
print(f"n_rewards       = {int(is_reward.sum())}")
print(f"n_patch_changes = {int(is_patch_change.sum())}")
print(f"well visit counts: {dict(enumerate(well_visit_counts.tolist()))}")

# %% [markdown]
# ## 1a. `MultinomialChoiceModel` — no-covariate baseline

# %%
model_1a = MultinomialChoiceModel(n_options=n_wells)
lls_1a = model_1a.fit_sgd(jnp.asarray(chosen_well), num_steps=500, verbose=False)
smoothed_1a = np.asarray(model_1a.smoothed_option_values_)  # (n_trials, n_wells)
print(f"1a final LL: {lls_1a[-1]:.2f}")

# %% [markdown]
# ## 1b. `CovariateChoiceModel` with prev-trial reward as a dynamics covariate

# %%
model_1b = CovariateChoiceModel(n_options=n_wells, n_covariates=1)
lls_1b = model_1b.fit_sgd(
    jnp.asarray(chosen_well),
    covariates=jnp.asarray(prev_reward[:, None]),
    num_steps=500,
    verbose=False,
)
smoothed_1b = np.asarray(model_1b.smoothed_option_values_)
print(f"1b final LL: {lls_1b[-1]:.2f}   (Δ vs 1a: {lls_1b[-1] - lls_1a[-1]:+.2f})")
print(f"1b learned input gain: {np.asarray(model_1b.input_gain_).ravel()}")

# %% [markdown]
# ## 1c. `SwitchingChoiceModel` with 2 discrete strategies
#
# A symmetric init (both states starting from β=1.0) collapses to an
# unidentifiable symmetric fit on this data — two "strategies" with
# identical parameters and LL ≈ 0. To break the symmetry and give SGD
# a non-degenerate basin, we initialize the two states at clearly
# different inverse temperatures: `[0.5, 5.0]` (roughly "exploratory"
# vs "exploitative"). This is just an initialization tweak — if the
# data truly doesn't support two distinct regimes, SGD will still drag
# them together.

# %%
model_1c = SwitchingChoiceModel(
    n_options=n_wells,
    n_discrete_states=2,
    init_inverse_temperatures=jnp.array([0.5, 5.0]),
)
lls_1c = model_1c.fit_sgd(jnp.asarray(chosen_well), num_steps=500, verbose=False)
smoothed_disc_1c = np.asarray(model_1c.smoothed_discrete_probs_)  # (n_trials, 2)
print(f"1c final LL: {lls_1c[-1]:.2f}   (Δ vs 1a: {lls_1c[-1] - lls_1a[-1]:+.2f})")
print(f"1c learned inv temperatures: {np.asarray(model_1c.inverse_temperatures_)}")
print(f"1c learned process noises:  {np.asarray(model_1c.process_noises_)}")
print(f"1c transition matrix:\n{np.asarray(model_1c.discrete_transition_matrix_)}")

# %% [markdown]
# ## 1d. `ContingencyBeliefModel` with 2 hidden rule states
#
# Unlike 1a–1c, this model observes both choices AND rewards, so its
# log-likelihood is on a different scale — do not compare it directly.

# %%
model_1d = ContingencyBeliefModel(n_states=2, n_options=n_wells)
lls_1d = model_1d.fit_sgd(
    jnp.asarray(chosen_well),
    jnp.asarray(is_reward),
    num_steps=500,
    verbose=False,
)
smoothed_belief_1d = np.asarray(model_1d.smoothed_state_posterior_)  # (n_trials, 2)
print(f"1d final LL: {lls_1d[-1]:.2f}")
print(f"1d learned reward probs by state×option:\n{np.asarray(model_1d.reward_probs_)}")

# %% [markdown]
# ## 1e. `SmithLearningModel` per well (6 independent fits)
#
# For each well we extract just the trials where that well was chosen and
# feed the 0/1 reward outcomes to a Smith model as Bernoulli observations.
# The latent "learning state" is really a smoothed reward-rate for that
# specific well. Wells with very few visits are prior-dominated.

# %%
smith_results: dict[int, dict] = {}
for well in range(n_wells):
    well_mask = chosen_well == well
    visits_idx = np.where(well_mask)[0]
    rewards_at_well = is_reward[well_mask].astype(np.int32)
    if len(rewards_at_well) < 2:
        print(f"well {well}: only {len(rewards_at_well)} visits — skipping")
        continue
    m = SmithLearningModel(prob_correct_by_chance=0.5)
    lls = m.fit_sgd(jnp.asarray(rewards_at_well), num_steps=500, verbose=False)
    smith_results[well] = {
        "model": m,
        "lls": lls,
        "visit_indices": visits_idx,
        "n_visits": int(len(rewards_at_well)),
        "rewards": rewards_at_well,
        "smoothed_prob": np.asarray(m.smoothed_prob_correct_response),
    }
    print(
        f"well {well}: {len(rewards_at_well):3d} visits, "
        f"{int(rewards_at_well.sum()):3d} rewarded, final LL {lls[-1]:.2f}"
    )

# %% [markdown]
# ## Plots
#
# ### Color scheme and shared helpers
#
# Wells are colored by **patch membership** (2 wells per patch, 3 patches).
# Each patch gets a distinct hue with two brightness levels for its pair of
# wells. This makes the spatial structure visible at a glance.

# %%
import matplotlib.gridspec as gridspec  # noqa: E402

patch_change_trials = trial_num[is_patch_change]

# Patch-based well colors: wells {0,1} = patch 1, {2,3} = patch 2, {4,5} = patch 3.
# Two shades per patch so you can distinguish the two wells within a patch.
WELL_COLORS = {
    0: "#1b7837",  # patch 1 — dark green
    1: "#7fbf7b",  # patch 1 — light green
    2: "#c51b7d",  # patch 2 — dark magenta
    3: "#de77ae",  # patch 2 — light magenta
    4: "#2166ac",  # patch 3 — dark blue
    5: "#92c5de",  # patch 3 — light blue
}
WELL_PATCH = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3}
WELL_LABELS = {w: f"well {w} (patch {WELL_PATCH[w]})" for w in range(n_wells)}


def _overlay_patch_changes(ax: plt.Axes) -> None:
    """Overlay patch-change vertical lines."""
    for tc in patch_change_trials:
        ax.axvline(tc, color="k", lw=0.5, alpha=0.3, ls="--", zorder=1)


def _add_reward_raster(
    ax: plt.Axes,
    chosen: np.ndarray,
    reward: np.ndarray,
    trials: np.ndarray,
    show_xlabel: bool = True,
) -> None:
    """Draw a reward-by-well raster: wells on y-axis, trial on x-axis.

    Each trial gets a small marker at the chosen well's y-position.
    Rewarded trials are filled circles; unrewarded are open circles.
    """
    for well in range(n_wells):
        mask = chosen == well
        t_well = trials[mask]
        r_well = reward[mask]
        # Unrewarded visits — open circles
        unrewarded = t_well[r_well == 0]
        ax.scatter(
            unrewarded,
            np.full_like(unrewarded, well, dtype=float),
            marker="o",
            s=12,
            facecolors="none",
            edgecolors=WELL_COLORS[well],
            linewidths=0.7,
            zorder=3,
        )
        # Rewarded visits — filled circles
        rewarded = t_well[r_well == 1]
        ax.scatter(
            rewarded,
            np.full_like(rewarded, well, dtype=float),
            marker="o",
            s=18,
            color=WELL_COLORS[well],
            zorder=4,
        )
    _overlay_patch_changes(ax)
    ax.set_yticks(range(n_wells))
    ax.set_yticklabels(
        [f"well {w}" for w in range(n_wells)], fontsize=7
    )
    ax.set_ylim(-0.5, n_wells - 0.5)
    ax.invert_yaxis()
    if show_xlabel:
        ax.set_xlabel("trial number")


# %% [markdown]
# ### SGD convergence curves

# %%
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(lls_1a, label="1a Multinomial", lw=1.5)
ax.plot(lls_1b, label="1b Covariate (+prev reward)", lw=1.5)
ax.plot(lls_1c, label="1c Switching (K=2)", lw=1.5)
ax.plot(lls_1d, label="1d Contingency (K=2)", lw=1.5, ls="--")
ax.set_xlabel("SGD step")
ax.set_ylabel("log-likelihood")
ax.set_title("SGD convergence (1d is on a different scale — dashed)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1a vs 1b — smoothed latent option values over trials
#
# Values are relative to the reference option (well 0), which is fixed
# at 0 by the softmax identifiability constraint. The other five traces
# show each well's relative value. The **reward raster** at the bottom
# shows every trial: filled circles = rewarded, open circles = unrewarded,
# colored by well. Vertical dashed lines = patch changes.

# %%
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(
    3, 1, height_ratios=[3, 3, 1.5], hspace=0.08, figure=fig
)
ax_1a = fig.add_subplot(gs[0])
ax_1b = fig.add_subplot(gs[1], sharex=ax_1a)
ax_raster = fig.add_subplot(gs[2], sharex=ax_1a)

for ax, smoothed, title in [
    (ax_1a, smoothed_1a, "1a Multinomial — smoothed option values"),
    (ax_1b, smoothed_1b, "1b Covariate (+ prev reward) — smoothed option values"),
]:
    for well in range(n_wells):
        ls = ":" if well == 0 else "-"
        lbl = WELL_LABELS[well] + (" (ref)" if well == 0 else "")
        ax.plot(
            trial_num,
            smoothed[:, well],
            color=WELL_COLORS[well],
            lw=1.4,
            ls=ls,
            label=lbl,
        )
    _overlay_patch_changes(ax)
    ax.set_ylabel("latent value")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)

ax_1a.legend(fontsize=7, ncol=3, loc="lower center", frameon=False)
plt.setp(ax_1a.get_xticklabels(), visible=False)
plt.setp(ax_1b.get_xticklabels(), visible=False)

_add_reward_raster(ax_raster, chosen_well, is_reward, trial_num)
ax_raster.set_title(
    "Reward raster (filled = rewarded, open = unrewarded)", fontsize=9
)
plt.show()

# %% [markdown]
# ### 1b — what the covariate gain tells us
#
# `input_gain_` is a (K-1, 1) vector: the per-option boost to the latent
# value when the animal was rewarded on the previous trial.  Positive gain
# means "after a reward, this well's value goes up" — a win-stay signal.
# Well 0 is the reference (always 0 by identifiability).

# %%
gain = np.asarray(model_1b.input_gain_).ravel()
# Prepend 0 for the reference well
gain_full = np.concatenate([[0.0], gain])

fig, ax = plt.subplots(figsize=(7, 3.5))
bars = ax.bar(
    range(n_wells),
    gain_full,
    color=[WELL_COLORS[w] for w in range(n_wells)],
    edgecolor="k",
    linewidth=0.5,
)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(n_wells))
ax.set_xticklabels([f"well {w}\n(patch {WELL_PATCH[w]})" for w in range(n_wells)])
ax.set_ylabel("input gain (prev-reward effect)")
ax.set_title(
    "1b Covariate model — learned input gain per well\n"
    "(positive = value increases after reward = win-stay)"
)
ax.grid(True, alpha=0.3, axis="y")
# Annotate the reference well
ax.annotate(
    "reference\n(fixed at 0)",
    xy=(0, 0),
    xytext=(0.5, gain_full.min() - 0.03),
    fontsize=7,
    ha="center",
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1c Switching — strategy posterior with reward context
#
# The top panel shows the smoothed posterior over 2 behavioral strategies
# (explore vs. exploit). The bottom panel shows the reward raster so you
# can see whether strategy switches align with reward patterns.

# %%
fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(
    2, 1, height_ratios=[3, 1.5], hspace=0.08, figure=fig
)
ax_switch = fig.add_subplot(gs[0])
ax_raster = fig.add_subplot(gs[1], sharex=ax_switch)

# Label states by their fitted inverse temperature
inv_temps = np.asarray(model_1c.inverse_temperatures_)
state_labels = []
for s in range(len(inv_temps)):
    kind = "explore" if inv_temps[s] < np.median(inv_temps) else "exploit"
    state_labels.append(f"state {s} ({kind}, \u03b2={inv_temps[s]:.2f})")

ax_switch.stackplot(
    trial_num,
    smoothed_disc_1c.T,
    labels=state_labels,
    alpha=0.8,
)
_overlay_patch_changes(ax_switch)
ax_switch.set_ylabel("P(strategy)")
ax_switch.set_title(
    "1c Switching — smoothed strategy posterior\n"
    f"process noises: {np.asarray(model_1c.process_noises_).round(4)}"
)
ax_switch.set_ylim(0, 1)
ax_switch.legend(fontsize=8, loc="upper right")
plt.setp(ax_switch.get_xticklabels(), visible=False)

_add_reward_raster(ax_raster, chosen_well, is_reward, trial_num)
ax_raster.set_title(
    "Reward raster (filled = rewarded, open = unrewarded)", fontsize=9
)
plt.show()

# %% [markdown]
# ### 1d Contingency belief — posterior and learned reward structure
#
# **Left:** smoothed posterior over hidden rule states with the reward
# raster below.  **Right:** the learned reward probability matrix —
# P(reward | state, well) — shown as a heatmap. This is what the model
# thinks each "rule state" looks like in terms of per-well reward rates.

# %%
fig = plt.figure(figsize=(14, 7))
gs = gridspec.GridSpec(
    2, 2,
    width_ratios=[3, 1],
    height_ratios=[3, 1.5],
    hspace=0.08,
    wspace=0.3,
    figure=fig,
)
ax_belief = fig.add_subplot(gs[0, 0])
ax_raster = fig.add_subplot(gs[1, 0], sharex=ax_belief)
ax_heatmap = fig.add_subplot(gs[:, 1])

# Belief posterior
ax_belief.stackplot(
    trial_num,
    smoothed_belief_1d.T,
    labels=[f"rule state {s}" for s in range(smoothed_belief_1d.shape[1])],
    alpha=0.8,
)
_overlay_patch_changes(ax_belief)
ax_belief.set_ylabel("P(rule state)")
ax_belief.set_title("1d Contingency belief — smoothed posterior")
ax_belief.set_ylim(0, 1)
ax_belief.legend(fontsize=8, loc="upper right")
plt.setp(ax_belief.get_xticklabels(), visible=False)

# Reward raster
_add_reward_raster(ax_raster, chosen_well, is_reward, trial_num)
ax_raster.set_title(
    "Reward raster (filled = rewarded, open = unrewarded)", fontsize=9
)

# Reward probability heatmap
reward_probs = np.asarray(model_1d.reward_probs_)  # (n_states, n_wells)
im = ax_heatmap.imshow(
    reward_probs,
    aspect="auto",
    cmap="YlOrRd",
    vmin=0,
    vmax=1,
    origin="upper",
)
ax_heatmap.set_xticks(range(n_wells))
ax_heatmap.set_xticklabels(
    [f"well {w}\n(p{WELL_PATCH[w]})" for w in range(n_wells)], fontsize=8
)
ax_heatmap.set_yticks(range(reward_probs.shape[0]))
ax_heatmap.set_yticklabels([f"state {s}" for s in range(reward_probs.shape[0])])
ax_heatmap.set_title("Learned P(reward | state, well)")
# Annotate cells with values
for s in range(reward_probs.shape[0]):
    for w in range(n_wells):
        ax_heatmap.text(
            w, s, f"{reward_probs[s, w]:.2f}",
            ha="center", va="center", fontsize=8,
            color="white" if reward_probs[s, w] > 0.5 else "black",
        )
fig.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04, label="P(reward)")
plt.show()

# %% [markdown]
# ### 1e Smith per-well smoothed reward probability
#
# Each subplot shows one well's Smith model fit. Wells are grouped by
# patch (columns) and colored accordingly. Filled circles = rewarded
# visits, x = unrewarded visits.

# %%
fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharey=True, sharex=True)
axes_flat = axes.ravel()

for well in range(n_wells):
    ax = axes_flat[well]
    if well not in smith_results:
        ax.set_title(f"well {well} (patch {WELL_PATCH[well]}) — insufficient visits")
        ax.axis("off")
        continue
    r = smith_results[well]
    visit_trials = trial_num[r["visit_indices"]]
    color = WELL_COLORS[well]
    ax.plot(
        visit_trials,
        r["smoothed_prob"],
        color=color,
        lw=1.8,
        marker="o",
        markersize=3,
        label="smoothed P(reward)",
    )
    rewarded_here = visit_trials[r["rewards"] == 1]
    unrewarded_here = visit_trials[r["rewards"] == 0]
    ax.scatter(
        rewarded_here,
        np.ones_like(rewarded_here, dtype=float) * 1.02,
        marker="v",
        color=color,
        s=18,
        zorder=5,
    )
    ax.scatter(
        unrewarded_here,
        np.full_like(unrewarded_here, -0.02, dtype=float),
        marker="x",
        color="0.5",
        s=12,
        zorder=5,
    )
    _overlay_patch_changes(ax)
    ax.set_title(
        f"well {well} (patch {WELL_PATCH[well]})  n={r['n_visits']}", fontsize=9
    )
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    if well in (0, 3):
        ax.set_ylabel("P(reward)")
    if well >= 3:
        ax.set_xlabel("trial number")

fig.suptitle(
    "1e Smith per-well smoothed reward rate",
    fontsize=11,
    y=1.01,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary table
#
# Final SGD log-likelihoods grouped by comparability:
#
# - **`A_choices_only`** — 1a and 1b are comparable; both model the
#   same choice-only observations. 1c also has only choices but adds
#   discrete switching, which gives it a lot more capacity — its LL
#   can be dramatically better on short sequences without meaning the
#   fit is scientifically better (see the state-collapse note above).
# - **`B_choices_plus_rewards`** — 1d conditions on both choices and
#   rewards, so its LL is on a larger budget and is **not** comparable
#   to A.
# - **`C_smith_per_well_{W}`** — each Smith run uses a different per-well
#   dataset and its LL depends on the visit count for that well. Not
#   comparable across wells or to A/B.

# %%
import pandas as pd  # noqa: E402 (kept local to the summary cell)

summary_rows = [
    {"model": "1a Multinomial",              "obs": "choices",           "final_ll": lls_1a[-1], "comparable_group": "A_choices_only"},
    {"model": "1b Covariate (+prev reward)", "obs": "choices",           "final_ll": lls_1b[-1], "comparable_group": "A_choices_only"},
    {"model": "1c Switching (K=2)",          "obs": "choices",           "final_ll": lls_1c[-1], "comparable_group": "A_choices_only_switching"},
    {"model": "1d Contingency belief (K=2)", "obs": "choices + rewards", "final_ll": lls_1d[-1], "comparable_group": "B_choices_plus_rewards"},
]
for well, r in smith_results.items():
    summary_rows.append(
        {
            "model": f"1e Smith — well {well} (n={r['n_visits']})",
            "obs": "per-well Bernoulli",
            "final_ll": r["lls"][-1],
            "comparable_group": f"C_smith_per_well_{well}",
        }
    )
summary_df = pd.DataFrame(summary_rows)
summary_df.round(2)

# %% [markdown]
# ## Notes
#
# - **Well visit counts are highly skewed** (wells 4 and 5 got ~70% of
#   visits). 1e Smith traces for rarely-visited wells are prior-dominated.
#   Two wells triggered a benign warning because the animal was almost
#   never rewarded there — the smoothed latent is still correct, it just
#   sits near the prior.
#
# - **1b vs 1a**: if 1b's final LL is higher, reward information is helping
#   the dynamics. Look also at `model_1b.input_gain_` — a positive gain on
#   a given well means the animal is more likely to return to that well
#   after being rewarded there. In this run the largest positive gains are
#   on wells 4 and 5, the two most-visited wells.
#
# - **1c Switching — asymmetric init is load-bearing.** With a symmetric
#   init (both states starting at beta=1.0), this model collapses to an
#   unidentifiable symmetric fit on 180 trials: identical inverse
#   temperatures, identical process noise, symmetric transition matrix,
#   LL ~ 0 from the softmax becoming deterministic. The current cell uses
#   `init_inverse_temperatures=[0.5, 5.0]` to break the symmetry. After
#   fitting you should see clearly distinct states — e.g., one with low
#   beta (soft/exploratory, fast-switching) and one with high beta
#   (hard/exploitative, sticky self-transition). If the two rows of
#   `discrete_transition_matrix_` look symmetric, or if the two
#   `inverse_temperatures_` come back identical, the init wasn't enough
#   to escape the symmetric basin — try a more extreme split, add L2
#   regularization on beta, or reduce `num_steps`. If the data truly
#   doesn't support two distinct regimes, SGD will still drag them
#   together even from an asymmetric init.
#
# - **1d**: `reward_probs_` describes the latent model's belief about each
#   state's reward structure per well. Compare to the true patch layout.
#
# - **Log-likelihoods in the summary table are NOT all comparable.** 1a/1b
#   share the same observation model (choices only). 1c uses choices but
#   conditions on discrete states. 1d conditions on *both* choices and
#   rewards (more observations -> more likelihood budget). 1e Smith runs
#   each use a different per-well dataset. Only compare 1a<->1b directly.
#
# Next: notebook 2 — place field model via `PlaceFieldModel.fit_sgd()` on
# 305 sorted CA1 units.

# %% [markdown]
# ---
#
# # 2. Comparison with published parameterizations from `LorenFrankLab/SpatialBanditTask`
#
# The Frank Lab repo
# [`SpatialBanditTask`](https://github.com/LorenFrankLab/SpatialBanditTask) is
# the standard reference for analyzing this task and is the basis for the
# published "metalearning" paper. It models choice behavior with three model
# families, all fit by hierarchical EM across sessions:
#
# 1. **Q-learner** — per-well Q[w], EWMA update (`Q[w] += α(r − Q[w])`).
# 2. **Beta-Bernoulli** ("beta model") — per-well (α, β) posterior over
#    reward probability; published headline model.
# 3. **HMM** — discrete latent over a *fixed catalog* of plausible
#    per-well reward-probability configurations (e.g., 60 permutations of
#    `[0.2, 0.2, 0.2, 0.5, 0.5, 0.8]`); only the volatility, softmax
#    temperature, and biases are fit — emissions are NOT learned.
#
# All three additionally factor each trial as a **stem-then-leaf** decision
# (3-way + 2-way), with stem-level βgo/βstay split, perseveration / turn /
# spatial biases, and (in some variants) a depletion model for the "decay"
# task variant. We don't have decay in our data, so we drop that branch.
#
# We port a minimal version of each to JAX in
# `state_space_playground.frank_models` with two simplifications:
#
# - **Flat 6-way well softmax** — so the per-trial log-likelihood is on
#   the same scale as 1a / 1b / 1c above.
# - **Single-session SGD** — no hierarchical EM. We're fitting one
#   23-minute session, not pooling across animals.
#
# We keep the published parameterizations otherwise: state-update rules,
# fixed contingency catalog, and a single `stay_bias` (perseveration)
# bias added to the previous well's logit.

# %% [markdown]
# ## Structural differences at a glance
#
# | Aspect | SSP models (1a/1b/1c/1d) | Frank-lab models (2.1/2.2/2.3) |
# |---|---|---|
# | Latent value | continuous, **drifts** (random walk) trial-to-trial | deterministic update from rewards (Q EWMA / Beta posterior / HMM filter) |
# | What's fit | latent trajectory + dynamics noise + softmax β | a few global parameters (α, β, vol, biases); state is a deterministic function of history |
# | Reward as input | only 1b (covariate); 1d as observation | always (state updates from rewards) |
# | Discrete latents | 1c: K=2 strategy regimes (β switches); 1d: K=2 reward rules (emission learned) | HMM: K=60+ contingency configs (emission **fixed**) |
# | Action structure | flat 6-way softmax | published: stem (3) + on-switch leaf (2); we collapse to flat for comparability |
# | Inference | smoothing posterior over latent | filtering / forward-only |
# | Fit | SGD on full marginal LL | hierarchical EM (we use single-session SGD) |

# %%
from state_space_playground.frank_models import (  # noqa: E402
    BetaBernoulliModel,
    FrankHMMModel,
    QLearnerModel,
    get_contingency_catalog,
)

choices_jnp = jnp.asarray(chosen_well)
rewards_jnp = jnp.asarray(is_reward)

# %% [markdown]
# ## 2.1 Q-learner (Frank-lab)
#
# Per-well Q-values, EWMA update on chosen well only:
# `Q[w] ← (1−α)·Q[w] + α·r`. Choice probability is a softmax over `β·Q`
# with a `stay_bias` added to the previously-chosen well's logit.

# %%
model_2a = QLearnerModel()
lls_2a = model_2a.fit_sgd(choices_jnp, rewards_jnp, num_steps=500, verbose=False)
print(
    f"2.1 Q-learner    final LL: {lls_2a[-1]:.2f}   "
    f"α={model_2a.alpha_:.3f}, β={model_2a.beta_:.3f}, "
    f"init_Q={model_2a.init_Q_:.3f}, stay_bias={model_2a.stay_bias_:.3f}"
)

# %% [markdown]
# ## 2.2 Beta-Bernoulli ("beta model" — published metalearning paper)
#
# Per-well Beta posterior `(a_w, b_w)` over reward probability.  Update
# on chosen well: `a_w += r`, `b_w += (1−r)`. Optional decay pulls all
# wells back toward the prior `(a_baseline, a_baseline)` each trial
# (Frank-lab `beta_decay`). Choice value is the posterior mean
# `Q_w = a_w / (a_w + b_w)`.

# %%
model_2b = BetaBernoulliModel()
lls_2b = model_2b.fit_sgd(choices_jnp, rewards_jnp, num_steps=500, verbose=False)
print(
    f"2.2 Beta-Bernoulli final LL: {lls_2b[-1]:.2f}   "
    f"β={model_2b.beta_:.3f}, a_baseline={model_2b.a_baseline_:.3f}, "
    f"decay={model_2b.decay_:.3f}, stay_bias={model_2b.stay_bias_:.3f}"
)

# %% [markdown]
# ## 2.3 HMM with fixed contingency catalog
#
# Discrete latent `z_t ∈ {1..K}`; each state corresponds to one
# 6-tuple of per-well reward probabilities, drawn from a fixed catalog
# (default: 60 unique permutations of `[0.2, 0.2, 0.2, 0.5, 0.5, 0.8]`).
# Symmetric transitions parameterized by a single `volatility` (off-
# diagonal mass). We fit `volatility, β, stay_bias` — emissions are NOT
# learned. This is the key contrast with our 1d `ContingencyBeliefModel`,
# which fits a K=2 emission matrix from scratch.

# %%
catalog = get_contingency_catalog()
print(f"contingency catalog shape: {catalog.shape}  (K={catalog.shape[0]} states)")
model_2c = FrankHMMModel(contingencies=catalog)
lls_2c = model_2c.fit_sgd(choices_jnp, rewards_jnp, num_steps=500, verbose=False)
print(
    f"2.3 Frank HMM    final LL: {lls_2c[-1]:.2f}   "
    f"volatility={model_2c.volatility_:.3f}, β={model_2c.beta_:.3f}, "
    f"stay_bias={model_2c.stay_bias_:.3f}"
)

# %% [markdown]
# ## 2.4 Side-by-side log-likelihoods (choices-only models)
#
# All five models below predict `P(well[t] | history)` — same observation
# budget — so their final SGD log-likelihoods are directly comparable.
# 1d (Contingency belief) and 1e (Smith) are not in this comparison
# because they condition on rewards as observations.
#
# Baseline: a uniform 6-way softmax gives `n_trials · log(1/6)`.

# %%
uniform_ll = n_trials * float(np.log(1.0 / n_wells))
choice_only_rows = [
    {"model": "uniform 6-way",          "final_ll": uniform_ll, "n_params": 0},
    {"model": "1a SSP Multinomial",     "final_ll": float(lls_1a[-1]), "n_params": "varies (latent trajectory)"},
    {"model": "1b SSP Covariate",       "final_ll": float(lls_1b[-1]), "n_params": "varies + 1 gain"},
    {"model": "1c SSP Switching (K=2)", "final_ll": float(lls_1c[-1]), "n_params": "varies + switching"},
    {"model": "2.1 Q-learner",          "final_ll": float(lls_2a[-1]), "n_params": 4},
    {"model": "2.2 Beta-Bernoulli",     "final_ll": float(lls_2b[-1]), "n_params": 4},
    {"model": "2.3 Frank HMM (K=60)",   "final_ll": float(lls_2c[-1]), "n_params": 3},
]
choice_only_df = pd.DataFrame(choice_only_rows)
choice_only_df["delta_vs_uniform"] = choice_only_df["final_ll"] - uniform_ll
choice_only_df.round(2)

# %% [markdown]
# ## 2.5 Per-trial latent value trajectories
#
# All three published models give a per-trial value vector over the 6
# wells (Q for the Q-learner, posterior mean for Beta-Bernoulli, expected
# reward `ϕᵀα` for the HMM). Compare these to the SSP smoothed latent
# values from 1a / 1b. The published models update *forward only*; the
# SSP models smooth (use the full sequence to refine each trial's
# estimate).

# %%
traj_2a = model_2a.trajectories(choices_jnp, rewards_jnp)
traj_2b = model_2b.trajectories(choices_jnp, rewards_jnp)
traj_2c = model_2c.trajectories(choices_jnp, rewards_jnp)

fig = plt.figure(figsize=(13, 11))
gs = gridspec.GridSpec(
    4, 1, height_ratios=[3, 3, 3, 1.5], hspace=0.12, figure=fig
)
ax_ql = fig.add_subplot(gs[0])
ax_bb = fig.add_subplot(gs[1], sharex=ax_ql)
ax_hm = fig.add_subplot(gs[2], sharex=ax_ql)
ax_raster = fig.add_subplot(gs[3], sharex=ax_ql)

for ax, Q, title in [
    (ax_ql, traj_2a["Q"], "2.1 Q-learner — per-well Q[w]"),
    (ax_bb, traj_2b["Q"], "2.2 Beta-Bernoulli — posterior mean a_w / (a_w + b_w)"),
    (ax_hm, traj_2c["Q"], "2.3 Frank HMM (K=60) — expected reward prob \u03d5\u1d40\u03b1 per well"),
]:
    for well in range(n_wells):
        ax.plot(
            trial_num,
            Q[:, well],
            color=WELL_COLORS[well],
            lw=1.4,
            label=WELL_LABELS[well],
        )
    _overlay_patch_changes(ax)
    ax.set_ylabel("value")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

ax_ql.legend(fontsize=7, ncol=3, loc="lower center", frameon=False)
plt.setp(ax_ql.get_xticklabels(), visible=False)
plt.setp(ax_bb.get_xticklabels(), visible=False)
plt.setp(ax_hm.get_xticklabels(), visible=False)

_add_reward_raster(ax_raster, chosen_well, is_reward, trial_num)
ax_raster.set_title("Reward raster (filled = rewarded, open = unrewarded)", fontsize=9)
plt.show()

# %% [markdown]
# ## 2.6 SGD convergence — published vs SSP choice-only models

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(lls_1a, label="1a SSP Multinomial", lw=1.5, color="C0")
ax.plot(lls_1b, label="1b SSP Covariate", lw=1.5, color="C1")
ax.plot(lls_1c, label="1c SSP Switching (K=2)", lw=1.5, color="C2")
ax.plot(lls_2a, label="2.1 Q-learner", lw=1.5, ls="--", color="C3")
ax.plot(lls_2b, label="2.2 Beta-Bernoulli", lw=1.5, ls="--", color="C4")
ax.plot(lls_2c, label="2.3 Frank HMM (K=60)", lw=1.5, ls="--", color="C5")
ax.axhline(uniform_ll, color="k", lw=0.8, ls=":", label=f"uniform 6-way ({uniform_ll:.0f})")
ax.set_xlabel("SGD step")
ax.set_ylabel("log-likelihood")
ax.set_title("SGD convergence — SSP (solid) vs Frank-lab parameterizations (dashed)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2.7 What the SSP framing surfaces that the published models don't
#
# - **Time-varying latent values without prescribing a learning rule.**
#   1a/1b's smoothed `option_values_` are unconstrained random walks in
#   logit space — they don't *assume* the animal does Q-learning,
#   Bayesian updating, or any specific reward → value mapping. Whatever
#   shape the data prefers, the latent picks it up. The Q-learner's
#   `α` and Beta-Bernoulli's `decay` constrain the *form* of trial-to-
#   trial change.
# - **Smoothing vs filtering.** SSP gives a posterior over the latent
#   that uses the *whole* sequence (forward-backward). The Frank-lab
#   models update forward only — each trial's value is a function of
#   strictly past trials. This matters when comparing inferred value
#   trajectories: the SSP curves are a smoothed picture of "what was the
#   latent doing", while the published-model curves are a causal "what
#   would the animal have known at this trial".
# - **Identifiable strategy switching (1c).** Discrete `K=2` switches in
#   inverse temperature give a posterior over "explore vs exploit"
#   regimes that the Frank-lab models can't represent: their HMM is over
#   the *world* (which contingency is true), not over the *agent's*
#   policy. Animal-level decision-noise dynamics are first-class in 1c,
#   absent in 2.1–2.3.
# - **Learned emission structure (1d).** Even at K=2, 1d's
#   `reward_probs_` matrix is fit from data — useful when you don't
#   know the contingency catalog a priori. The Frank-lab HMM's emission
#   is fixed: it can only express world-states the catalog includes.
#
# ## What the published parameterizations capture that SSP doesn't (here)
#
# - **Stem/leaf action hierarchy.** The published code factors each
#   trial as a 3-way stem softmax + (on switch) a 2-way leaf softmax,
#   with separate `βgo` / `βstay` and per-stem `spatial_bias`. SSP and
#   our flat ports treat all 6 wells symmetrically. If you're trying to
#   tease apart "switch cost" from "value", the published factorization
#   is the right hammer.
# - **Cumulative evidence vs forgetting.** The Beta-Bernoulli model
#   accumulates counts across the entire session by default; with
#   `decay=0` an early reward at well `w` keeps influencing `Q_w` 100
#   trials later. SSP's random-walk latent has no such notion of
#   "evidence weight" — it just drifts. Whether the animal really
#   integrates over long horizons or smoothly forgets is a structural
#   claim that lives explicitly in the Frank-lab parameterization.
# - **Group-level inference.** Hierarchical EM across sessions/animals
#   gives the published code population-level priors and per-subject
#   shrinkage. SSP fits are per-session; pooling would need extra
#   plumbing.
#
# ## Suggested next experiments
#
# - Refit 2.1 / 2.2 / 2.3 across all 7 run epochs of `j1620210710_.nwb`
#   and look at **parameter stability** — does `α` or `volatility` drift
#   across the day? SSP's switching posterior (1c) gives a within-epoch
#   answer to a similar question.
# - Build a wrapper that evaluates the SSP latent trajectories under the
#   Frank-lab stem/leaf observation model. That's the apples-to-apples
#   way to ask "does SSP's flexible latent + the published action
#   factorization beat both pieces alone?"
# - Compare SSP 1d (`reward_probs_`) to the Frank HMM's catalog:
#   does the K=2 emission matrix learned by 1d look like any of the 60
#   pre-specified contingency configurations?

# %% [markdown]
# ---
#
# # 3. SSP `CovariateChoiceModel` with Frank-lab-style covariates
#
# The Frank-lab models all included a `stay_bias` (perseveration on the
# previous well), which the Q-learner in 2.1 latched onto strongly
# (`stay_bias ≈ -6` — anti-perseveration, i.e., depletion). The SSP
# baselines 1a / 1b can't express that signal natively: 1a has no
# covariates at all, and 1b only carries a scalar `prev_reward` into the
# latent dynamics.
#
# But `CovariateChoiceModel` already exposes both
# `covariates` (dynamics-time, accumulates on `x_t`) **and**
# `obs_covariates` (observation-time, per-trial logit shift, no memory).
# `obs_covariates` is mathematically identical to Frank-lab's `stay_bias`
# mechanism — we just need to feed it the right design matrix.
#
# We add three Frank-lab-style covariates:
#
# - **prev-reward** (scalar, dynamics): kept identical to 1b for
#   continuity — does last trial's reward push the latent value?
# - **perseveration** (per-well one-hot of prev-well, observation):
#   per-well stay-bias. A negative entry on well *w* means "less likely
#   to revisit *w* on trial *t* given I was just there" — the depletion
#   signal Q-learner found.
# - **win-stay** (per-well, observation, gated by prev-reward): per-well
#   "go back if rewarded" weight — separates pure perseveration from
#   reward-driven return.
#
# The model now has the same input space as the Frank-lab Q-learner,
# layered on top of the SSP smoothed latent value. If the gap between
# 2.1 and 1b really is mostly anti-perseveration, this should close it.

# %%
prev_well_int = np.concatenate([[0], chosen_well[:-1]]).astype(np.int32)
prev_well_oh = np.eye(n_wells, dtype=np.float32)[prev_well_int]  # (T, 6)
win_stay_oh = (prev_reward[:, None] * prev_well_oh).astype(np.float32)  # (T, 6)
obs_covariates_3 = np.hstack([prev_well_oh, win_stay_oh]).astype(np.float32)  # (T, 12)
print(f"obs_covariates shape: {obs_covariates_3.shape}  "
      f"(6 perseveration + 6 win-stay)")

model_3 = CovariateChoiceModel(
    n_options=n_wells,
    n_covariates=1,           # prev_reward (dynamics, identical to 1b)
    n_obs_covariates=12,      # 6 perseveration + 6 win-stay
)
lls_3 = model_3.fit_sgd(
    jnp.asarray(chosen_well),
    covariates=jnp.asarray(prev_reward[:, None]),
    obs_covariates=jnp.asarray(obs_covariates_3),
    num_steps=500,
    verbose=False,
)
smoothed_3 = np.asarray(model_3.smoothed_option_values_)
print(
    f"3 SSP+Frank-cov  final LL: {lls_3[-1]:.2f}   "
    f"(Δ vs 1b: {lls_3[-1] - lls_1b[-1]:+.2f}, "
    f"Δ vs 2.1 Q-learner: {lls_3[-1] - lls_2a[-1]:+.2f})"
)

# %% [markdown]
# ## 3.1 Learned covariate weights
#
# Three pieces to read off:
#
# - **Dynamics-time `input_gain_`** (K-1 vector): same role as in 1b —
#   the per-option boost to the latent random-walk drift after a
#   rewarded trial. Should look broadly similar to 1b's gains.
# - **Observation `obs_weights_[:, :6]`** (perseveration, K×6): each
#   *column* w gives the per-well logit offset on the trial *after* the
#   animal visited well w. The diagonal is the "stay" effect for each
#   well; off-diagonal entries say "after well w, well w' becomes
#   more/less likely". A negative diagonal = depletion.
# - **Observation `obs_weights_[:, 6:]`** (win-stay, K×6): same shape,
#   but only fires on rewarded trials. A positive diagonal = "after a
#   reward at w, return to w".

# %%
input_gain_3 = np.asarray(model_3.input_gain_).ravel()  # (K-1,)
obs_weights_3 = np.asarray(model_3.obs_weights_)        # (K, 12)
persev_W = obs_weights_3[:, :n_wells]                   # (K, 6)
winstay_W = obs_weights_3[:, n_wells:]                  # (K, 6)

# Per-well "stay" effect = diagonal of perseveration matrix
persev_diag = np.diag(persev_W)
winstay_diag = np.diag(winstay_W)
print("perseveration diagonal (logit shift on well w after visiting w):")
print(np.round(persev_diag, 3))
print("win-stay diagonal (extra logit shift if last trial at w was rewarded):")
print(np.round(winstay_diag, 3))
print(f"dynamics-time input_gain (cf 1b: {np.asarray(model_1b.input_gain_).ravel().round(3)})")
print(np.round(input_gain_3, 3))

# %% [markdown]
# ## 3.2 Visualizing the learned biases
#
# Three panels: (a) per-well perseveration **diagonal** vs. Q-learner's
# scalar `stay_bias` (the SSP version split per well); (b) full
# perseveration matrix `obs_weights_[:, :6]` as a heatmap; (c) full
# win-stay matrix `obs_weights_[:, 6:]` as a heatmap. The two heatmaps
# carry off-diagonal information that the scalar Frank-lab `stay_bias`
# can't express — e.g., "after well 4, well 5 becomes more likely"
# (within-patch alternation).

# %%
fig = plt.figure(figsize=(14, 4.5))
gs = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 2], wspace=0.35, figure=fig)
ax_diag = fig.add_subplot(gs[0])
ax_per = fig.add_subplot(gs[1])
ax_win = fig.add_subplot(gs[2])

x = np.arange(n_wells)
width = 0.35
ax_diag.bar(
    x - width / 2, persev_diag, width,
    color=[WELL_COLORS[w] for w in range(n_wells)],
    edgecolor="k", linewidth=0.4, label="perseveration diag",
)
ax_diag.bar(
    x + width / 2, winstay_diag, width,
    color=[WELL_COLORS[w] for w in range(n_wells)],
    edgecolor="k", linewidth=0.4, hatch="//", label="win-stay diag",
)
ax_diag.axhline(0, color="k", lw=0.6)
ax_diag.axhline(
    model_2a.stay_bias_, color="C3", lw=1.0, ls="--",
    label=f"2.1 Q-learner scalar stay_bias ({model_2a.stay_bias_:.2f})",
)
ax_diag.set_xticks(x)
ax_diag.set_xticklabels([f"w{w}\n(p{WELL_PATCH[w]})" for w in range(n_wells)])
ax_diag.set_ylabel("logit shift")
ax_diag.set_title("3 — per-well stay / win-stay (diag)\nvs. Q-learner scalar")
ax_diag.legend(fontsize=7, loc="best")
ax_diag.grid(True, alpha=0.3, axis="y")

vmax = float(np.max(np.abs(obs_weights_3)))
for ax, mat, title in [
    (ax_per, persev_W, "perseveration weights\nrow = effect on well, col = previous well"),
    (ax_win, winstay_W, "win-stay weights\n(perseveration × prev_reward)"),
]:
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_wells))
    ax.set_xticklabels([f"prev w{w}" for w in range(n_wells)], fontsize=7, rotation=30)
    ax.set_yticks(range(n_wells))
    ax.set_yticklabels([f"this w{w}" for w in range(n_wells)], fontsize=7)
    for i in range(n_wells):
        for j in range(n_wells):
            ax.text(
                j, i, f"{mat[i, j]:.1f}",
                ha="center", va="center", fontsize=7,
                color="white" if abs(mat[i, j]) > 0.6 * vmax else "black",
            )
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3.3 Updated head-to-head — choices-only, with SSP+covariates added

# %%
extended_rows = choice_only_rows + [
    {
        "model": "3 SSP+Frank covariates",
        "final_ll": float(lls_3[-1]),
        "n_params": "varies + 1 dyn + 12 obs",
    },
]
extended_df = pd.DataFrame(extended_rows)
extended_df["delta_vs_uniform"] = extended_df["final_ll"] - uniform_ll
extended_df = extended_df.sort_values("final_ll", ascending=False).reset_index(drop=True)
extended_df.round(2)

# %%
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(lls_1a, label="1a SSP Multinomial", lw=1.3, color="C0")
ax.plot(lls_1b, label="1b SSP Covariate (+prev_reward)", lw=1.3, color="C1")
ax.plot(lls_1c, label="1c SSP Switching (K=2)", lw=1.3, color="C2")
ax.plot(lls_2a, label="2.1 Q-learner", lw=1.3, ls="--", color="C3")
ax.plot(lls_2b, label="2.2 Beta-Bernoulli", lw=1.3, ls="--", color="C4")
ax.plot(lls_2c, label="2.3 Frank HMM (K=60)", lw=1.3, ls="--", color="C5")
ax.plot(lls_3, label="3 SSP+Frank covariates", lw=2.0, color="k")
ax.axhline(uniform_ll, color="k", lw=0.8, ls=":", label=f"uniform 6-way ({uniform_ll:.0f})")
ax.set_xlabel("SGD step")
ax.set_ylabel("log-likelihood")
ax.set_title("SGD convergence — adding Frank-lab covariates to SSP")
ax.legend(fontsize=7, loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3.4 Reading the result
#
# - If `lls_3` ends up close to or above `lls_2a`, then most of what
#   the Q-learner was capturing was the per-well stay/win-stay logit
#   structure — *not* a learning rule. SSP+covariates expresses the
#   same effect with a more flexible latent on top.
# - If `lls_3` overshoots `lls_2a` substantially, the SSP latent's
#   smoothed drift is doing real explanatory work that the Q-learner's
#   fixed `α` can't reach (e.g., slow within-block drift in well
#   preference that EWMA over-smooths or under-smooths).
# - The off-diagonal `persev_W` entries are **not expressible** in the
#   Frank-lab parameterization (their `stay_bias` is a scalar, not a
#   K×K matrix). If those entries are large and structured (e.g.,
#   strong same-patch off-diagonals), that's a structural finding the
#   published model is mute on — animals aren't just biased toward the
#   last well, but toward the last *patch*.
# - The dynamics-time `input_gain_` of model 3 should be similar to 1b's,
#   since both feed `prev_reward` through the same channel. Differences
#   would mean the obs-time covariates absorbed the reward signal,
#   leaving the dynamics gain free to fit a different aspect of the
#   data.

# %% [markdown]
# ---
#
# # 4. SSP ↔ Frank-lab: trajectory & prediction comparison
#
# **Frame.** The SSP latent is a minimally-committed random-walk
# trajectory in logit space; the Frank-lab Q(t), E[p](t), and ϕᵀα(t)
# are more-committed mechanistic trajectories. Where all three agree,
# multiple mechanisms are consistent with the behavior. Where they
# diverge, some mechanism fails to capture structure the SSP latent
# sees.
#
# Three complementary comparisons — each answers a different question:
#
# | § | Comparison | What it tells us |
# |---|---|---|
# | 4.1 | **Predictive agreement** — per-trial choice probability | Are these models *behaviorally* distinguishable on this data? If not, arguing about their trajectories is unfalsifiable. |
# | 4.2 | **Trajectory overlay** — per-well value over time | Where do the mechanisms' latents agree with SSP's? Where they diverge, that mechanism can't produce what SSP sees. |
# | 4.3 | **Residual analysis** — model 3's SSP latent after obs covariates | What structure does the SSP latent still carry *after* perseveration + win-stay are controlled for? That's the residual nobody's mechanism explains. |
#
# All comparisons are on the 180 trials of `j1620210710_ / 02_r1`.

# %% [markdown]
# ## § 4.0 Extract per-trial choice probabilities for each model
#
# Each model's per-trial `P(well | history)` is a `(T, K=6)` matrix.
# We compute this for 1a, 1b, 2.1, 2.2, 2.3 from their fitted
# parameters / trajectories. (1c, 1d aren't included — 1c's
# degenerate near-deterministic fit and 1d's different observation
# budget make them not directly comparable.)

# %%
def softmax_from_logits(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=-1, keepdims=True)


# SSP models store smoothed_option_values_ (T, K) in logit space with
# option 0 pinned to 0. Per-trial P(well) = softmax(β · values).
beta_1a = float(model_1a.inverse_temperature)
p_1a = softmax_from_logits(beta_1a * np.asarray(model_1a.smoothed_option_values_))
beta_1b = float(model_1b.inverse_temperature)
p_1b = softmax_from_logits(beta_1b * np.asarray(model_1b.smoothed_option_values_))

# Frank-lab models return log_p directly from trajectories().
traj_2a = model_2a.trajectories(choices_jnp, rewards_jnp)
traj_2b = model_2b.trajectories(choices_jnp, rewards_jnp)
traj_2c = model_2c.trajectories(choices_jnp, rewards_jnp)
p_2a = np.exp(traj_2a["log_p"])
p_2b = np.exp(traj_2b["log_p"])
p_2c = np.exp(traj_2c["log_p"])

# Sanity: each row should sum to 1
for name, p in [("1a", p_1a), ("1b", p_1b), ("2.1", p_2a), ("2.2", p_2b), ("2.3", p_2c)]:
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6), f"{name} rows don't sum to 1"
print("All per-trial P(well) matrices computed, each (T, K=6).")

model_probs = {
    "1a Multinomial": p_1a,
    "1b Covariate": p_1b,
    "2.1 Q-learner": p_2a,
    "2.2 Beta-Bernoulli": p_2b,
    "2.3 Frank HMM": p_2c,
}

# %% [markdown]
# ## § 4.1 Predictive agreement — are these models distinguishable on this data?
#
# For each pair of models (A, B) we report:
#
# 1. **Mean symmetric KL** across trials — `0.5 * (KL(P_A||P_B) + KL(P_B||P_A))` averaged over T.
#    0 means the models give identical per-trial distributions; higher
#    = more distinguishable. Interpretable in nats.
# 2. **Correlation of log P(chosen)** across trials — 1 means models
#    rank trials identically in terms of predicted surprise.
#
# **Why this is the first comparison.** If two models have mean KL ≈
# 0, arguing about whether one is "more correct" based on their latent
# trajectories is a distinction the data cannot adjudicate. If mean
# KL is large, the models are genuinely telling different stories
# about which wells are most likely on each trial.

# %%
def symmetric_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)
    kl_pq = (p * (np.log(p) - np.log(q))).sum(axis=1)
    kl_qp = (q * (np.log(q) - np.log(p))).sum(axis=1)
    return float((0.5 * (kl_pq + kl_qp)).mean())


model_names = list(model_probs.keys())
n_models = len(model_names)
kl_mat = np.zeros((n_models, n_models))
logp_chosen_mat = np.array([
    np.log(model_probs[m][np.arange(n_trials), chosen_well] + 1e-12)
    for m in model_names
])  # (n_models, T)
corr_mat = np.corrcoef(logp_chosen_mat)

for i in range(n_models):
    for j in range(n_models):
        kl_mat[i, j] = symmetric_kl(model_probs[model_names[i]], model_probs[model_names[j]])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, mat, title, fmt, vmin, vmax, cmap in [
    (axes[0], kl_mat, "Mean symmetric KL (nats/trial)\nhigher = more distinguishable", "{:.2f}", 0, kl_mat.max(), "YlOrRd"),
    (axes[1], corr_mat, "Correlation of log P(chosen) across trials\nhigher = same predictive ordering", "{:.2f}", -1, 1, "RdBu_r"),
]:
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_names, fontsize=8)
    ax.set_title(title, fontsize=9)
    for i in range(n_models):
        for j in range(n_models):
            color = "white" if (cmap == "YlOrRd" and mat[i, j] > 0.5 * vmax) or (cmap == "RdBu_r" and abs(mat[i, j]) > 0.5) else "black"
            ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# %% [markdown]
# **How to read.**
#
# - **KL heatmap**: diagonal is 0 (a model vs itself). Off-diagonal
#   entries quantify per-trial disagreement. A KL of 0.1 nats/trial
#   means that on an average trial, the two models assign the chosen
#   well log-probabilities that differ by ~0.1 — small; the models are
#   giving similar predictions. 1+ nats/trial means dramatic
#   disagreement.
# - **Correlation heatmap**: if two models have correlation 0.9 in
#   `log P(chosen)` across trials, they largely agree on *which trials
#   are predictable* even if they assign slightly different numbers.
#   Low correlation = they disagree on which trials are surprising.
# - **SSP vs Frank-lab block**: the top-left 2×2 (SSP 1a/1b) and the
#   bottom-right 3×3 (Frank-lab) should each be internally similar.
#   The top-right 2×3 rectangle (SSP vs Frank) is where the framework
#   comparison lives — if that block has low KL, the frameworks are
#   behaviorally equivalent on this session; if high, SSP is genuinely
#   telling a different behavioral story.

# %% [markdown]
# ## § 4.2 Trajectory overlay — per-well value over time
#
# For each well, overlay five trajectories:
#
# - SSP 1a: `smoothed_option_values_[:, w]` — logit-space random walk.
# - SSP 1b: same thing but with `prev_reward` as a dynamics covariate.
# - Q-learner Q(t) — linear reward EWMA.
# - Beta-Bernoulli E[p](t) = α_w / (α_w + β_w).
# - Frank HMM ϕᵀα(t) — expected reward prob from the posterior over
#   contingency configurations.
#
# **Scale mismatch is real.** SSP values are in logit space (unbounded,
# with well 0 pinned at 0); Frank-lab values are in probability space
# `[0, 1]`. We show **P(well | model, trial)** on a common y-axis so
# everything is in the same scale — that's the softmax-transformed
# view of each model's latent.

# %%
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.15, figure=fig)
axes_6 = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]

line_specs = [
    ("1a Multinomial", p_1a, "-", 1.4, "C0"),
    ("1b Covariate", p_1b, "-", 1.4, "C1"),
    ("2.1 Q-learner", p_2a, "--", 1.2, "C3"),
    ("2.2 Beta-Bernoulli", p_2b, "--", 1.2, "C4"),
    ("2.3 Frank HMM", p_2c, "--", 1.2, "C5"),
]

for w, ax in enumerate(axes_6):
    for name, p, ls, lw, color in line_specs:
        ax.plot(trial_num, p[:, w], ls, lw=lw, color=color, alpha=0.85, label=name)
    # Mark trials where this well was chosen (filled=rewarded, open=unrewarded)
    mask_w = chosen_well == w
    reward_w = is_reward[mask_w]
    trials_w = trial_num[mask_w]
    ax.scatter(trials_w[reward_w == 0], np.full((reward_w == 0).sum(), 0.02),
               marker="o", s=12, facecolors="none", edgecolors=WELL_COLORS[w], linewidths=0.6)
    ax.scatter(trials_w[reward_w == 1], np.full((reward_w == 1).sum(), 0.02),
               marker="o", s=18, color=WELL_COLORS[w])
    _overlay_patch_changes(ax)
    ax.set_title(f"well {w} (patch {WELL_PATCH[w]})  — "
                 f"{mask_w.sum()} visits / {int(reward_w.sum())} rewards",
                 fontsize=10)
    ax.set_ylim(-0.03, 1.0)
    ax.set_ylabel("P(well chosen)")
    if w >= 4:
        ax.set_xlabel("trial")
    ax.grid(True, alpha=0.3)

axes_6[0].legend(fontsize=7, loc="upper right", ncol=1, framealpha=0.9)
fig.suptitle(
    "Per-trial P(well) overlay — SSP (solid) vs Frank-lab (dashed)\n"
    "markers show actual visits at y≈0 (filled = rewarded, open = unrewarded)",
    fontsize=11, y=0.995,
)
plt.show()

# %% [markdown]
# **How to read each panel.**
#
# - Look at wells 4 and 5 first — they have the most visits and the
#   most information, so trajectories are best constrained there.
# - **SSP 1a/1b (solid)** adapt smoothly to the visit pattern without
#   a prescribed reward response. They track "how often did the animal
#   actually go there recently" in a model-agnostic way.
# - **Q-learner (dashed red)** typically has a very different shape:
#   if `α ≈ 0` (our 02_r1 fit) the Q-values barely move, so the
#   probabilities reflect almost purely the stay_bias. If `α > 0` the
#   Q-values step up on every rewarded visit.
# - **Beta-Bernoulli (dashed purple)** accumulates: a rewarded visit
#   pushes the posterior mean up sharply, with less movement on
#   unrewarded visits. Its trajectory crosses SSP's when the animal
#   starts to deviate from a pure reward-integrator.
# - **Frank HMM (dashed brown)** is often flatter because its expected
#   reward `ϕᵀα` is a convex combination over 60 pre-specified
#   contingencies; the support doesn't include extreme per-well
#   probabilities without evidence.
# - **Where mechanisms diverge from SSP is where they fail.** If
#   Q-learner predicts P(well 5) rising linearly with rewards but SSP
#   shows it *already high* from the start, the mechanism's ramp-up
#   story doesn't match the data — the animal arrived with a prior.

# %% [markdown]
# ## § 4.3 Residual analysis — what does model 3's SSP latent still carry?
#
# Model 3 combines the SSP random-walk latent with observation-time
# perseveration + win-stay covariates. The covariates absorb per-trial
# stay-or-leave biases; whatever latent trajectory remains is the
# part of the behavior that **isn't** perseveration, **isn't**
# reward-triggered, and **isn't** captured by any fixed parameterization
# — it's the residual drift.
#
# Here we plot model 3's smoothed latent value per well in logit space
# (where it lives natively, no softmax), and overlay the Q-learner
# and Beta-Bernoulli Q(t) centered-and-scaled into the same space for
# shape comparison.

# %%
vals_3 = np.asarray(model_3.smoothed_option_values_)  # (T, K=6) in logit space, option 0 pinned
# Center Q-learner and Beta-Bernoulli trajectories to match model 3's
# reference convention: subtract well-0 value so option 0 is pinned.
def pin_ref(Q: np.ndarray) -> np.ndarray:
    return Q - Q[:, [0]]


Q_2a_ref = pin_ref(traj_2a["Q"])
Q_2b_ref = pin_ref(traj_2b["Q"])

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.15, figure=fig)
axes_6 = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]

for w, ax in enumerate(axes_6):
    ax.plot(trial_num, vals_3[:, w], "-", lw=1.8, color="k",
            label="model 3 — SSP latent (residual)")
    if w != 0:
        ax.plot(trial_num, Q_2a_ref[:, w], "--", lw=1.2, color="C3",
                alpha=0.8, label="2.1 Q-learner (pinned to w0)")
        ax.plot(trial_num, Q_2b_ref[:, w], "--", lw=1.2, color="C4",
                alpha=0.8, label="2.2 Beta-Bernoulli (pinned to w0)")
    else:
        ax.axhline(0, color="0.4", lw=0.8)
        ax.text(trial_num[0], 0.02, "reference (pinned at 0)", fontsize=8, color="0.4")
    _overlay_patch_changes(ax)
    ax.set_title(f"well {w} (patch {WELL_PATCH[w]})", fontsize=10)
    ax.set_ylabel("logit-space value (ref: well 0)")
    ax.grid(True, alpha=0.3)
    if w >= 4:
        ax.set_xlabel("trial")

axes_6[1].legend(fontsize=7, loc="best")
fig.suptitle(
    "Model 3 residual SSP latent vs Frank-lab Q (all pinned to well 0 = 0)\n"
    "The black line is drift NOT explained by perseveration, win-stay, or prev_reward.",
    fontsize=11, y=0.995,
)
plt.show()

# %% [markdown]
# **Interpretation.**
#
# - Model 3's SSP latent (black) captures **slow drift in relative
#   well preference after the perseveration and win-stay covariates
#   are partialled out.** If it's flat, the data's preference structure
#   is fully explained by the Frank-lab-style covariates — no residual.
#   If it drifts systematically (e.g., well 5's value rising across
#   trials), there's a time-varying bias that no mechanism in §2
#   captures.
# - **Q-learner's pinned Q (dashed red)** shows the reward-EWMA-only
#   story. Where it matches the black line, reward integration explains
#   the residual drift — and by implication, the SSP latent was just
#   standing in for Q-learning. Where they diverge, reward integration
#   is wrong.
# - **Beta-Bernoulli's pinned E[p] (dashed purple)** shows the
#   cumulative-Bayesian story. Its characteristic signature is being
#   *steeper* than Q-learner near the start (low prior counts) and
#   *flatter* later (high counts resist updates).
# - The most striking pattern to look for: **the black line drifting
#   steadily in one direction across the session without sharp jumps
#   at rewards**. That would be motivational / satiation drift —
#   something no reward-based mechanism can generate.

# %% [markdown]
# ## § 4.4 Take-homes for the SSP ↔ Frank-lab comparison
#
# - **Predictive agreement tells you when the debate is adjudicable.**
#   If the top-right block of the KL heatmap is near zero, the two
#   frameworks give indistinguishable predictions on this session;
#   claiming one is "correct" would be unfalsifiable. If it's large,
#   the frameworks tell meaningfully different stories and §4.2 / §4.3
#   are worth reading.
# - **Trajectory overlay diagnoses mechanism failure.** Disagreement
#   between Frank-lab's Q(t) and SSP's smoothed value at a particular
#   well / time is a specific, interpretable claim: "the reward-driven
#   mechanism can't produce the preference pattern the data prefers
#   here".
# - **Residual analysis isolates the unexplained.** Model 3's SSP
#   latent after covariates is a picture of "what the data asks for
#   that no mechanism in the Frank-lab catalog can produce".  That's
#   where new mechanistic hypotheses should live.
# - **Caveats.** This entire comparison is on training data (no
#   held-out split) and single-seed fits. The LL-on-training Model 3
#   improvement of ~190 nats needs a held-out validation before it
#   can be reported as a real finding rather than a fit to session-
#   specific idiosyncrasies. Repeating §4 on a held-out session (e.g.,
#   `04_r2`) with 02_r1-fitted parameters would tell us whether the
#   residual SSP latent represents a general behavioral feature or a
#   session-specific artifact.

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
# Helpers first: patch-change and reward-marker overlays, shared across
# the trial-level plots.

# %%
patch_change_trials = trial_num[is_patch_change]
rewarded_trials = trial_num[is_reward == 1]


def _overlay_task_marks(ax: plt.Axes) -> None:
    """Overlay patch-change verticals + rewarded-trial triangles on a trial-axis plot."""
    for tc in patch_change_trials:
        ax.axvline(tc, color="k", lw=0.5, alpha=0.3, ls="--", zorder=1)
    ax.plot(
        rewarded_trials,
        np.full_like(rewarded_trials, 0.98, dtype=float),
        marker="v",
        color="g",
        markersize=3,
        linestyle="none",
        transform=ax.get_xaxis_transform(),
        zorder=5,
    )


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
# Values are relative to the reference option (well 0), which is fixed
# at 0 by the softmax identifiability constraint — **not** because of
# sparse visits. The other five traces show each well's relative value.
# Vertical dashed lines = patch changes. Green triangles at the top =
# rewarded trials.

# %%
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
colors = plt.cm.tab10(np.arange(n_wells))

for ax, smoothed, title in [
    (axes[0], smoothed_1a, "1a Multinomial"),
    (axes[1], smoothed_1b, "1b Covariate (+ prev reward)"),
]:
    for well in range(n_wells):
        label = f"well {well}" + (" (reference)" if well == 0 else "")
        ls = ":" if well == 0 else "-"
        ax.plot(
            trial_num,
            smoothed[:, well],
            color=colors[well],
            lw=1.2,
            ls=ls,
            label=label,
        )
    _overlay_task_marks(ax)
    ax.set_ylabel("latent option value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

axes[0].legend(fontsize=7, ncol=6, loc="lower center", frameon=False)
axes[-1].set_xlabel("trial number")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1c vs 1d — discrete state posteriors
# 1c: posterior over behavioral strategies. 1d: posterior over hidden
# rule/contingency states. Dashed lines = patch changes.

# %%
fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)

axes[0].stackplot(
    trial_num,
    smoothed_disc_1c.T,
    labels=[f"strategy {s}" for s in range(smoothed_disc_1c.shape[1])],
    alpha=0.8,
)
axes[0].set_ylabel("P(strategy)")
axes[0].set_title("1c Switching — smoothed strategy posterior")
axes[0].set_ylim(0, 1)
axes[0].legend(fontsize=8, loc="upper right")
for tc in patch_change_trials:
    axes[0].axvline(tc, color="k", lw=0.5, alpha=0.4, ls="--")

axes[1].stackplot(
    trial_num,
    smoothed_belief_1d.T,
    labels=[f"rule state {s}" for s in range(smoothed_belief_1d.shape[1])],
    alpha=0.8,
)
axes[1].set_ylabel("P(rule state)")
axes[1].set_xlabel("trial number")
axes[1].set_title("1d Contingency belief — smoothed posterior")
axes[1].set_ylim(0, 1)
axes[1].legend(fontsize=8, loc="upper right")
for tc in patch_change_trials:
    axes[1].axvline(tc, color="k", lw=0.5, alpha=0.4, ls="--")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1e Smith per-well smoothed reward probability

# %%
fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True, sharex=True)
axes_flat = axes.ravel()

for well in range(n_wells):
    ax = axes_flat[well]
    if well not in smith_results:
        ax.set_title(f"well {well} — insufficient visits")
        ax.axis("off")
        continue
    r = smith_results[well]
    visit_trials = trial_num[r["visit_indices"]]
    ax.plot(
        visit_trials,
        r["smoothed_prob"],
        color="C0",
        lw=1.5,
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
        color="g",
        s=15,
        zorder=5,
    )
    ax.scatter(
        unrewarded_here,
        np.full_like(unrewarded_here, -0.02, dtype=float),
        marker="x",
        color="r",
        s=15,
        zorder=5,
    )
    ax.set_title(f"well {well}  (n={r['n_visits']})")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    if well in (0, 3):
        ax.set_ylabel("P(reward)")
    if well >= 3:
        ax.set_xlabel("trial number")

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
#   init (both states starting at β=1.0), this model collapses to an
#   unidentifiable symmetric fit on 180 trials: identical inverse
#   temperatures, identical process noise, symmetric transition matrix,
#   LL ≈ 0 from the softmax becoming deterministic. The current cell uses
#   `init_inverse_temperatures=[0.5, 5.0]` to break the symmetry. After
#   fitting you should see clearly distinct states — e.g., one with low
#   β (soft/exploratory, fast-switching) and one with high β
#   (hard/exploitative, sticky self-transition). If the two rows of
#   `discrete_transition_matrix_` look symmetric, or if the two
#   `inverse_temperatures_` come back identical, the init wasn't enough
#   to escape the symmetric basin — try a more extreme split, add L2
#   regularization on β, or reduce `num_steps`. If the data truly
#   doesn't support two distinct regimes, SGD will still drag them
#   together even from an asymmetric init.
#
# - **1d**: `reward_probs_` describes the latent model's belief about each
#   state's reward structure per well. Compare to the true patch layout.
#
# - **Log-likelihoods in the summary table are NOT all comparable.** 1a/1b
#   share the same observation model (choices only). 1c uses choices but
#   conditions on discrete states. 1d conditions on *both* choices and
#   rewards (more observations → more likelihood budget). 1e Smith runs
#   each use a different per-well dataset. Only compare 1a↔1b directly.
#
# Next: notebook 2 — place field model via `PlaceFieldModel.fit_sgd()` on
# 305 sorted CA1 units.

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
# # Notebook 1c — Choice-model parameter stability across the day
#
# `j1620210710_.nwb` has 7 run epochs (`02_r1` through `14_r7`),
# each ~20–25 minutes apart, performed across one experimental
# day. Notebook 01 fit choice models on epoch `02_r1` only; this
# notebook fits the same models **independently on each of the 7
# epochs** and asks how the parameters move across the day.
#
# **Why this matters.** Frank-lab's hierarchical EM averages over
# sessions — it explicitly *removes* this signal as nuisance. SSP
# fits are per-session, so they preserve it. If perseveration
# strengthens across the day (animal gets sated and increasingly
# avoids depleted wells), or learning rate drops (less reward-
# driven adjustment late in the day), or per-well preferences
# drift, those are real cognitive observations that the
# pooled-EM approach cannot see.
#
# Three model families fit per epoch:
#
# - **Q-learner** (Frank-lab parameterization, port from
#   `state_space_playground.frank_models`)
# - **Beta-Bernoulli** (Frank-lab parameterization, same module)
# - **SSP CovariateChoiceModel + Frank-lab covariates** = notebook
#   01 model 3

# %% [markdown]
# ## Setup

# %%
from state_space_playground.gpu import pick_free_gpu

pick_free_gpu(min_free_mb=20_000)

# %%
import logging
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from state_space_practice.covariate_choice import CovariateChoiceModel

from state_space_playground.frank_models import BetaBernoulliModel, QLearnerModel
from state_space_playground.session import load_session

logging.basicConfig(level=logging.WARNING)

EPOCHS = ["02_r1", "04_r2", "06_r3", "08_r4", "10_r5", "12_r6", "14_r7"]
N_WELLS = 6
WELL_COLORS = {
    0: "#1b7837", 1: "#7fbf7b",
    2: "#c51b7d", 3: "#de77ae",
    4: "#2166ac", 5: "#92c5de",
}
WELL_PATCH = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3}

# %% [markdown]
# ## Load all 7 epochs
#
# First call per epoch is ~2 min if not cached. Subsequent calls
# are sub-second.

# %%
sessions: dict[str, dict] = {}
for epoch in EPOCHS:
    t0 = time.time()
    sessions[epoch] = load_session(
        nwb_file_name="j1620210710_.nwb",
        epoch_name=epoch,
        use_sorted_hpc=True,
    )
    print(f"  {epoch}: {len(sessions[epoch]['trials']):4d} trials  "
          f"({time.time() - t0:.1f}s)")

# %% [markdown]
# ## Per-epoch trial arrays

# %%
def extract_arrays(session: dict) -> dict:
    trials = session["trials"]
    chosen = trials["to_well"].to_numpy().astype(np.int32)
    reward = trials["is_reward"].to_numpy().astype(np.int32)
    is_pc = trials["is_patch_change"].to_numpy()
    prev_r = np.concatenate([[0], reward[:-1]]).astype(np.float32)
    prev_w = np.concatenate([[0], chosen[:-1]]).astype(np.int32)
    return {
        "chosen": chosen, "reward": reward, "prev_reward": prev_r,
        "prev_well": prev_w, "is_patch_change": is_pc,
        "n_trials": len(trials),
    }


arrays = {epoch: extract_arrays(sessions[epoch]) for epoch in EPOCHS}

# %% [markdown]
# ## § 0. Behavioral overview — raw data across the day
#
# Before any modeling, look at how the basic behavior changes across
# the 7 run epochs. Total trials, reward rate, and well-visit
# distribution per epoch.

# %%
overview_rows = []
for epoch in EPOCHS:
    a = arrays[epoch]
    counts = np.bincount(a["chosen"], minlength=N_WELLS)
    overview_rows.append({
        "epoch": epoch,
        "n_trials": a["n_trials"],
        "n_rewards": int(a["reward"].sum()),
        "reward_rate": float(a["reward"].mean()),
        "n_patch_changes": int(a["is_patch_change"].sum()),
        "frac_repeated": float((a["chosen"][1:] == a["chosen"][:-1]).mean()),
        **{f"visits_w{w}": int(counts[w]) for w in range(N_WELLS)},
    })
overview = pd.DataFrame(overview_rows)
overview

# %% [markdown]
# ### Visualize trial counts and reward rates across epochs

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ax_n, ax_r, ax_pc = axes

x = np.arange(len(EPOCHS))
ax_n.bar(x, overview["n_trials"], color="C0", edgecolor="k", linewidth=0.5)
ax_n.set_xticks(x)
ax_n.set_xticklabels(EPOCHS, rotation=30)
ax_n.set_ylabel("# trials")
ax_n.set_title("Trials per epoch")
ax_n.grid(True, alpha=0.3, axis="y")

ax_r.bar(x, overview["reward_rate"], color="C2", edgecolor="k", linewidth=0.5)
ax_r.axhline(1 / 6, color="k", lw=0.8, ls="--", label="chance (1/6)")
ax_r.set_xticks(x)
ax_r.set_xticklabels(EPOCHS, rotation=30)
ax_r.set_ylabel("reward rate")
ax_r.set_title("Reward rate per epoch")
ax_r.legend(fontsize=8)
ax_r.grid(True, alpha=0.3, axis="y")

ax_pc.bar(x, overview["frac_repeated"], color="C3",
          edgecolor="k", linewidth=0.5, label="frac trials w/ repeat")
ax_pc2 = ax_pc.twinx()
ax_pc2.plot(x, overview["n_patch_changes"], "o-", color="C4", lw=1.5,
            label="# patch changes (right axis)")
ax_pc.set_xticks(x)
ax_pc.set_xticklabels(EPOCHS, rotation=30)
ax_pc.set_ylabel("frac trials with same well as prev", color="C3")
ax_pc2.set_ylabel("# patch changes", color="C4")
ax_pc.set_title("Repetition / switching behavior")
ax_pc.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.show()

# %% [markdown]
# **What to look for.**
#
# - **Reward rate** trending down across epochs would say the animal
#   is getting *less* reward later in the day — could be a depleting
#   environment (the task config might reduce reward probabilities)
#   or could be the animal's choices getting worse.
# - **Repetition rate** trending up = increasing perseveration
#   (animal locks into preferred wells late in the day). Trending
#   down = increasing exploration.
# - **Patch changes** trending down = animal exploits within fewer
#   patches; trending up = animal tries more spatial regions.

# %% [markdown]
# ### Visit distribution per epoch
#
# Stacked bar showing what fraction of each epoch's trials went to
# each well.

# %%
visit_mat = np.array([
    [overview[f"visits_w{w}"].iloc[i] / overview["n_trials"].iloc[i] for w in range(N_WELLS)]
    for i in range(len(EPOCHS))
])  # (n_epochs, n_wells)

fig, ax = plt.subplots(figsize=(11, 4))
bottom = np.zeros(len(EPOCHS))
for w in range(N_WELLS):
    ax.bar(x, visit_mat[:, w], bottom=bottom, color=WELL_COLORS[w],
           edgecolor="k", linewidth=0.4, label=f"well {w} (p{WELL_PATCH[w]})")
    bottom = bottom + visit_mat[:, w]
ax.set_xticks(x)
ax.set_xticklabels(EPOCHS)
ax.set_ylabel("fraction of trials")
ax.set_title("Per-well visit distribution across epochs")
ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Reading.** Each colored stack = the within-epoch fraction of
# trials at that well. If the visit distribution is **stable across
# epochs**, the animal has a consistent spatial preference that
# doesn't depend on within-day learning. If it shifts (e.g.,
# patch 3 dominates early but patch 1 grows later), the animal is
# doing something that within-epoch models can't capture by design.

# %% [markdown]
# ### Reward raster across all 7 epochs (concatenated time)

# %%
fig, axes = plt.subplots(len(EPOCHS), 1, figsize=(14, 9), sharex=False)
for ax, epoch in zip(axes, EPOCHS):
    a = arrays[epoch]
    chosen, reward = a["chosen"], a["reward"]
    t = np.arange(len(chosen))
    for w in range(N_WELLS):
        mask = chosen == w
        t_w = t[mask]
        r_w = reward[mask]
        ax.scatter(t_w[r_w == 0], np.full((r_w == 0).sum(), w, dtype=float),
                   marker="o", s=10, facecolors="none",
                   edgecolors=WELL_COLORS[w], linewidths=0.6)
        ax.scatter(t_w[r_w == 1], np.full((r_w == 1).sum(), w, dtype=float),
                   marker="o", s=14, color=WELL_COLORS[w])
    ax.set_yticks(range(N_WELLS))
    ax.set_yticklabels([f"w{w}" for w in range(N_WELLS)], fontsize=7)
    ax.set_ylim(-0.5, N_WELLS - 0.5)
    ax.invert_yaxis()
    ax.set_ylabel(epoch, fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
axes[-1].set_xlabel("trial number within epoch")
fig.suptitle("Reward raster per epoch (filled = rewarded, open = unrewarded)",
             y=0.995, fontsize=11)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
#
# # § 1. Per-epoch fits — three model families
#
# Each model is fit independently on each epoch's trial sequence
# via SGD. We use the same parameterizations as notebook 01:
#
# - **Q-learner**: `α, β, init_Q, stay_bias` (4 params)
# - **Beta-Bernoulli**: `β, decay, a_baseline, stay_bias` (4 params)
# - **SSP+Frank covariates**: dynamics-time `prev_reward` + obs-time
#   `(prev_well_oh, win_stay_oh)` → 1 dyn weight + 12 obs cols ×
#   K=6 obs weights = 73 params + 5 dynamics-state params

# %%
def build_obs_covariates(a: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_t = a["n_trials"]
    prev_reward = a["prev_reward"]
    prev_well = a["prev_well"]
    prev_well_oh = np.eye(N_WELLS, dtype=np.float32)[prev_well]
    win_stay = (prev_reward[:, None] * prev_well_oh).astype(np.float32)
    obs = np.hstack([prev_well_oh, win_stay]).astype(np.float32)
    return prev_reward[:, None], prev_well_oh, obs


fits: dict[str, dict] = {}
for epoch in EPOCHS:
    a = arrays[epoch]
    chosen_jnp = jnp.asarray(a["chosen"])
    reward_jnp = jnp.asarray(a["reward"])
    dyn_cov, _, obs_cov = build_obs_covariates(a)
    print(f"=== {epoch} ({a['n_trials']} trials) ===")

    ql = QLearnerModel()
    ql_lls = ql.fit_sgd(chosen_jnp, reward_jnp, num_steps=400, verbose=False)
    print(f"  Q-learner       LL={ql_lls[-1]:8.2f}  "
          f"α={ql.alpha_:.3f}, β={ql.beta_:.2f}, "
          f"init_Q={ql.init_Q_:.3f}, stay_bias={ql.stay_bias_:.3f}")

    bb = BetaBernoulliModel()
    bb_lls = bb.fit_sgd(chosen_jnp, reward_jnp, num_steps=400, verbose=False)
    print(f"  Beta-Bernoulli  LL={bb_lls[-1]:8.2f}  "
          f"β={bb.beta_:.2f}, decay={bb.decay_:.3f}, "
          f"stay_bias={bb.stay_bias_:.3f}")

    ssp = CovariateChoiceModel(n_options=N_WELLS, n_covariates=1, n_obs_covariates=12)
    ssp_lls = ssp.fit_sgd(
        chosen_jnp,
        covariates=jnp.asarray(dyn_cov),
        obs_covariates=jnp.asarray(obs_cov),
        num_steps=400,
        verbose=False,
    )
    print(f"  SSP+Frank cov   LL={ssp_lls[-1]:8.2f}  "
          f"input_gain={np.asarray(ssp.input_gain_).ravel().round(2).tolist()}")

    fits[epoch] = {
        "n_trials": a["n_trials"],
        "ql": {"model": ql, "lls": ql_lls,
               "params": {"alpha": ql.alpha_, "beta": ql.beta_,
                          "init_Q": ql.init_Q_, "stay_bias": ql.stay_bias_}},
        "bb": {"model": bb, "lls": bb_lls,
               "params": {"beta": bb.beta_, "decay": bb.decay_,
                          "a_baseline": bb.a_baseline_, "stay_bias": bb.stay_bias_}},
        "ssp": {"model": ssp, "lls": ssp_lls},
    }

# %% [markdown]
# ## § 2. Q-learner parameter stability across epochs

# %%
ql_params_df = pd.DataFrame([
    {"epoch": e, **fits[e]["ql"]["params"]} for e in EPOCHS
])
ql_params_df

# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, name in zip(axes, ["alpha", "beta", "init_Q", "stay_bias"]):
    ax.plot(x, ql_params_df[name], "o-", color="C0", lw=1.6, markersize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(EPOCHS, rotation=30, fontsize=8)
    ax.set_title(f"Q-learner: {name}", fontsize=10)
    ax.grid(True, alpha=0.3)
    if name == "alpha":
        ax.set_ylabel("learning rate")
        ax.set_ylim(-0.05, 1.05)
    elif name == "beta":
        ax.set_ylabel("inverse temperature")
        ax.set_yscale("log")
    elif name == "init_Q":
        ax.set_ylabel("initial Q")
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.set_ylabel("perseveration logit")
        ax.axhline(0, color="k", lw=0.6)
fig.suptitle("Q-learner parameters across the day (j1620210710_)", y=1.04)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpretation.**
#
# - **`alpha` near 0 across all epochs**: the animal is *not*
#   reward-driven across the day — Q updates carry no signal. This
#   confirms the notebook 01 finding (α≈0 in 02_r1) at the level of
#   the whole experimental day, not just one session.
# - **`alpha` rising or falling with epoch**: would suggest that
#   the animal's reward sensitivity changes within the day. Rising
#   = "warming up to the task". Falling = "satiation / disengagement".
# - **`beta` huge (>50)**: the model collapsed to a near-deterministic
#   solution (typically when the data has strong perseveration that
#   `stay_bias` alone can capture). β on log scale to make this
#   readable.
# - **`stay_bias` getting more negative**: increasing avoidance of
#   the just-visited well — strengthening depletion behavior, possibly
#   because the animal increasingly conditions on internal "I just
#   went there" signal.

# %% [markdown]
# ## § 3. Beta-Bernoulli parameter stability across epochs

# %%
bb_params_df = pd.DataFrame([
    {"epoch": e, **fits[e]["bb"]["params"]} for e in EPOCHS
])
bb_params_df

# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, name in zip(axes, ["beta", "decay", "a_baseline", "stay_bias"]):
    ax.plot(x, bb_params_df[name], "o-", color="C4", lw=1.6, markersize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(EPOCHS, rotation=30, fontsize=8)
    ax.set_title(f"Beta-Bernoulli: {name}", fontsize=10)
    ax.grid(True, alpha=0.3)
    if name == "beta":
        ax.set_yscale("log")
        ax.set_ylabel("inverse temperature")
    elif name == "decay":
        ax.set_ylabel("β-decay (toward prior)")
        ax.set_ylim(-0.02, 1.02)
    elif name == "a_baseline":
        ax.set_ylabel("prior pseudo-count")
    else:
        ax.set_ylabel("perseveration logit")
        ax.axhline(0, color="k", lw=0.6)
fig.suptitle("Beta-Bernoulli parameters across the day", y=1.04)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpretation.**
#
# - **`decay` near 0**: belief evidence is treated as cumulative —
#   the model integrates rewards across the entire epoch with no
#   forgetting. **`decay` near 1**: each trial nearly resets the
#   posterior — no learning. A meaningful intermediate value (e.g.,
#   0.05–0.3) would indicate genuine within-epoch updating with a
#   reasonable forgetting timescale.
# - **`stay_bias`** comparable to Q-learner's. Both models tend to
#   route the depletion/perseveration signal through this scalar.

# %% [markdown]
# ## § 4. SSP CovariateChoiceModel — per-well perseveration / win-stay across epochs
#
# Plot the 6 per-well perseveration *diagonals* and 6 per-well
# win-stay *diagonals* of `obs_weights_` as functions of epoch.
# These are the SSP analogs of the Frank-lab `stay_bias` scalar but
# split per-well.

# %%
def ssp_diagonals(model: CovariateChoiceModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    W = np.asarray(model.obs_weights_)  # (K, 12)
    persev_diag = np.diag(W[:, :N_WELLS])
    winstay_diag = np.diag(W[:, N_WELLS:])
    input_gain = np.asarray(model.input_gain_).ravel()  # (K-1,)
    return persev_diag, winstay_diag, input_gain


persev_mat = np.zeros((len(EPOCHS), N_WELLS))
winstay_mat = np.zeros((len(EPOCHS), N_WELLS))
input_gain_mat = np.zeros((len(EPOCHS), N_WELLS - 1))
for i, epoch in enumerate(EPOCHS):
    pd_, wd_, ig_ = ssp_diagonals(fits[epoch]["ssp"]["model"])
    persev_mat[i] = pd_
    winstay_mat[i] = wd_
    input_gain_mat[i] = ig_

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
ax_per, ax_ws, ax_ig = axes

for w in range(N_WELLS):
    ax_per.plot(x, persev_mat[:, w], "o-", color=WELL_COLORS[w], lw=1.4,
                label=f"w{w} (p{WELL_PATCH[w]})")
    ax_ws.plot(x, winstay_mat[:, w], "o-", color=WELL_COLORS[w], lw=1.4,
               label=f"w{w}")
ax_per.axhline(0, color="k", lw=0.6)
ax_per.set_xticks(x); ax_per.set_xticklabels(EPOCHS, rotation=30, fontsize=8)
ax_per.set_ylabel("perseveration logit shift")
ax_per.set_title("Per-well perseveration (diag) across epochs")
ax_per.legend(fontsize=7, ncol=2, loc="best")
ax_per.grid(True, alpha=0.3)

ax_ws.axhline(0, color="k", lw=0.6)
ax_ws.set_xticks(x); ax_ws.set_xticklabels(EPOCHS, rotation=30, fontsize=8)
ax_ws.set_ylabel("win-stay logit shift")
ax_ws.set_title("Per-well win-stay (diag) across epochs")
ax_ws.legend(fontsize=7, ncol=2, loc="best")
ax_ws.grid(True, alpha=0.3)

# Input gain is K-1 long: prepend NaN for the reference well 0
ig_full = np.full((len(EPOCHS), N_WELLS), np.nan)
ig_full[:, 1:] = input_gain_mat
for w in range(N_WELLS):
    ax_ig.plot(x, ig_full[:, w], "o-", color=WELL_COLORS[w], lw=1.4,
               label=f"w{w}" + (" (ref)" if w == 0 else ""))
ax_ig.axhline(0, color="k", lw=0.6)
ax_ig.set_xticks(x); ax_ig.set_xticklabels(EPOCHS, rotation=30, fontsize=8)
ax_ig.set_ylabel("dynamics gain on prev_reward")
ax_ig.set_title("input_gain (dynamics-time prev_reward) across epochs")
ax_ig.legend(fontsize=7, ncol=2, loc="best")
ax_ig.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpretation.**
#
# - **Perseveration diag**: in notebook 01 we saw -1.8 to -2.2
#   uniformly across wells (depletion). If those numbers stay tightly
#   bunched and around the same value across epochs, depletion is a
#   stable behavioral feature. If they spread or drift, depletion
#   strength is changing — possibly with satiation.
# - **Win-stay diag**: notebook 01 found these mostly negative
#   (anti-win-stay). If they stay negative across epochs, the
#   anti-win-stay pattern is robust. If some epochs show positive
#   win-stay, the strategy genuinely changes within the day.
# - **Input gain on prev_reward**: how much last trial's reward
#   shifts the latent value at each well. If the per-well gains
#   converge across epochs, the dynamics-time reward integration is
#   consistent. If wells 4/5 (the most-visited) show consistent
#   positive gain while others fluctuate, that's a stable preferred-
#   patch effect.

# %% [markdown]
# ## § 5. Model comparison across epochs
#
# Per-trial log-likelihood (LL ÷ n_trials) so epochs of different
# lengths are comparable. Higher = better.

# %%
ll_table = pd.DataFrame([
    {
        "epoch": e,
        "n_trials": fits[e]["n_trials"],
        "uniform": float(np.log(1 / N_WELLS)),
        "Q-learner": float(fits[e]["ql"]["lls"][-1]) / fits[e]["n_trials"],
        "Beta-Bernoulli": float(fits[e]["bb"]["lls"][-1]) / fits[e]["n_trials"],
        "SSP+Frank cov": float(fits[e]["ssp"]["lls"][-1]) / fits[e]["n_trials"],
    }
    for e in EPOCHS
])
ll_table.round(3)

# %%
fig, ax = plt.subplots(figsize=(11, 4.5))
for col, color, marker in [
    ("Q-learner", "C3", "s"),
    ("Beta-Bernoulli", "C4", "^"),
    ("SSP+Frank cov", "k", "o"),
]:
    ax.plot(x, ll_table[col], marker + "-", color=color, lw=1.6,
            markersize=7, label=col)
ax.axhline(np.log(1 / N_WELLS), color="0.5", lw=0.8, ls="--",
           label=f"uniform 6-way ({np.log(1/N_WELLS):.2f})")
ax.set_xticks(x)
ax.set_xticklabels(EPOCHS, rotation=30)
ax.set_ylabel("log-likelihood per trial")
ax.set_title("Per-trial LL by model and epoch")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Reading.**
#
# - **SSP+Frank cov should sit far above the others on every epoch**
#   if the notebook 01 finding (latent + covariates >> covariates
#   alone) generalizes across the day.
# - **Within-epoch model ranking stable across epochs** = the choice
#   of best model is a property of *the task and animal*, not a
#   property of *which session you happen to look at*.
# - **Within-epoch model ranking flips between epochs** = there's a
#   regime change somewhere in the day where one model becomes more
#   appropriate. Cross-reference with the parameter trajectories
#   above to see what shifted.
# - The **gap between the per-trial LL and uniform** measures how
#   much *more* than chance the model explains. A constant gap
#   across epochs = consistent behavior; a shrinking gap late in the
#   day = the animal becoming "harder to predict" as fatigue /
#   satiation kicks in.

# %% [markdown]
# ## § 6. Take-homes
#
# - The Frank-lab models' **single per-day parameter** for `α`,
#   `decay`, `volatility`, etc. (a consequence of hierarchical EM
#   with one prior across sessions) implicitly assumes parameter
#   stability across epochs. This notebook gives a model-free
#   visualization of whether that assumption holds.
# - For *this animal* (`j1620210710_`), the most informative
#   parameters to watch for stability are:
#   1. Q-learner's **`stay_bias`** — depletion-strength proxy.
#   2. Beta-Bernoulli's **`decay`** — does the animal forget?
#   3. SSP+cov's **per-well perseveration diag** — same depletion
#      story but split per well, so you can see *which* wells'
#      depletion behavior dominates.
# - If you want to add a within-day covariate to explain parameter
#   drift (e.g., epoch index as a nuisance regressor), the natural
#   place is `n_obs_covariates += 1` with a per-trial scalar
#   carrying the within-day phase. That's the next step if any of
#   the parameter trajectories above show non-trivial structure.

# Notebook 2 — Place Field Model — WIP status

Picking this up after upstream fixes in `state_space_practice` and a
subsequent PSD-loss investigation that unblocked us via an f64 workaround.
This file is the handoff note so we can resume without re-deriving
everything.

## What's resolved vs still open (as of the f64 investigation)

**Resolved upstream:**
- `bin_spike_times` boundary bug — fixed. Now right-bounds the last
  bin and emits a `UserWarning` when spikes fall outside the window.
- Default `init_cov_scale=0.01`, `init_process_noise=1e-6`,
  `max_firing_rate_hz=500` — all shipped.
- `warm_start=True` default with built-in stationary Poisson GLM —
  shipped. Makes `fit_sgd` usable from any `fit_sgd(position, spikes)`
  call without caller-side warm-start.
- SGD saturation warning (`_warn_if_rate_saturated`) — shipped.

**Discovered after the upstream fix:**
- The `stochastic_point_process_filter` Kalman-style covariance
  propagation **loses PSD on `pred_cov` in f32** after enough bins
  when `init_cov` is even mildly ill-conditioned. This manifests as
  silent NaN outputs in two regimes:
  - **Problem A** — `warm_start=False, n_knots=3, T=1501`: NaN at
    `fit_sgd` step 11
  - **Problem B** — `warm_start=True, n_knots=5, T=28k`: NaN at
    `fit_sgd` step 2
  Root cause is `symmetrize + psd_solve` with `diagonal_boost=1e-9` in
  f32 — classic Kalman numerical stability issue. See the upstream
  proposal doc at `/tmp/state_space_practice_psd_loss_proposal.md` for
  the full trace and fix options.

**Workaround (this playground's current solution):**
- **f64 empirically eliminates both failures.** Problem A runs 30 clean
  SGD steps under f64 (LL -2634 → -1473). Problem B runs 10 clean
  steps (LL -26k → -16k). Runtime cost is ~40% slower per step, not
  2×. `pick_free_gpu()` now calls `jax.config.update("jax_enable_x64",
  True)` by default, so all playground notebooks run in f64
  automatically. Opt out via `pick_free_gpu(enable_x64=False)` only if
  you have a specific reason (e.g., benchmarking a Joseph-form fix).

## Current state of notebook 2

- `notebooks/02_place_field_model.py` is **uncommitted** on disk in its
  initial draft form. It is **not safe to run** as written — see
  problems list below. Either rewrite it in place before running, or
  delete it and start fresh from this plan.
- Plumbing from notebook 1 (`gpu.py`, `session.py`, `plot.py`, `eval.py`,
  `scripts/save_session.py`) is already in place and working.
- Session cache is primed at
  `cache/sessions/j1620210710_02_r1_sortedHPC.pkl` (415 MB). Warm
  `load_session(...)` takes ~0.3 s.
- `notebooks/01_choice_models.py` is committed and working (commit
  `d5e8581`).

## Problems with the current draft (why it's not safe to run)

Most of these were real at the time of the initial draft. After the
upstream bin_spike_times / defaults fixes and our f64 workaround, items
2 and 3 are gone. Items 1, 4, and 5 still need to be fixed in the
rewrite.

1. **Wrong unit selection**: picks top-20 by raw firing rate. These are
   almost all interneurons (rate > 5 Hz, no spatial tuning). Need to
   filter to pyramidal candidates first: `rate < 5 Hz AND n_spikes ≥ 200`.
   On this session that gives 155 candidates out of 305 units.
   **(Still needs fixing.)**
2. ~~Hits the `bin_spike_times` right-unbounded-last-bin bug~~ —
   **RESOLVED UPSTREAM.** `bin_spike_times` now right-bounds the last
   bin and emits a `UserWarning` when spikes fall outside.
3. ~~No warm-start~~ — **RESOLVED UPSTREAM.** `fit_sgd` now has
   `warm_start=True` by default, which runs a stationary Poisson GLM
   Laplace approximation internally. Also `init_cov_scale=0.01` and
   `init_process_noise=1e-6` are the new defaults.
4. **Runtime is still a constraint** but much better: measured
   ~20 s/step at T=28k, d=36 under f32; ~28 s/step under f64. At d=64
   (n_interior_knots=5, the library default), ~17 s/step f32 → NaN,
   ~28 s/step f64 works. So in f64, 5 cells × 20 steps × 28 s ≈ 47 min.
   10 cells ≈ 93 min. 20 cells ≈ 187 min. Manageable, not instant.
5. **300 SGD steps is still excessive** — warm-start puts us near the
   MLE already. 10–30 steps should suffice. **(Still needs fixing.)**

## Decisions already made (don't re-litigate)

| Decision | Value | Rationale |
|---|---|---|
| Session | `j1620210710_.nwb / 02_r1` | The goldmine — 305 sorted CA1 + 74 sorted mPFC units, 7 run epochs. Same session the whole playground uses. |
| Precision | **f64** | `pick_free_gpu()` enables `jax_enable_x64=True` by default. Required to avoid the Kalman PSD-loss pathology in the upstream filter. ~40% slower per step, fully eliminates the NaN. |
| Unit cell-type filter | `rate < 5 Hz AND n_spikes >= 200` | Pyramidal candidates only. 155 out of 305 pass. |
| Unit ranking within candidates | by total spike count, descending | Higher SNR for place field estimation |
| Time bin width `dt` | 50 ms (T ≈ 28k) | Good fidelity / runtime tradeoff. Per-step cost under f64 is ~28s at T=28k. |
| Spatial basis | `n_interior_knots = 5` (64 basis, library default) | We can use the default now that f64 handles the conditioning — smaller basis is only needed for f32 workarounds, which we don't need. |
| Warm-start strategy | Built-in `fit_sgd(warm_start=True)` (library default) | Uses `_warm_start_parameters` which does a stationary Poisson GLM Laplace approximation for both `init_mean` and `init_cov`. No notebook-side warm-start needed. |
| `init_cov_scale` | `0.01` (library default) | Used by warm_start to set P0's scale before the GLM adjustment. The GLM then overwrites with the Laplace inverse-Hessian. |
| `init_process_noise` | `1e-6` (library default) | Principled value from the upstream docs. Note: we looked at 1e-7 earlier based on a rougher biological argument, but the library default works fine under f64 and is simpler to defend. |
| Fit per neuron vs. multi-neuron block-diagonal | Per neuron | For 305 units × 64 basis, multi-neuron state dim is ~19,520; the Kalman covariance is then `(T, 19520, 19520)` which is hundreds of terabytes. Per-neuron is block-diagonally equivalent mathematically but fits in memory. |
| SGD step update flags | Library defaults (`update_init_state=True`, `update_process_cov=True`) | Under f64 there's no reason to freeze these. Verified empirically that SGD converges cleanly from warm-start in f64. |
| SGD step count | 20 steps per unit | Warm-start puts us near the optimum; 20 steps is plenty for refinement. |

## Decisions still open

1. **How many cells to fit in the first pass?** Options:
   - 5 cells: ~50 min runtime at current per-step cost. Enough for a first figure.
   - 10 cells: ~2 h.
   - 20 cells: ~5 h.
   - 155 cells (all pyramidal candidates): overnight+.

   Leaning 5 for the initial notebook-1-style commit, then scale later.
   **Revisit after upstream performance fix lands** — if per-step cost
   drops 10×, 20 cells in <1 h becomes reasonable.

2. **Whether to wait for the upstream fix before committing anything.**
   Options:
   - **(a)** Wait for upstream fix, then rewrite the notebook to use the
     fixed `bin_spike_times` (or whatever API lands) and clean defaults.
   - **(b)** Write the notebook now with the caller-side workarounds
     (pre-filter spikes, explicit warm-start, tight priors) and commit
     it. Later, simplify once upstream ships.
   - **(c)** Write the notebook as a scratch script in `/tmp/`, run it
     end-to-end on 5 cells to validate the full pipeline, don't commit
     yet. Commit after upstream lands.

   Currently leaning **(c)** — proves the pipeline works without
   committing code we'll immediately have to rewrite.

3. **EM vs SGD**. The user's plan is SGD throughout. But the EM `.fit()`
   path is probably subject to the same blowup mechanism (same filter
   under the hood). If upstream fixes the filter, both paths should
   work. If upstream only fixes the SGD path, we need to verify EM
   separately. Defer until upstream lands.

## Concrete pickup path

When upstream has landed and we're ready to resume:

1. **Pull upstream changes and re-sync venv**:
   ```bash
   uv lock --project /cumulus/edeno/state-space-playground --upgrade-package state_space_practice
   uv sync --project /cumulus/edeno/state-space-playground
   ```
2. **Re-run the warm-start verification experiment** to confirm the
   upstream fix produces sane LLs without the caller-side workaround.
   Reference script: `/tmp/warm_start_exp.py` (may need recreation).
   Expected result: default `PlaceFieldModel.fit_sgd(position, spikes)`
   on unit 133 with 1500 bins should give first LL around -1500, not -4.86e8.
3. **Benchmark per-step runtime** at the target config (T=28k, d=36,
   n_basis_per_neuron from `n_interior_knots=3`). Expected:
   - Before upstream perf fix: ~40 s/step → infeasible
   - After upstream perf fix (if one lands): target < 5 s/step
4. **Write the notebook** using the plan below.
5. **Smoke test** on 1 cell end-to-end.
6. **Full run** on N cells (N decided based on per-step cost).
7. **Code review via scientific-code-reviewer**.
8. **Commit + push**.

## The notebook plan itself (when we're ready to write it)

```
notebooks/02_place_field_model.py — jupytext percent format

# %% [markdown] header
# Title, model statement (x_t = x_{t-1} + w; y ~ Poisson(exp(Z@x)*dt)),
# use case (time-varying place fields), architecture choice
# (per-neuron fits because multi-neuron is memory-infeasible), warm-start
# rationale (stationary Poisson GLM → init_mean), biological priors
# (init_cov_scale=0.01, init_process_noise=1e-7).

# %% Setup — pick_free_gpu() FIRST, then imports
# (same template as notebook 1; gpu → session → state_space_practice)

# %% Load session, filter units
# is_pyr = (rate < 5) & (n_spikes >= 200)
# pick top-N by spike count

# %% Build time bins, downsample position, bin spikes PER UNIT
# NOTE: pre-filter spike_times to the window before bin_spike_times
# (or, if upstream has fixed bin_spike_times, just call it directly)

# %% Occupancy map — context plot

# %% Helper: stationary Poisson GLM warm-start
# def glm_warm_start(Z, spikes, dt, n_steps=500, lr=0.1):
#     w = jnp.zeros(Z.shape[1])
#     opt = optax.adam(lr); state = opt.init(w)
#     def loss(w):
#         log_rate = Z @ w
#         return jnp.sum(jnp.exp(log_rate) * dt - spikes * log_rate)
#     gfn = jax.jit(jax.value_and_grad(loss))
#     for _ in range(n_steps):
#         l, g = gfn(w); u, state = opt.update(g, state, w)
#         w = optax.apply_updates(w, u)
#     return w  # shape (n_basis,)

# %% Per-unit fit loop
# for unit in top_units:
#     1. bin spikes for that unit
#     2. w_mle = glm_warm_start(Z_base, spikes, DT)
#     3. model = PlaceFieldModel(
#            dt=DT, n_interior_knots=3,
#            init_cov_scale=0.01, init_process_noise=1e-7,
#            update_init_state=False, update_process_cov=True,
#        )
#     4. model._build_design_matrix(xy)   # initializes basis
#     5. model.init_mean = w_mle  # or a cleaner API if one lands
#     6. lls = model.fit_sgd(xy, spikes, num_steps=20, verbose=False)
#     7. store model + lls

# %% Plots
# - SGD convergence curves (20 overlaid)
# - 2D rate maps, 4x5 grid of top-20 (from predict_rate_map)
# - Rate map credible intervals for a single example cell
# - Drift trajectories (centers over time) overlaid on arena
# - Drift summary table (total_drift, cumulative_drift per unit)

# %% Summary table
# Per unit: n_spikes, final_LL, GLM_LL, drift_cm, peak_rate
```

## Reference files

- **Notebook draft (uncommitted, not safe to run as-is)**:
  `notebooks/02_place_field_model.py`
- **Upstream proposal doc**:
  `/tmp/state_space_practice_proposal.md`
- **Experiment logs**:
  - `/tmp/smoke_pfm.log` — initial smoke test on unit 156 (interneuron),
    50 SGD steps in 2 h 13 min, first LL = -87982 (not even the worst
    case because this unit is a fast firer with reasonable total count)
  - `/tmp/smoke_pyr.log` — pyramidal smoke at small scale, exhibited the
    -4.85e8 LL
  - `/tmp/init_exp.log` — sweep of `init_cov_scale ∈ {1.0, 0.1, 0.01,
    0.001}` showing that tightening alone can't fix the blowup
  - `/tmp/warm_start.log` — the warm-start + biological Q sweep showing
    the working configuration (LL = -32,492)
  - `/tmp/investigate.log` — per-bin instrumented forward filter (first
    30 bins, confirms early bins are healthy)
  - `/tmp/find_blowup.log` — full forward pass, finds bin 1500 with 5926
    spikes and ||x||=435
  - `/tmp/verify_fix.log` — BUGGY vs FIXED `bin_spike_times` comparison
    showing LL goes from -485,822,478 to -1,488
  - `/tmp/jaxpr.log` — jax.make_jaxpr dump of `_point_process_laplace_update`
    with/without normalization, + clip isolation

## Headline numbers for context

| | Value |
|---|---|
| Session | j1620210710_.nwb / 02_r1, 23.6 min, 180 trials |
| Total HPC units | 305 (sorted via mountainsort4 + `franklab_tetrode_hippocampus_30KHz`) |
| Pyramidal candidates (`rate < 5 Hz`, `n ≥ 200`) | **155** |
| mPFC sorted units (bonus) | 74 |
| Arena | ~200 × 190 cm plus-maze |
| Fraction of time running (`speed > 4`) | 56.5% |
| Default fit config that **doesn't work** | `init_mean=0`, `P₀=I`, `Q=1e-5`, no warm-start, SGD from scratch |
| Working config | Warm-start from stationary Poisson GLM → `init_mean`, `P₀=0.01·I`, `Q=1e-7`, `update_init_state=False` |
| Stationary GLM alone, small problem (T=1501, d=36) | LL = -35,491 |
| SSM warm-started, same problem | LL = -32,492 (gain ~3000 from drift detection) |
| Default filter on same problem | LL = -485,822,478 |
| Per-step cost at T=71k, d=64 | ~155 s (upstream perf fix hopefully lands) |

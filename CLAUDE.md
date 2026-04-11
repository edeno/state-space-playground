# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project overview

This is a **playground** for experimenting with state-space models on neural and behavioral data from the spatial bandit task. It is intentionally scrappy — this is not production analysis code, and there is no paper riding on it. Feel free to add notebooks, experiment with different model formulations, and throw things away.

The data loading code is vendored from the sibling project [`continuum-swr-replay`](https://github.com/edeno/continuum-swr-replay). That project is the authoritative source for everything about the spatial bandit task, the NWB/Spyglass pipeline, and the structure of the loaded session dict. When in doubt, check its `CLAUDE.md`.

### What lives where

```text
state-space-playground/
├── pyproject.toml            # uv project; JAX CUDA 12, dynamax, state_space_practice, non_local_detector, spyglass
├── scripts/
│   └── check_gpu.py          # verifies JAX can see and use a GPU
└── src/state_space_playground/
    ├── __init__.py
    ├── data_loaders/         # VENDORED from continuum_swr_replay — do not treat as authoritative
    ├── parameters.py         # VENDORED — only exists to satisfy data_loaders imports
    └── types.py              # VENDORED — only exists to satisfy data_loaders imports
```

## Vendored code rules

`data_loaders/`, `parameters.py`, and `types.py` are **verbatim copies** from `continuum-swr-replay` as of the initial commit. Rules:

- **Do not refactor them casually.** Keep them byte-identical to upstream where possible so it's easy to re-sync.
- **If you fix a bug here,** note it so it can be ported upstream.
- **If you need to add a field** to a loaded data dict, prefer adding a wrapper function in new code over mutating the vendored loaders.
- **`parameters.py` is not a public API of this project** — you don't import from it directly in new playground code. It exists only because `data_loaders/utils.py` and `data_loaders/lfp.py` import from it via `from ..parameters import ...`.
- Same goes for `types.py` — the TypedDicts in it are used internally by the loaders. New playground code can use them if helpful but shouldn't add to them.

When upstream `continuum-swr-replay` drifts and you want to re-sync, the safe pattern is:

```bash
SRC=/cumulus/edeno/continuum-swr-replay/src/continuum_swr_replay
DST=/cumulus/edeno/state-space-playground/src/state_space_playground
cp -r "$SRC/data_loaders" "$DST/data_loaders"
cp "$SRC/parameters.py" "$DST/parameters.py"
cp "$SRC/types.py" "$DST/types.py"
```

Mirror the layout exactly — the loaders use relative imports like `from ..parameters import TrainingType`, which only work if `parameters.py` and `types.py` sit directly beside `data_loaders/`.

## Environment setup

This project uses [**uv**](https://docs.astral.sh/uv/) (not pip/conda). Python 3.11–3.12.

```bash
# runtime deps
uv sync

# with dev extras (jupyterlab, jupytext, ruff, pytest, mypy)
uv sync --extra dev
```

JAX is installed with the bundled CUDA 12 pip wheels per the [JAX install docs](https://github.com/jax-ml/jax/blob/main/docs/installation.md). **Do not** switch to `jax[cuda12-local]` — there is no system CUDA toolkit expected here, and the bundled wheels are the JAX team's recommended path.

## Running commands — gotchas

These pitfalls have each bitten us at least once. Keep them in mind.

- **Always use `uv run --project /cumulus/edeno/state-space-playground ...`** (or `cd` into the playground first). Without `--project`, uv walks up the filesystem looking for a `pyproject.toml` and may silently pick the sibling `continuum-swr-replay` venv — which has a different Python-package set (notably, it ships `numba`, which the playground only gets via the `track_linearization[opt]` extra). This produces confusing "works in one shell, fails in another" failures.

- **`pick_free_gpu()` must be the very first thing in every notebook**, before any other `state_space_playground` import. Importing `state_space_playground.session` transitively pulls in jax/jaxlib via the vendored data loaders (`session → data_loaders → spyglass → track_linearization → … → jax`), and once jax is loaded its CUDA backend is initialized and `CUDA_VISIBLE_DEVICES` has no effect. The guard in `gpu.py` raises a clear `RuntimeError` if you get the order wrong, but do yourself a favor and just put it at the top.

- **The playground runs in float64 by default.** `pick_free_gpu()` enables `jax_enable_x64=True` as a side effect — all JAX computations downstream default to f64. This is a deliberate choice, not a JAX default. Upstream `state_space_practice`'s Kalman-style point-process filter (`stochastic_point_process_filter` in `point_process_kalman.py`) propagates posterior covariance via `symmetrize + psd_solve` with `diagonal_boost=1e-9` in f32, and we observed it losing PSD after enough bins when `init_cov` is even mildly ill-conditioned — producing silent NaN outputs that look like SGD instability but are actually a numerical-precision floor. Running in f64 empirically eliminates both failure modes we found (`PlaceFieldModel.fit_sgd` step-11 NaN from a cold start, and step-2 NaN from a warm start on a long session) at a cost of ~40% slower per-step runtime. If you have a specific reason to use f32 — e.g., you're benchmarking a Joseph-form filter fix upstream — pass `pick_free_gpu(enable_x64=False)` and the function will explicitly set `jax_enable_x64=False` so the precision is deterministic. Full investigation and experimental evidence are captured in `notebooks/02_STATUS.md`.

- **Do not drop `[opt]` from `track_linearization[opt]` in `pyproject.toml`.** The module imports without numba, but its Viterbi decoder silently falls back to a Python path that is orders of magnitude slower. Anything that touches linearized position will mysteriously become glacial.

- **When running long scripts**, avoid piping to `| tail` / `| head` — those buffer stdout until the producer exits, so a hung process shows nothing. Use `> /tmp/foo.log 2>&1` + `run_in_background` instead, so progress is visible while the process runs.

### Verifying JAX GPU

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> uv run python scripts/check_gpu.py
```

The script sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` so it plays nicely on the shared GPU box. **Pick a GPU at runtime via `CUDA_VISIBLE_DEVICES`** — do not hard-code GPU pinning into analysis code, because this box is shared and free GPUs change.

Helper one-liner to find a free GPU:

```bash
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -n -r | head -1
```

## Loading data

Every script/notebook starts the same way: pin a GPU (which also enables f64), then load a session. The GPU step **must** come first — see the gotchas section above.

```python
# 1. Pin a GPU and enable f64 before importing anything else from
#    state_space_playground. pick_free_gpu() sets CUDA_VISIBLE_DEVICES,
#    disables XLA preallocation, and calls jax.config.update(
#    "jax_enable_x64", True). After this returns, jnp defaults to f64.
from state_space_playground.gpu import pick_free_gpu
pick_free_gpu()  # raises RuntimeError if jax/jaxlib is already loaded

# 2. Now it's safe to import session and the models.
from state_space_playground.session import load_session

data = load_session(
    nwb_file_name="j1620210710_.nwb",
    epoch_name="02_r1",
    use_sorted_hpc=True,
)
```

`load_session()` is a pickle cache on top of the vendored `data_loaders.load_data()`. First call for a given `(nwb, epoch, use_sorted_hpc)` triple takes ~2 min; subsequent calls are sub-second (reads `cache/sessions/*.pkl`). If you need to force a re-fetch, pass `force=True`. Run `scripts/save_session.py` once to prime the cache for the default session.

For direct, uncached access you can still call the underlying loader:

```python
from state_space_playground.data_loaders import load_data, get_epoch_info

epoch_info = get_epoch_info()
data = load_data(
    nwb_file_name="j1620210710_.nwb",
    epoch_name="02_r1",
    ripple_detector_name="Kay",
)
```

The returned dict is the same shape as in the parent project. Key entries:

- `position_info` — time-indexed `pd.DataFrame` with raw/projected/linear position
- `spike_times` — `dict[brain_area] -> list[np.ndarray]`; HPC is clusterless (per tetrode), mPFC/OFC are sorted units
- `spike_waveform_features` — HPC only; `dict[brain_area] -> list[np.ndarray]` of shape `(n_spikes, n_features)`
- `ripple_times` — `pd.DataFrame` with `start_time`, `end_time`
- `trials`, `well_visits`, `task_variables` — behavioral
- `theta_phase`, `theta_power`, `ripple_consensus_trace` — LFP-derived

See `src/state_space_playground/types.py::SessionData` for the full schema, and the parent project's `CLAUDE.md` for a tour.

### Sorted single units (only available for a few sessions)

The **default** HPC loader path uses clusterless waveform marks per tetrode. Most bandit sessions have **no sorted CA1 units** — just clusterless. Exceptions where CA1 was manually sorted with `mountainsort4` + `sorter_params_name="franklab_tetrode_hippocampus_30KHz"`:

| NWB file | Sorted CA1 units | Sort groups | Run epochs | Notes |
|---|---|---|---|---|
| **`j1620210710_.nwb`** | **1,239** | 21 (of 22 tetrodes) | 7 (`02_r1`…`14_r7`) | **Also has 74 sorted mPFC units** — cross-region single-unit experiments possible |
| `senor20201030_.nwb` | 9 | 1 tetrode | 7 (`02_r1`…`14_r7`) | Too thin for serious single-unit work |

**`j1620210710_.nwb` is the goldmine** for anything that needs sorted CA1 place cells. To load it:

```python
data = load_data(
    nwb_file_name="j1620210710_.nwb",
    epoch_name="02_r1",  # or 04_r2, 06_r3, 08_r4, 10_r5, 12_r6, 14_r7
    use_sorted_hpc=True,
)
# data["spike_times"]["HPC"]  → sorted CA1 units (~305 active in a 23-min run epoch)
# data["spike_times"]["mPFC"] → 74 sorted mPFC units (loaded automatically via the
#                               existing cortex sorter path — no extra flag needed)
# data["spike_waveform_features"] has NO "HPC" entry when use_sorted_hpc=True
```

Unit count per epoch is smaller than the raw total because `filter_spike_times` drops units that are silent in the requested position time window (≈75% of the 1,239 CA1 units don't fire during any given 23-min run epoch — this is normal CA1 epoch/context specificity).

**PFC/OFC LFP does not exist** in Spyglass for any bandit session. The LFP pipeline was only populated with ~22 CA1 electrodes + 1–2 callosum references per file. mPFC/OFC electrodes exist in the raw recordings (1–2 per file) but were never fed through `LFPElectrodeGroup`/`LFPV1`. Cross-region LFP analyses would require extracting those channels from the raw 30 kHz NWB data — not a one-liner.

### Important time/index conventions (inherited from parent)

- **Time is always in seconds**, never ms.
- All time-indexed DataFrames use `.loc[start:end]` for slicing, not `.iloc`.
- Brain area names are **`"HPC"`, `"mPFC"`, `"OFC"`** — exact capitalization.
- Not all sessions have all brain areas — check `data["spike_times"].keys()` before assuming.

## Data availability

Data lives wherever Spyglass + the NWB database say it does — the vendored `parameters.py` contains paths (`ROOT_DIR`, `PROCESSED_DATA_DIR`, `FIGURE_DIR`) that resolve relative to this repo, but **those directories don't actually exist here** and are only used when saving outputs. Don't rely on them for input data.

NWB file list and epoch counts are in `src/state_space_playground/data_loaders/constants.py::NWB_FILES` and `get_epoch_info()`.

## Code quality

Match the parent project's tooling:

```bash
uv run ruff check src/
uv run ruff format src/
uv run mypy src/
uv run pytest
```

Ruff config in `pyproject.toml` matches the parent (line length 88, py311 target, same rule selection). Vendored loader files are left mostly unlinted — don't churn them.

## Things not to do

- **Don't add `parameters.py` symbols to playground code.** If you need a constant, define it in your own module. `parameters.py` is vendored baggage, not a config surface.
- **Don't build "clever" abstractions around the vendored loaders.** Call `load_data()` and work with the returned dict directly. The parent project has learned this lesson.
- **Don't pin to a specific GPU in committed code.** `CUDA_VISIBLE_DEVICES` at runtime only.
- **Don't add paper figures, manuscript prose, or production outputs to this repo.** That belongs in `continuum-swr-replay`. This is a playground.
- **Don't commit notebooks as `.ipynb`** — `.gitignore` already excludes `notebooks/*.ipynb`. Use jupytext pairing (`.py` lives alongside the `.ipynb`).

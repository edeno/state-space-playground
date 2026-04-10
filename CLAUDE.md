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

```python
from state_space_playground.data_loaders import load_data, get_epoch_info

epoch_info = get_epoch_info()
data = load_data(
    nwb_file_name="chimi20200213_.nwb",
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

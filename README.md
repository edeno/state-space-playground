# state-space-playground

A sandbox for experimenting with state-space models (dynamax, `state_space_practice`, custom JAX) on neural and behavioral data from the spatial bandit task.

## What's in here

- `src/state_space_playground/data_loaders/` — vendored from [`continuum-swr-replay`](https://github.com/edeno/continuum-swr-replay). Loads session data (position, spikes, LFP, ripples, task events, trials) via Spyglass / NWB.
- `src/state_space_playground/parameters.py`, `types.py` — vendored alongside the loaders to satisfy their internal imports (`TrainingType`, typed dicts, etc.). Not intended as a public API of this project.
- `scripts/check_gpu.py` — quick sanity check that JAX can see and use a GPU.

The loader code is a **verbatim copy**, not an editable install, so it will drift from the parent project over time. If you fix a bug here that also exists upstream, port it.

## Setup

This project uses [uv](https://docs.astral.sh/uv/). JAX is installed with the bundled CUDA 12 wheels per the [JAX install docs](https://github.com/jax-ml/jax/blob/main/docs/installation.md) — no system CUDA toolkit required.

```bash
cd /cumulus/edeno/state-space-playground
uv sync                         # runtime deps
uv sync --extra dev             # include jupyterlab/jupytext/ruff/pytest/mypy
```

## Verify JAX GPU

```bash
# Use whatever GPU has free memory:
CUDA_VISIBLE_DEVICES=0 uv run python scripts/check_gpu.py
```

The script sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` so it shares nicely with other users on this box. Pick a GPU at runtime with `CUDA_VISIBLE_DEVICES`.

## Loading data

```python
from state_space_playground.data_loaders import load_data, get_epoch_info

epoch_info = get_epoch_info()
data = load_data(
    nwb_file_name="chimi20200212_.nwb",
    epoch_name="02_r1",
    ripple_detector_name="Kay",
)
position_info = data["position_info"]
spike_times = data["spike_times"]        # dict[brain_area] -> list[np.ndarray]
ripple_times = data["ripple_times"]
```

See the parent project's `CLAUDE.md` / `README.md` for a full tour of the returned dict.

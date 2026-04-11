"""GPU selection and JAX runtime configuration for the playground.

``pick_free_gpu()`` must be called **before jax/jaxlib is loaded**.
JAX initializes its CUDA backend on first import and reads the
``CUDA_VISIBLE_DEVICES`` / ``XLA_PYTHON_CLIENT_PREALLOCATE`` environment
variables at that point — setting them afterward is a no-op for that
process.

**Note on transitive imports:** ``state_space_playground.session`` (and
anything that imports ``state_space_playground.data_loaders``) pulls in
``jax`` transitively via vendored dependencies. In a notebook, call
``pick_free_gpu()`` as the very first thing, before any other
``state_space_playground`` import. The guard below will raise with a
clear error if you get the order wrong.

**Why f64 is the default here**
--------------------------------
By default ``pick_free_gpu()`` also enables ``jax_enable_x64``, so every
playground notebook runs JAX in double precision. This is a deliberate
choice, not a JAX default. Every notebook in this repo sets a specific
tone: we are running state-space models on long neural recordings
(T ~ 10^4–10^5 time bins) with Kalman-style covariance propagation, and
we observed that the upstream ``state_space_practice`` filter loses PSD
on its posterior covariance after enough bins in f32 — a classic
numerical-stability failure mode that manifests as silent NaN outputs
(see ``notebooks/02_STATUS.md`` for the full trace). Running in f64
empirically eliminates the issue; the runtime cost is approximately
40% slower per step on A100.

If you have a specific reason to use f32 — e.g., you're benchmarking a
Joseph-form fix upstream or you're running a throughput-sensitive model
that doesn't have this failure mode — pass ``enable_x64=False``. The
function then explicitly sets ``jax_enable_x64=False`` rather than
leaving it unset, so the precision is deterministic regardless of
whether other code in the process later imports JAX.

Usage
-----
::

    from state_space_playground.gpu import pick_free_gpu
    pick_free_gpu()                              # MUST come first
    from state_space_playground.session import load_session  # pulls in jax
    import jax, jax.numpy as jnp
    # jax.config.jax_enable_x64 is True; jnp.zeros(1).dtype == float64
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)


def pick_free_gpu(
    min_free_mb: int = 20_000,
    preallocate: bool = False,
    enable_x64: bool = True,
) -> int:
    """Select the GPU with the most free memory and pin the process to it.

    Runs ``nvidia-smi --query-gpu=index,memory.free``, sorts descending
    by free memory, and picks the first GPU meeting ``min_free_mb``.
    Sets ``CUDA_VISIBLE_DEVICES``, disables XLA preallocation by default,
    and (by default) enables ``jax_enable_x64`` so the rest of the
    process runs in double precision.

    Parameters
    ----------
    min_free_mb : int, default 20_000
        Minimum free memory (MB) required on the selected GPU. 20 GB is
        enough headroom for the models in this playground.
    preallocate : bool, default False
        If True, allow XLA to preallocate ~75% of the GPU at first use.
        The default (False) sets ``XLA_PYTHON_CLIENT_PREALLOCATE=false``
        so the process only uses memory it actually needs.
    enable_x64 : bool, default True
        Sets ``jax_enable_x64`` to the given value. The default (True)
        makes all subsequent JAX computations default to float64 / int64.
        This is the playground default because upstream
        ``state_space_practice``'s Kalman-style filters lose PSD on the
        posterior covariance in f32 over long sequences (see module
        docstring). When False, this function still explicitly sets
        ``jax_enable_x64=False`` so the precision is deterministic even
        if other code in the process later imports JAX.

    Returns
    -------
    int
        Physical GPU index that was pinned. JAX will see it as ``cuda:0``.

    Raises
    ------
    RuntimeError
        If jax is already imported, if ``nvidia-smi`` fails, or if no
        GPU has enough free memory.
    """
    # `jax` alone is not enough — `jaxlib` is the C extension that
    # actually initializes the CUDA backend, and some libraries pull
    # it in transitively without importing `jax` first. Check both.
    already_loaded = {
        name for name in ("jax", "jaxlib", "jaxlib.xla_extension")
        if name in sys.modules
    }
    if already_loaded:
        raise RuntimeError(
            "pick_free_gpu() must be called BEFORE jax/jaxlib is imported. "
            f"Already loaded: {sorted(already_loaded)}. "
            "Their CUDA backend is already initialized and "
            "CUDA_VISIBLE_DEVICES will have no effect."
        )

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ) as e:
        raise RuntimeError(f"nvidia-smi query failed: {e}") from e

    # Parse lines of "index, free_mb"
    gpus: list[tuple[int, int]] = []
    for line in result.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2:
            continue
        gpus.append((int(parts[0]), int(parts[1])))

    if not gpus:
        raise RuntimeError("nvidia-smi returned no GPU rows.")

    # Sort descending by free memory; stable so earlier-index GPUs break ties.
    gpus.sort(key=lambda t: -t[1])

    free_gpus = [(i, m) for i, m in gpus if m >= min_free_mb]
    if not free_gpus:
        best_idx, best_free = gpus[0]
        raise RuntimeError(
            f"No GPU with at least {min_free_mb} MB free. "
            f"Best candidate: GPU {best_idx} with {best_free} MB. "
            f"Lower min_free_mb or wait for the box to free up."
        )

    chosen_idx, chosen_free = free_gpus[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen_idx)
    if preallocate:
        os.environ.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
    else:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # Set jax_enable_x64 explicitly (regardless of enable_x64's value)
    # so the precision state is deterministic. Importing jax here is
    # safe: (a) we've already verified it was not yet imported, (b) the
    # env vars set above make `import jax` initialize the CUDA backend
    # correctly, and (c) jax_enable_x64 must be set before any jax array
    # is created, which this function does not do before returning.
    import jax

    jax.config.update("jax_enable_x64", enable_x64)

    logger.info(
        "Pinned to GPU %d (%d MB free). JAX will see it as cuda:0. "
        "jax_enable_x64=%s",
        chosen_idx,
        chosen_free,
        enable_x64,
    )
    print(
        f"[pick_free_gpu] Using physical GPU {chosen_idx} "
        f"({chosen_free} MB free, jax_enable_x64={enable_x64})",
        flush=True,
    )
    return chosen_idx

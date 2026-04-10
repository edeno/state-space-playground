"""GPU selection utilities for the shared CUDA box.

This module must be imported (and ``pick_free_gpu()`` called) **before**
``import jax``. JAX initializes its CUDA backend on first import and
reads the ``CUDA_VISIBLE_DEVICES`` / ``XLA_PYTHON_CLIENT_PREALLOCATE``
environment variables at that point — setting them afterward is a
no-op for that process.

Usage
-----
::

    from state_space_playground.gpu import pick_free_gpu
    pick_free_gpu()
    import jax  # sees the chosen physical GPU as ``cuda:0``
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
) -> int:
    """Select the GPU with the most free memory and pin the process to it.

    Runs ``nvidia-smi --query-gpu=index,memory.free``, sorts descending
    by free memory, and picks the first GPU meeting ``min_free_mb``.
    Sets ``CUDA_VISIBLE_DEVICES`` and (by default) disables XLA
    preallocation so this process shares the GPU politely.

    Parameters
    ----------
    min_free_mb : int, default 20_000
        Minimum free memory (MB) required on the selected GPU. 20 GB is
        enough headroom for the models in this playground.
    preallocate : bool, default False
        If True, allow XLA to preallocate ~75% of the GPU at first use.
        The default (False) sets ``XLA_PYTHON_CLIENT_PREALLOCATE=false``
        so the process only uses memory it actually needs.

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

    logger.info(
        "Pinned to GPU %d (%d MB free). JAX will see it as cuda:0.",
        chosen_idx,
        chosen_free,
    )
    print(
        f"[pick_free_gpu] Using physical GPU {chosen_idx} "
        f"({chosen_free} MB free)",
        flush=True,
    )
    return chosen_idx

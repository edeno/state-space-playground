"""Cached session loader for the state-space playground.

Loading a bandit session via the vendored ``data_loaders.load_data``
takes ~2 minutes (DataJoint queries, NWB reads, spike filtering, task
variable computation). Every notebook restart re-does that work, which
is painful when iterating.

This module wraps ``load_data`` with a pickle cache keyed on
``(nwb_file_name, epoch_name, use_sorted_hpc)``. First call populates
the cache; subsequent calls return in <5s.

Cache files live under ``<repo_root>/cache/sessions/`` which is
gitignored.

Usage
-----
::

    from state_space_playground.session import load_session
    data = load_session(
        "j1620210710_.nwb", "02_r1", use_sorted_hpc=True,
    )
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

from .data_loaders import load_data

logger = logging.getLogger(__name__)

# Repo root is three levels up from this file
# (src/state_space_playground/session.py -> repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_ROOT = _REPO_ROOT / "cache" / "sessions"


def _cache_path(
    nwb_file_name: str,
    epoch_name: str,
    use_sorted_hpc: bool,
) -> Path:
    """Return the pickle path for a given (nwb, epoch, sorted-flag) tuple."""
    stem = nwb_file_name.removesuffix(".nwb").removesuffix("_")
    suffix = "_sortedHPC" if use_sorted_hpc else "_clusterless"
    return CACHE_ROOT / f"{stem}_{epoch_name}{suffix}.pkl"


def load_session(
    nwb_file_name: str,
    epoch_name: str,
    use_sorted_hpc: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Load a session via the cache, falling back to ``load_data``.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name (e.g., ``"j1620210710_.nwb"``).
    epoch_name : str
        Epoch name (e.g., ``"02_r1"``).
    use_sorted_hpc : bool, default False
        Forwarded to ``load_data``. Cached separately per flag value
        since clusterless and sorted HPC data are structurally different.
    force : bool, default False
        If True, skip the cache, re-fetch, and overwrite the cache file.

    Returns
    -------
    dict
        The full session dict (same shape as ``load_data``).
    """
    path = _cache_path(nwb_file_name, epoch_name, use_sorted_hpc)

    if path.exists() and not force:
        logger.info("Loading cached session from %s", path)
        with path.open("rb") as f:
            return pickle.load(f)  # noqa: S301 (trusted local cache)

    logger.info(
        "Cache miss — calling load_data(%s, %s, use_sorted_hpc=%s)",
        nwb_file_name,
        epoch_name,
        use_sorted_hpc,
    )
    data = load_data(
        nwb_file_name=nwb_file_name,
        epoch_name=epoch_name,
        use_sorted_hpc=use_sorted_hpc,
    )

    # Atomic write: dump to a sibling .tmp file, then rename. A killed
    # process leaves the .tmp (or nothing), never a half-written cache
    # file that would blow up on the next warm load.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    size_mb = path.stat().st_size / 1e6
    logger.info("Cached session to %s (%.1f MB)", path, size_mb)

    return data

"""Utility functions for data loading.

This module provides helper functions for data loading, session management, and
time-based data filtering. These utilities support the main data loading pipeline
by handling NWB file parsing, epoch discovery, event labeling, and training data
selection for decoding models.

Main Functions
--------------
get_epoch_info
    Queries Spyglass database for all available epochs across animals. Results are
    cached for efficiency. This is the primary entry point for discovering available
    data sessions.
get_training_timepoints
    Creates boolean masks to select time points for decoder training, with options
    to exclude ripple times or use all data.
get_labels
    Efficiently labels time points with event IDs using vectorized binary search.
parse_nwb_filename
    Extracts animal name and date from NWB filenames.

Features
--------
- Cached database queries: get_epoch_info uses @cache to avoid redundant queries
- Efficient time labeling: get_labels uses np.searchsorted for O(n log n) complexity
- Type-safe training selection: Supports both TrainingType enum and string inputs
- Robust parsing: Validates NWB filename format with regex

Notes
-----
- All database queries use Spyglass DataJoint tables (Session, IntervalList, etc.)
- Warning filters suppress verbose pynwb/hdmf/datajoint logging
- Cached results persist for the entire Python session
- Time indices are always in seconds

See Also
--------
continuum_swr_replay.data_loaders.bandit_task : Main data loading orchestration
continuum_swr_replay.data_loaders.constants : NWB_FILES list and task constants
continuum_swr_replay.parameters : TrainingType enum definition
continuum_swr_replay.types : SessionData TypedDict

Examples
--------
>>> # Get all available epochs
>>> epoch_info = get_epoch_info()
>>> epoch_info[["animal", "date", "epoch"]].head()

>>> # Parse filename
>>> animal, date = parse_nwb_filename("chimi20200212_.nwb")

>>> # Get training mask excluding ripples
>>> from continuum_swr_replay.parameters import TrainingType
>>> mask = get_training_timepoints(data, TrainingType.NO_RIPPLE)

"""

from __future__ import annotations

import re
import warnings
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from spyglass.common import IntervalList, PositionIntervalMap, Session

if TYPE_CHECKING:
    from ..types import SessionData

from ..parameters import TrainingType
from .constants import NWB_FILES

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pynwb")
warnings.filterwarnings("ignore", category=UserWarning, module="hdmf")
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint")


def parse_nwb_filename(filename: str) -> tuple[str, int]:
    """Parse NWB filename to extract animal name and date.

    Parameters
    ----------
    filename : str
        NWB filename in format: {animal}{YYYYMMDD}_.nwb
        Example: "chimi20200212_.nwb"

    Returns
    -------
    tuple[str, int]
        Animal name and date as (animal, date)
        Example: ("chimi", 20200212)

    Raises
    ------
    ValueError
        If filename doesn't match expected format

    Examples
    --------
    >>> parse_nwb_filename("chimi20200212_.nwb")
    ('chimi', 20200212)

    >>> parse_nwb_filename("peanut20201125_.nwb")
    ('peanut', 20201125)

    >>> parse_nwb_filename("j1620210706_.nwb")
    ('j16', 20210706)

    """
    # Pattern: {animal_name}{8_digits}_.nwb
    # Animal name is one or more alphanumeric chars (non-greedy), date is exactly 8 digits
    # The date is always YYYYMMDD format, so we match the last 8 digits before _.nwb
    pattern = r"^([a-zA-Z0-9]+?)(\d{8})_\.nwb$"
    match = re.match(pattern, filename)

    if not match:
        raise ValueError(
            f"Invalid NWB filename format: {filename}. "
            f"Expected format: {{animal}}{{YYYYMMDD}}_.nwb"
        )

    animal = match.group(1)
    date = int(match.group(2))

    return animal, date


@cache
def get_epoch_info() -> pd.DataFrame:
    """Get information about all available epochs across all animals.

    This function queries the Spyglass database, which is expensive. Results
    are cached for the lifetime of the Python process to avoid redundant
    database queries.

    Returns
    -------
    pd.DataFrame
        Epoch information with columns:
        - animal, date, epoch, epoch_tag
        - exposure (session number for that animal)
        - start_time, end_time

    Notes
    -----
    The cache persists for the entire Python session. If the database is
    modified externally during runtime, the cached result will be stale.
    This is acceptable for typical analysis workflows where the database
    is static.

    """
    epoch_info = (
        pd.DataFrame(
            (
                Session
                & [
                    {
                        "nwb_file_name": nwb_file_name,
                        "session_description": "Spatial bandit task (regular)",
                    }
                    for nwb_file_name in NWB_FILES
                ]
            )
            * PositionIntervalMap
        )
        .set_index(["nwb_file_name", "interval_list_name"])
        .sort_index()
    )

    # only use run epochs
    epoch_info = epoch_info[
        epoch_info.index.get_level_values("interval_list_name").str.match(
            r"^[0-9]+_r[0-9]+$"
        )
    ]

    animal, date = zip(
        *[
            parse_nwb_filename(file_name)
            for file_name in epoch_info.index.get_level_values(
                "nwb_file_name"
            ).to_numpy()
        ],
        strict=True,
    )
    epoch_info["animal"] = animal
    epoch_info["date"] = date

    epoch, epoch_tag = zip(
        *epoch_info.index.get_level_values("interval_list_name").str.split("_"),
        strict=True,
    )
    epoch_info["epoch"] = np.asarray(epoch, dtype=int)
    epoch_info["epoch_tag"] = epoch_tag

    def _count_exposure(df: pd.DataFrame) -> pd.DataFrame:
        df["exposure"] = np.arange(len(df)) + 1
        return df

    epoch_info = (
        epoch_info.sort_values(["animal", "date", "epoch"])
        .groupby("animal", group_keys=False)
        .apply(_count_exposure)
    )

    interval_times = np.stack(
        (
            IntervalList
            & epoch_info.reset_index()
            .loc[:, ("nwb_file_name", "interval_list_name")]
            .to_dict(orient="records")
        ).fetch("valid_times")
    ).squeeze()

    epoch_info["start_time"] = pd.to_datetime(interval_times[:, 0], unit="s")
    epoch_info["end_time"] = pd.to_datetime(interval_times[:, 1], unit="s")

    return epoch_info


def get_labels(times: pd.DataFrame, time: pd.Index) -> pd.DataFrame:
    """Create event labels for each time point.

    Uses vectorized binary search (np.searchsorted) for efficient labeling
    of large datasets. Complexity is O(n_events * log(n_times)) instead of
    O(n_events * n_times) with explicit boolean masking.

    Parameters
    ----------
    times : pd.DataFrame
        Event times with start_time and end_time columns
    time : pd.Index
        Time index to label

    Returns
    -------
    pd.DataFrame
        Event labels (0 for no event, 1+ for event number)

    Notes
    -----
    Time points within [start_time, end_time] (inclusive) are labeled with
    sequential integers starting from 1. Time points outside all events are
    labeled 0.

    """
    event_labels = pd.DataFrame(0, index=time, columns=["event_number"], dtype=int)

    time_array = time.to_numpy()

    # Vectorized labeling using binary search (O(n_events * log(n_times)))
    for i, (start, end) in enumerate(
        zip(times["start_time"], times["end_time"], strict=True), start=1
    ):
        start_idx = np.searchsorted(time_array, start, side="left")
        end_idx = np.searchsorted(time_array, end, side="right")

        # Use iloc for integer-based indexing (loc would be label-based)
        event_labels.iloc[start_idx:end_idx, 0] = i

    return event_labels


def get_training_timepoints(
    data: SessionData, training_type: str | TrainingType = TrainingType.NO_RIPPLE
) -> np.ndarray:
    """Get boolean mask of time points to use for training.

    Parameters
    ----------
    data : dict
        Data dictionary with position_info and ripple_times
    training_type : str or TrainingType, optional
        Type of training data selection. Default is TrainingType.NO_RIPPLE.
        Accepts either TrainingType enum or equivalent string:
        - TrainingType.ALL or "all": use all time points
        - TrainingType.NO_RIPPLE or "no_ripple": exclude ripple times

    Returns
    -------
    np.ndarray
        Boolean mask of time points to use for training

    Examples
    --------
    >>> # Using enum (type-safe, recommended)
    >>> mask = get_training_timepoints(data, TrainingType.ALL)
    >>> # Using string (backward compatible)
    >>> mask = get_training_timepoints(data, "all")

    """
    # Convert string to enum if needed (backward compatibility)
    if isinstance(training_type, str):
        try:
            training_type = TrainingType(training_type)
        except ValueError:
            raise ValueError(
                f"Unknown training_type: {training_type}. "
                f"Must be TrainingType.ALL, TrainingType.NO_RIPPLE, 'all', or 'no_ripple'."
            ) from None

    if training_type == TrainingType.ALL:
        n_time = len(data["position_info"].index)
        return np.ones((n_time,), dtype=bool)

    elif training_type == TrainingType.NO_RIPPLE:
        ripple_labels = get_labels(
            data["ripple_times"], data["position_info"].index
        )
        is_ripple = ripple_labels > 0
        not_ripple = ~np.asarray(is_ripple).squeeze()

        return not_ripple
    else:
        raise ValueError(
            f"Unknown training_type: {training_type}. "
            f"Must be TrainingType.ALL or TrainingType.NO_RIPPLE."
        )

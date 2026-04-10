"""Main data loading orchestrator for the spatial bandit task.

This module provides the primary `load_data()` function that orchestrates
loading all data types for a given session and epoch: position tracking,
neural spikes, LFP signals, task events, and computed task variables.

Functions
---------
load_data
    Load all data for a session and epoch (main entry point).

Notes
-----
This is the recommended way to load data for analysis. It coordinates
all submodules (position, spikes, lfp, events, trials, task_variables)
and ensures temporal alignment across all data types.

All data is restricted to the same time bounds (position data time range),
and all time-indexed data uses seconds as the time unit.

See Also
--------
get_epoch_info : Get list of all available sessions and epochs
get_training_timepoints : Select time points for decoder training

"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pandas as pd
from spyglass.common import PositionIntervalMap

# Suppress Spyglass warnings
logging.getLogger("spyglass.utils").setLevel(logging.ERROR)
logging.getLogger("spyglass").setLevel(logging.ERROR)

if TYPE_CHECKING:
    from ..types import (
        DIOEventData,
        PositionInfoDict,
        RippleData,
        SessionData,
        SpikeDataDict,
        ThetaData,
    )

from .events import load_dios
from .lfp import filter_ripple_lfp_time, get_multiunit, get_ripple, get_theta
from .position import get_position_info
from .spikes import filter_spike_times, get_electrode_group_info, get_spike_data
from .task_variables import make_task_variables
from .trials import make_trials_df_from_well_visits, make_well_visits_df

logger = logging.getLogger(__name__)


def _process_spike_data(
    spike_data: SpikeDataDict,
    position_index: pd.Index,
    should_load: Callable[[str], bool],
    result: dict[str, Any],
) -> None:
    """Process raw spike data and add to result dict.

    Filters spike times to position bounds and adds requested keys to result.
    """
    from ..types import SessionDataKeys

    spike_data["spike_times"], spike_data["spike_waveform_features"] = filter_spike_times(
        spike_data["spike_times"],
        spike_data["spike_waveform_features"],
        position_index,
    )
    if should_load(SessionDataKeys.SPIKE_TIMES):
        result["spike_times"] = spike_data["spike_times"]
    if should_load(SessionDataKeys.SPIKE_WAVEFORM_FEATURES):
        result["spike_waveform_features"] = spike_data["spike_waveform_features"]


def _process_ripple_data(
    ripple_data: RippleData,
    position_index: pd.Index,
    should_load: Callable[[str], bool],
    result: dict[str, Any],
) -> None:
    """Process raw ripple data and add to result dict."""
    filtered_ripple = filter_ripple_lfp_time(ripple_data, position_index)
    for key, value in filtered_ripple.items():
        if should_load(key):
            result[key] = value


def _process_theta_data(
    theta_data: ThetaData,
    should_load: Callable[[str], bool],
    result: dict[str, Any],
) -> None:
    """Process theta data and add requested keys to result dict."""
    for key, value in theta_data.items():
        if should_load(key):
            result[key] = value


def _process_multiunit_data(
    spike_times: dict[str, Any],
    position_info: pd.DataFrame,
    should_load: Callable[[str], bool],
    result: dict[str, Any],
) -> None:
    """Process multiunit data and add requested keys to result dict."""
    from ..types import SessionDataKeys

    if not any(
        should_load(key)
        for key in [
            SessionDataKeys.HSE_TIMES,
            SessionDataKeys.MULTIUNIT_FIRING_RATE,
            SessionDataKeys.MULTIUNIT_RATE_ZSCORE,
        ]
    ):
        return

    multiunit_data = get_multiunit(spike_times, position_info)
    for key, value in multiunit_data.items():
        if should_load(key):
            result[key] = value


def _process_behavioral_data(
    dio_events: DIOEventData,
    position_data: PositionInfoDict,
    should_load: Callable[[str], bool],
    result: dict[str, Any],
) -> None:
    """Process DIO events and compute behavioral data (well visits, trials, task variables)."""
    from ..types import SessionDataKeys

    # Add DIO events to result if requested
    for key, value in dio_events.items():
        if should_load(key):
            result[key] = value

    # Sequential: well visits, trials, task variables (depend on each other)
    if not (
        should_load(SessionDataKeys.WELL_VISITS)
        or should_load(SessionDataKeys.TRIALS)
        or should_load(SessionDataKeys.TASK_VARIABLES)
    ):
        return

    well_visits = make_well_visits_df(
        dio_events["beam_breaks"],
        dio_events["pump_events"],
        position_data["position_info"],
        track_graph=position_data["track_graph"],
    )

    if should_load(SessionDataKeys.WELL_VISITS):
        result["well_visits"] = well_visits

    if should_load(SessionDataKeys.TRIALS) or should_load(SessionDataKeys.TASK_VARIABLES):
        trials_df = make_trials_df_from_well_visits(well_visits)

        if should_load(SessionDataKeys.TRIALS):
            result["trials"] = trials_df

        if should_load(SessionDataKeys.TASK_VARIABLES):
            task_variables_df = make_task_variables(
                position_info=position_data["position_info"],
                trials_df=trials_df,
                track_graph=position_data["track_graph"],
            )
            result["task_variables"] = task_variables_df


def load_data(
    nwb_file_name: str,
    epoch_name: str,
    ripple_detector_name: str = "Kay",
    include: frozenset[str] | None = None,
) -> SessionData:
    """Load data for a given session and epoch with selective loading.

    This is the main entry point for loading data. It orchestrates loading
    of position, spikes, LFP, task events, and computes derived task variables.

    Selective loading allows you to load only the data needed for specific
    analyses, improving performance and reducing memory usage.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name (e.g., "chimi20200212_.nwb")
    epoch_name : str
        Epoch name (e.g., "02_r1")
    ripple_detector_name : str, optional
        Ripple detector name (currently only "Kay" is supported)
    include : frozenset[str] | None, optional
        Set of SessionDataKeys to load. If None (default), loads all data.
        Use DataLoadingPreset for common configurations or create custom sets.
        Position data (position_info, track_graph) is always loaded as it
        defines time bounds for other data types

    Returns
    -------
    SessionData
        TypedDict containing all loaded and computed data. Includes:
        - Position data: position_info, track_graph, linearization info
        - Neural data: spike_times, spike_waveform_features
        - LFP data: ripple and theta band signals and metrics
        - Multiunit activity: hse_times, multiunit_firing_rate, multiunit_rate_zscore
        - Task events: beam_breaks, pump_events, light_events
        - Behavioral data: well_visits, trials, task_variables
        - Metadata: electrode_group_info

        See `continuum_swr_replay.analysis.hippocampus.types.SessionData` for
        complete field documentation.

    See Also
    --------
    get_epoch_info : Get list of all available sessions
    get_training_timepoints : Select time points for decoder training
    get_position_info : Position loading (called internally)
    get_spike_data : Spike loading (called internally)

    Examples
    --------
    Load all data (backward compatible):

    >>> data = load_data("chimi20200212_.nwb", "02_r1")
    >>> data.keys()
    dict_keys(['position_info', 'track_graph', 'spike_times', ...])

    Load minimal data for basic decoding:

    >>> from continuum_swr_replay.types import DataLoadingPreset
    >>> data = load_data("chimi20200212_.nwb", "02_r1",
    ...                  include=DataLoadingPreset.MINIMAL)
    >>> data.keys()
    dict_keys(['position_info', 'track_graph', 'linear_edge_order',
               'linear_edge_spacing', 'spike_times', 'spike_waveform_features'])

    Load custom subset of data:

    >>> from continuum_swr_replay.types import SessionDataKeys
    >>> data = load_data(
    ...     "chimi20200212_.nwb", "02_r1",
    ...     include={SessionDataKeys.POSITION_INFO, SessionDataKeys.SPIKE_TIMES, SessionDataKeys.RIPPLE_TIMES, SessionDataKeys.TRIALS}
    ... )

    Extend a preset with additional keys:

    >>> custom = DataLoadingPreset.NEURAL | {SessionDataKeys.TRIALS, SessionDataKeys.TASK_VARIABLES}
    >>> data = load_data("chimi20200212_.nwb", "02_r1", include=custom)

    Access position data:

    >>> pos = data['position_info']
    >>> pos[['head_position_x', 'head_position_y', 'linear_position']].head()
       head_position_x  head_position_y  linear_position
    0     125.3              85.7            42.5
    1     126.1              86.2            43.8
    ...

    Check which brain areas have spike data:

    >>> list(data['spike_times'].keys())
    ['HPC', 'mPFC']  # OFC not recorded in this session

    Access hippocampal spike data:

    >>> hpc_spikes = data['spike_times']['HPC']
    >>> print(f"Number of tetrodes: {len(hpc_spikes)}")
    Number of tetrodes: 8
    >>> print(f"Total spikes: {sum(len(st) for st in hpc_spikes)}")
    Total spikes: 45231

    Get trial information:

    >>> trials = data['trials']
    >>> trials[['start_time', 'end_time', 'is_reward', 'from_well', 'to_well']].head(3)
       start_time  end_time  is_reward  from_well  to_well
    0   1234.5     1240.2    True       0          3
    1   1241.0     1247.8    False      3          1
    2   1249.3     1256.1    True       1          4

    Access task variables:

    >>> task_vars = data['task_variables']
    >>> task_vars[['trial_number', 'path_progress', 'dist_to_goal']].head()
       trial_number  path_progress  dist_to_goal
    0     1           0.00           85.3
    1     1           0.12           72.1
    2     1           0.25           58.7
    ...

    Get ripple times:

    >>> ripples = data['ripple_times']
    >>> print(f"Number of ripples: {len(ripples)}")
    Number of ripples: 342
    >>> ripples[['start_time', 'end_time', 'duration']].head(3)
       start_time  end_time  duration
    0   1235.123   1235.248   0.125
    1   1237.891   1238.012   0.121
    2   1242.456   1242.598   0.142

    """
    from ..types import SessionDataKeys

    # Helper to check if we should load this key
    def should_load(key: str) -> bool:
        return include is None or key in include

    # Always load position data (required for time bounds)
    position_interval_name = (
        PositionIntervalMap
        & {"nwb_file_name": nwb_file_name, "interval_list_name": epoch_name}
    ).fetch1("position_interval_name")
    position_data = get_position_info(nwb_file_name, epoch_name, position_interval_name)

    # Drop time points with NaN positions (typically at start/end when tracking unavailable)
    # This ensures all time-indexed data will be aligned to valid position times
    n_before = len(position_data["position_info"])
    position_data["position_info"] = position_data["position_info"].dropna(
        subset=["head_position_x", "head_position_y"]
    )
    n_dropped = n_before - len(position_data["position_info"])
    if n_dropped > 0:
        logger.info(
            f"Dropped {n_dropped} time points with NaN position "
            f"({100 * n_dropped / n_before:.1f}%)"
        )

    # Initialize result with position data (always included)
    result: dict = {
        "position_info": position_data["position_info"],
        "track_graph": position_data["track_graph"],
        "linear_edge_order": position_data["linear_edge_order"],
        "linear_edge_spacing": position_data["linear_edge_spacing"],
    }

    # Load and process spike data
    if should_load(SessionDataKeys.SPIKE_TIMES) or should_load(
        SessionDataKeys.SPIKE_WAVEFORM_FEATURES
    ):
        spike_data = get_spike_data(nwb_file_name)
        _process_spike_data(
            spike_data, position_data["position_info"].index, should_load, result
        )

    # Load and process ripple data
    if any(
        should_load(key)
        for key in [
            SessionDataKeys.RIPPLE_TIMES,
            SessionDataKeys.RIPPLE_FILTERED_LFPS,
            SessionDataKeys.RIPPLE_LFPS,
            SessionDataKeys.RIPPLE_CONSENSUS_TRACE,
            SessionDataKeys.ZSCORED_RIPPLE_CONSENSUS_TRACE,
        ]
    ):
        ripple_data = get_ripple(nwb_file_name, epoch_name, ripple_detector_name)
        _process_ripple_data(
            ripple_data, position_data["position_info"].index, should_load, result
        )

    # Load and process theta data
    if any(
        should_load(key)
        for key in [
            SessionDataKeys.THETA_FILTERED_LFP,
            SessionDataKeys.THETA_PHASE,
            SessionDataKeys.THETA_POWER,
        ]
    ):
        theta_data = get_theta(nwb_file_name, epoch_name)
        _process_theta_data(theta_data, should_load, result)

    # Load and process DIO events and behavioral data
    if any(
        should_load(key)
        for key in [
            SessionDataKeys.BEAM_BREAKS,
            SessionDataKeys.PUMP_EVENTS,
            SessionDataKeys.LIGHT_EVENTS,
            SessionDataKeys.WELL_VISITS,
            SessionDataKeys.TRIALS,
            SessionDataKeys.TASK_VARIABLES,
        ]
    ):
        dio_events = load_dios(
            nwb_file_name, position_data["position_info"].index.to_numpy()[[0, -1]]
        )
        _process_behavioral_data(dio_events, position_data, should_load, result)

    # Process multiunit data (depends on spike_times being loaded)
    if "spike_times" in result:
        _process_multiunit_data(
            result["spike_times"], position_data["position_info"], should_load, result
        )

    # Load electrode group info
    if should_load(SessionDataKeys.ELECTRODE_GROUP_INFO):
        result["electrode_group_info"] = get_electrode_group_info(nwb_file_name)

    # Return result dict - type checkers will validate it matches SessionData
    # The dict is built dynamically based on the include parameter
    return result  # type: ignore[return-value]

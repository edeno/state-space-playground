"""Central type definitions for the continuum-swr-replay project.

This module provides shared TypedDicts and type aliases used across
data loading, analysis, and visualization modules.

All TypedDicts are consolidated here for:
- Single source of truth for type definitions
- Easier discovery and maintenance
- Consistent type annotations across modules
- Better IDE autocomplete support
"""

from enum import StrEnum
from typing import TypedDict

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray


class SessionData(TypedDict, total=False):
    """Data structure returned by load_data() from data_loaders.

    This TypedDict defines the expected structure of session data loaded
    from NWB files through the Spyglass pipeline.

    Note: Uses total=False because not all sessions have recordings from all
    brain areas (e.g., some sessions lack mPFC or OFC data).

    Attributes
    ----------
    position_info : pd.DataFrame
        Time-indexed DataFrame with position tracking data including:
        - head_position_x, head_position_y: Raw 2D position
        - projected_x_position, projected_y_position: Position on track graph
        - linear_position: 1D linearized position along track
        - track_segment_id: Current edge of track graph
        - head_speed: Instantaneous speed
        - head_orientation: Directional heading
    track_graph : nx.Graph
        NetworkX graph representing track topology.
        Nodes are wells (0-5), edges are track segments.
    linear_edge_order : list
        List of edges for linearization
    linear_edge_spacing : list
        List of edge spacings for linearization
    spike_times : dict[str, list[np.ndarray]]
        Spike times organized by brain area (e.g., "HPC", "mPFC", "OFC").
        Each area contains a list of arrays, one per sort group/unit.
    spike_waveform_features : dict[str, list[np.ndarray]]
        Waveform features for clusterless decoding (primarily HPC).
        Shape: (n_spikes, n_features) per sort group.
    ripple_times : pd.DataFrame
        Detected sharp-wave ripple events with start_time, end_time columns.
    ripple_filtered_lfps : pd.DataFrame
        Ripple-band filtered LFP signals (time-indexed).
    ripple_lfps : pd.DataFrame
        Raw ripple LFP signals (time-indexed).
    ripple_consensus_trace : pd.Series
        Ripple consensus trace across electrodes (time-indexed).
    zscored_ripple_consensus_trace : pd.Series
        Z-scored ripple detection signal (time-indexed).
    theta_filtered_lfp : pd.DataFrame
        Theta-band filtered LFP signals (time-indexed).
    theta_phase : pd.Series
        Instantaneous theta phase in radians (time-indexed).
    theta_power : pd.Series
        Theta band power (time-indexed).
    hse_times : dict[str, pd.DataFrame]
        High-synchrony events (multi-unit activity bursts) per brain area.
    multiunit_firing_rate : dict[str, pd.DataFrame]
        Multi-unit firing rate per brain area (time-indexed).
    multiunit_rate_zscore : dict[str, pd.Series]
        Z-scored multi-unit firing rates per brain area (time-indexed).
    beam_breaks : pd.DataFrame
        Poke events (nose poke in/out at wells) with start_time, end_time.
    pump_events : pd.DataFrame
        Reward pump events with timestamps.
    light_events : pd.DataFrame
        Light cue events with timestamps.
    electrode_group_info : pd.DataFrame
        Metadata about electrode groups and their locations.
    well_visits : pd.DataFrame
        Well visit intervals with poke in/out times and well IDs.
    trials : pd.DataFrame
        Trial-level data with start_time, end_time, start_node, end_node, etc.
    task_variables : pd.DataFrame
        Time-indexed task variables including:
        - path_progress: 0-1 normalized progress through trial
        - dist_to_goal: Distance remaining to goal well
        - time_to_goal: Time since trial start
        - turn_direction: Sequence of turns in trial

    """

    # Position and tracking
    position_info: pd.DataFrame
    track_graph: nx.Graph
    linear_edge_order: list
    linear_edge_spacing: list

    # Neural data - spike times and waveforms
    spike_times: dict[str, list[np.ndarray]]
    spike_waveform_features: dict[str, list[np.ndarray]]

    # LFP data - ripple band
    ripple_times: pd.DataFrame
    ripple_filtered_lfps: pd.DataFrame
    ripple_lfps: pd.DataFrame
    ripple_consensus_trace: pd.Series
    zscored_ripple_consensus_trace: pd.Series

    # LFP data - theta band
    theta_filtered_lfp: pd.DataFrame
    theta_phase: pd.Series
    theta_power: pd.Series

    # Multi-unit activity
    hse_times: dict[str, pd.DataFrame]
    multiunit_firing_rate: dict[str, pd.DataFrame]
    multiunit_rate_zscore: dict[str, pd.DataFrame]

    # Task events from DIO
    beam_breaks: pd.DataFrame
    pump_events: pd.DataFrame
    light_events: pd.DataFrame

    # Behavioral data
    well_visits: pd.DataFrame
    trials: pd.DataFrame
    task_variables: pd.DataFrame

    # Metadata
    electrode_group_info: pd.DataFrame


class SessionDataKeys(StrEnum):
    """String constants for SessionData dictionary keys.

    Use these enum members instead of string literals to prevent typos
    and enable IDE autocomplete when accessing SessionData dictionaries.

    Notes
    -----
    This enum inherits from StrEnum (Python 3.11+), which provides:
    - Direct string comparison: SessionDataKeys.POSITION_INFO == "position_info"
    - Iteration: list(SessionDataKeys) to get all keys
    - Membership testing: "position_info" in SessionDataKeys
    - Automatic string coercion in contexts expecting strings

    Examples
    --------
    >>> from continuum_swr_replay.types import SessionDataKeys
    >>> data = load_data(...)
    >>> position_info = data[SessionDataKeys.POSITION_INFO]  # Instead of data["position_info"]
    >>> spike_times = data[SessionDataKeys.SPIKE_TIMES]      # Instead of data["spike_times"]
    >>>
    >>> # Iterate over all available keys
    >>> for key in SessionDataKeys:
    ...     if key in data:
    ...         print(f"Found {key}")
    >>>
    >>> # Check membership
    >>> if "position_info" in SessionDataKeys:
    ...     print("Valid key")

    """

    # Position and tracking
    POSITION_INFO = "position_info"
    TRACK_GRAPH = "track_graph"
    LINEAR_EDGE_ORDER = "linear_edge_order"
    LINEAR_EDGE_SPACING = "linear_edge_spacing"

    # Neural data - spike times and waveforms
    SPIKE_TIMES = "spike_times"
    SPIKE_WAVEFORM_FEATURES = "spike_waveform_features"

    # LFP data - ripple band
    RIPPLE_TIMES = "ripple_times"
    RIPPLE_FILTERED_LFPS = "ripple_filtered_lfps"
    RIPPLE_LFPS = "ripple_lfps"
    RIPPLE_CONSENSUS_TRACE = "ripple_consensus_trace"
    ZSCORED_RIPPLE_CONSENSUS_TRACE = "zscored_ripple_consensus_trace"

    # LFP data - theta band
    THETA_FILTERED_LFP = "theta_filtered_lfp"
    THETA_PHASE = "theta_phase"
    THETA_POWER = "theta_power"

    # Multi-unit activity
    HSE_TIMES = "hse_times"
    MULTIUNIT_FIRING_RATE = "multiunit_firing_rate"
    MULTIUNIT_RATE_ZSCORE = "multiunit_rate_zscore"

    # Task events from DIO
    BEAM_BREAKS = "beam_breaks"
    PUMP_EVENTS = "pump_events"
    LIGHT_EVENTS = "light_events"

    # Behavioral data
    WELL_VISITS = "well_visits"
    TRIALS = "trials"
    TASK_VARIABLES = "task_variables"

    # Metadata
    ELECTRODE_GROUP_INFO = "electrode_group_info"


class DataLoadingPreset:
    """Predefined data loading presets for common analysis workflows.

    Use these presets with load_data(include=...) to load only the data
    needed for specific analyses, improving performance and memory usage.

    Each preset is a frozenset of SessionDataKeys constants.

    Examples
    --------
    >>> from continuum_swr_replay.data_loaders.bandit_task import load_data
    >>> from continuum_swr_replay.types import DataLoadingPreset
    >>>
    >>> # Load minimal data for basic decoding
    >>> data = load_data("chimi20200212_.nwb", "02_r1",
    ...                  include=DataLoadingPreset.MINIMAL)
    >>>
    >>> # Extend a preset with custom keys
    >>> from continuum_swr_replay.types import SessionDataKeys
    >>> custom = DataLoadingPreset.NEURAL | {SessionDataKeys.TRIALS, SessionDataKeys.TASK_VARIABLES}
    >>> data = load_data("chimi20200212_.nwb", "02_r1", include=custom)

    """

    # Minimal - just position and spikes for basic decoding
    MINIMAL: frozenset[str] = frozenset(
        {
            SessionDataKeys.POSITION_INFO,
            SessionDataKeys.TRACK_GRAPH,
            SessionDataKeys.LINEAR_EDGE_ORDER,
            SessionDataKeys.LINEAR_EDGE_SPACING,
            SessionDataKeys.SPIKE_TIMES,
            SessionDataKeys.SPIKE_WAVEFORM_FEATURES,
        }
    )

    # Neural - all neural data (spikes + LFP + multiunit)
    NEURAL: frozenset[str] = MINIMAL | frozenset(
        {
            SessionDataKeys.RIPPLE_TIMES,
            SessionDataKeys.RIPPLE_FILTERED_LFPS,
            SessionDataKeys.RIPPLE_CONSENSUS_TRACE,
            SessionDataKeys.ZSCORED_RIPPLE_CONSENSUS_TRACE,
            SessionDataKeys.THETA_FILTERED_LFP,
            SessionDataKeys.THETA_PHASE,
            SessionDataKeys.THETA_POWER,
            SessionDataKeys.MULTIUNIT_FIRING_RATE,
            SessionDataKeys.HSE_TIMES,
        }
    )

    # Behavioral - position + task events + trials (no neural data)
    BEHAVIORAL: frozenset[str] = frozenset(
        {
            SessionDataKeys.POSITION_INFO,
            SessionDataKeys.TRACK_GRAPH,
            SessionDataKeys.LINEAR_EDGE_ORDER,
            SessionDataKeys.LINEAR_EDGE_SPACING,
            SessionDataKeys.TRIALS,
            SessionDataKeys.TASK_VARIABLES,
            SessionDataKeys.WELL_VISITS,
            SessionDataKeys.BEAM_BREAKS,
            SessionDataKeys.PUMP_EVENTS,
            SessionDataKeys.LIGHT_EVENTS,
        }
    )

    # Decoding - everything needed for decoding analysis
    DECODING: frozenset[str] = NEURAL | frozenset(
        {
            SessionDataKeys.TRIALS,
            SessionDataKeys.TASK_VARIABLES,
            SessionDataKeys.ELECTRODE_GROUP_INFO,
        }
    )


class ClusterlessParams(TypedDict):
    """Parameters for clusterless decoder (used for HPC).

    Attributes
    ----------
    position_std : float
        Standard deviation for position likelihood in cm. Controls spatial
        uncertainty in the decoding model.
    waveform_std : float
        Standard deviation for waveform feature likelihood in µV. Controls
        how tightly spikes must match waveform features to contribute to
        the posterior probability.
    block_size : int
        Block size for GPU memory management during decoding. Controls how
        many time bins are processed simultaneously. Should be a power of 2
        for optimal GPU performance.

    """

    position_std: float
    waveform_std: float
    block_size: int


class SortedSpikesParams(TypedDict):
    """Parameters for sorted spikes decoder (used for mPFC/OFC).

    Attributes
    ----------
    position_std : float
        Standard deviation for position likelihood in cm. Controls spatial
        uncertainty in the decoding model.
    block_size : int
        Block size for GPU memory management during decoding. Controls how
        many time bins are processed simultaneously. Should be a power of 2
        for optimal GPU performance.

    """

    position_std: float
    block_size: int


# Data loader return types
# =========================


class PositionInfoDict(TypedDict):
    """Return type for get_position_info() in data_loaders.position.

    Attributes
    ----------
    position_info : pd.DataFrame
        Time-indexed DataFrame with position tracking data
    linear_edge_order : list[tuple[int, int]]
        List of edges for linearization in order
    linear_edge_spacing : list[float]
        List of edge spacings for linearization
    track_graph : nx.Graph
        NetworkX graph representing track topology

    """

    position_info: pd.DataFrame
    linear_edge_order: list[tuple[int, int]]
    linear_edge_spacing: list[float]
    track_graph: nx.Graph


class SpikeDataDict(TypedDict):
    """Return type for get_spike_data() in data_loaders.spikes.

    Attributes
    ----------
    spike_times : dict[str, list[NDArray[np.float64]]]
        Spike times organized by brain area (e.g., "HPC", "mPFC", "OFC").
        Each area contains a list of arrays, one per sort group/unit.
    spike_waveform_features : dict[str, list[NDArray[np.float64]]]
        Waveform features for clusterless decoding (primarily HPC).
        Shape: (n_spikes, n_features) per sort group.

    """

    spike_times: dict[str, list[NDArray[np.float64]]]
    spike_waveform_features: dict[str, list[NDArray[np.float64]]]


class DIOEventData(TypedDict):
    """Return type for load_dios() in data_loaders.events.

    Attributes
    ----------
    beam_breaks : pd.DataFrame
        DataFrame with beam break events (poke-in/poke-out)
    pump_events : pd.DataFrame
        DataFrame with reward pump events
    light_events : pd.DataFrame
        DataFrame with light on/off events

    """

    beam_breaks: pd.DataFrame
    pump_events: pd.DataFrame
    light_events: pd.DataFrame


class ThetaData(TypedDict):
    """Return type for get_theta() in data_loaders.lfp.

    Attributes
    ----------
    theta_filtered_lfp : pd.DataFrame
        Theta-band filtered LFP data
    theta_phase : pd.Series
        Theta phase timeseries
    theta_power : pd.Series
        Theta power timeseries

    """

    theta_filtered_lfp: pd.DataFrame
    theta_phase: pd.Series
    theta_power: pd.Series


class RippleData(TypedDict):
    """Return type for get_ripple() in data_loaders.lfp.

    Attributes
    ----------
    ripple_times : pd.DataFrame
        DataFrame with ripple start/end times
    ripple_filtered_lfps : pd.DataFrame
        Ripple-band filtered LFP data
    ripple_lfps : pd.DataFrame
        Raw LFP data for ripple band
    ripple_consensus_trace : pd.Series
        Ripple consensus trace (combined signal across electrodes)
    zscored_ripple_consensus_trace : pd.Series
        Z-scored ripple consensus trace

    """

    ripple_times: pd.DataFrame
    ripple_filtered_lfps: pd.DataFrame
    ripple_lfps: pd.DataFrame
    ripple_consensus_trace: pd.Series
    zscored_ripple_consensus_trace: pd.Series


class MultiunitData(TypedDict):
    """Return type for get_multiunit() in data_loaders.lfp.

    Attributes
    ----------
    hse_times : dict[str, pd.DataFrame]
        High synchrony event times for each brain area
    multiunit_firing_rate : dict[str, pd.DataFrame]
        Population firing rates for each brain area
    multiunit_rate_zscore : dict[str, pd.DataFrame]
        Z-scored firing rates for each brain area

    """

    hse_times: dict[str, pd.DataFrame]
    multiunit_firing_rate: dict[str, pd.DataFrame]
    multiunit_rate_zscore: dict[str, pd.DataFrame]


class WellNodeData(TypedDict):
    """Return type for get_start_end_node_id() in data_loaders.trials.

    Attributes
    ----------
    from_well : NDArray[np.int_]
        Array of starting well node IDs
    to_well : NDArray[np.int_]
        Array of ending well node IDs

    """

    from_well: NDArray[np.int_]
    to_well: NDArray[np.int_]


__all__ = [
    # Main data structures
    "SessionData",
    "SessionDataKeys",
    "DataLoadingPreset",
    # Decoding parameters
    "ClusterlessParams",
    "SortedSpikesParams",
    # Data loader return types
    "PositionInfoDict",
    "SpikeDataDict",
    "DIOEventData",
    "ThetaData",
    "RippleData",
    "MultiunitData",
    "WellNodeData",
]

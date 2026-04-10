"""LFP, ripple, theta, and multiunit data loading.

This module provides functions for loading local field potential (LFP) data from
Spyglass/NWB files, including theta oscillations, ripple events, and multiunit
activity. All data is extracted from the Spyglass DataJoint pipeline and returned
as time-indexed pandas DataFrames or TypedDicts.

Main Functions
--------------
get_theta
    Load theta-band (5-11 Hz) filtered LFP with phase and power estimates.
get_ripple
    Load ripple-band (150-250 Hz) filtered LFP and detected ripple events.
get_multiunit
    Compute multiunit high-synchrony events (HSE) from spike times across brain areas.

Helper Functions
----------------
get_multiunit_hse
    Compute HSE detection for a single brain area's spike data.
filter_ripple_lfp_time
    Filter ripple data to match position time bounds.

Data Sources
------------
The module queries the following Spyglass DataJoint tables:
- LFPBandV1: Band-pass filtered LFP data
- LFPOutput: Raw LFP traces
- RippleTimesV1: Detected ripple events using Kay detector

Notes
-----
- All time indices are in seconds
- Theta phase is computed from the first reference electrode
- Ripple consensus trace combines multiple electrodes for robust detection
- HSE detection uses the ripple_detection package with default thresholds
- Z-scored traces handle constant values to avoid division by zero

See Also
--------
continuum_swr_replay.types : ThetaData, RippleData, MultiunitData TypedDicts
continuum_swr_replay.data_loaders.spikes : Spike data loading

Examples
--------
>>> theta_data = get_theta("chimi20200212_.nwb", "02_r1")
>>> ripple_data = get_ripple("chimi20200212_.nwb", "02_r1", "Kay")
>>> multiunit_data = get_multiunit(spike_times, position_info)

"""

import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from ripple_detection import (
    get_multiunit_population_firing_rate,
    multiunit_HSE_detector,
)
from scipy.stats import zscore
from spyglass.lfp.analysis.v1.lfp_band import LFPBandSelection, LFPBandV1
from spyglass.lfp.lfp_merge import LFPOutput
from spyglass.ripple.v1 import RippleTimesV1

from ..parameters import (
    HSE_CLOSE_EVENT_THRESHOLD_SEC,
    HSE_MINIMUM_DURATION_SEC,
    HSE_ZSCORE_THRESHOLD,
    MAX_SPEED_THRESHOLD_CM_PER_SEC,
)
from ..types import MultiunitData, RippleData, ThetaData

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pynwb")
warnings.filterwarnings("ignore", category=UserWarning, module="hdmf")
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint")


def get_theta(nwb_file_name: str, epoch_name: str) -> ThetaData:
    """Load theta-band LFP data.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name
    epoch_name : str
        Epoch name

    Returns
    -------
    ThetaData
        TypedDict containing:
        - theta_filtered_lfp: pd.DataFrame with filtered LFP
        - theta_phase: pd.Series with theta phase
        - theta_power: pd.Series with theta power

    """
    theta_lfp_key = {
        "nwb_file_name": nwb_file_name,
        "filter_name": "theta_5_11",
        "filter_sampling_rate": 1000,
        "target_interval_list_name": epoch_name + " noPrePostTrialTimes",
    }
    theta_table = LFPBandV1 & theta_lfp_key
    theta_filtered_lfp = theta_table.fetch1_dataframe()
    # Pick the first reference electrode
    electrode_list = [
        theta_table.fetch_nwb()[0]["lfp_band"].electrodes.data[0]
    ]
    theta_phase = theta_table.compute_signal_phase(
        electrode_list=electrode_list
    )
    theta_power = theta_table.compute_signal_power(
        electrode_list=electrode_list
    )
    return {
        "theta_filtered_lfp": theta_filtered_lfp,
        "theta_phase": theta_phase,
        "theta_power": theta_power,
    }


def get_ripple(
    nwb_file_name: str, epoch_name: str, ripple_detector_name: str
) -> RippleData:
    """Load ripple detection data.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name
    epoch_name : str
        Epoch name
    ripple_detector_name : str
        Ripple detector name (currently only "Kay" is supported)

    Returns
    -------
    RippleData
        TypedDict containing:
        - ripple_times: pd.DataFrame with ripple start/end times
        - ripple_filtered_lfps: pd.DataFrame with ripple-band filtered LFP
        - ripple_lfps: pd.DataFrame with raw ripple LFP
        - ripple_consensus_trace: pd.Series with ripple consensus trace
        - zscored_ripple_consensus_trace: pd.Series with z-scored consensus trace

    """
    # Ripples
    if ripple_detector_name == "Kay":
        ripple_params = "default"
    else:
        raise ValueError("Invalid ripple detector name")

    ripple_key = {
        "nwb_file_name": nwb_file_name,
        "target_interval_list_name": epoch_name + " noPrePostTrialTimes",
        "group_name": "CA1",
        "ripple_params": ripple_params,
        "filter_name": "ripple_150_250",
        "filter_sampling_rate": 1000,
    }
    lfp_band_restriction = {
        "nwb_file_name": nwb_file_name,
        "filter_name": ripple_key["filter_name"],
        "target_interval_list_name": ripple_key["target_interval_list_name"],
        "lfp_band_sampling_rate": ripple_key["filter_sampling_rate"],
    }
    lfp_band_key = (LFPBandV1 & lfp_band_restriction).fetch1("KEY")
    ripple_times = (RippleTimesV1 & ripple_key).fetch1_dataframe()
    ripple_filtered_lfp = (LFPBandV1 & lfp_band_key).fetch1_dataframe()
    lfp_merge_id = (LFPBandSelection & ripple_key).fetch1("lfp_merge_id")
    lfp_df = (LFPOutput & {"merge_id": lfp_merge_id}).fetch1_dataframe()

    ripple_consensus_trace = RippleTimesV1.get_Kay_ripple_consensus_trace(
        ripple_filtered_lfp, sampling_frequency=ripple_key["filter_sampling_rate"]
    )
    zscored_ripple_consensus_trace = zscore(ripple_consensus_trace, nan_policy="omit")
    zscored_ripple_consensus_trace = pd.Series(
        zscored_ripple_consensus_trace.squeeze(),
        index=ripple_consensus_trace.index,
        name="zscored_ripple_consensus_trace",
    )

    return {
        "ripple_times": ripple_times,
        "ripple_filtered_lfps": ripple_filtered_lfp,
        "ripple_lfps": lfp_df,
        "ripple_consensus_trace": ripple_consensus_trace,
        "zscored_ripple_consensus_trace": zscored_ripple_consensus_trace,
    }


def get_multiunit_hse(
    spike_times: list[NDArray[np.float64]],
    speed: NDArray[np.float64],
    bin_time: NDArray[np.float64],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute multiunit high synchrony events (HSE).

    Parameters
    ----------
    spike_times : list[NDArray[np.float64]]
        List of spike time arrays for all units
    speed : NDArray[np.float64]
        Speed of the animal in cm/s
    bin_time : NDArray[np.float64]
        Time bins for spike counting in seconds

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        - hse_times: pd.DataFrame with HSE start/end times
        - multiunit_firing_rate: pd.DataFrame with population firing rate
        - multiunit_rate_zscore: pd.DataFrame with z-scored firing rate

    """
    sampling_frequency = 1 / np.median(np.diff(bin_time))
    start_time, end_time = bin_time[[0, -1]]
    multiunit_spikes = np.stack(
        [
            np.bincount(
                np.digitize(
                    unit_spike_times[
                        np.logical_and(
                            unit_spike_times >= start_time,
                            unit_spike_times <= end_time,
                        )
                    ],
                    bin_time[1:-1],
                ),
                minlength=bin_time.shape[0],
            )
            for unit_spike_times in spike_times
        ],
        axis=1,
    )

    multiunit_firing_rate = pd.DataFrame(
        get_multiunit_population_firing_rate(multiunit_spikes, sampling_frequency),
        index=bin_time,
        columns=["firing_rate"],
    )
    mean = np.nanmean(multiunit_firing_rate)
    std = np.nanstd(multiunit_firing_rate)

    # Avoid division by near-zero if firing rate is constant
    if std < 1e-10 or np.isnan(std):
        multiunit_rate_zscore = pd.DataFrame(
            np.zeros_like(multiunit_firing_rate),
            index=bin_time,
            columns=["firing_rate"],
        )
    else:
        multiunit_rate_zscore = (multiunit_firing_rate - mean) / std

    hse_times = multiunit_HSE_detector(
        bin_time,
        multiunit_spikes,
        speed,
        sampling_frequency,
        speed_threshold=MAX_SPEED_THRESHOLD_CM_PER_SEC,
        minimum_duration=HSE_MINIMUM_DURATION_SEC,
        zscore_threshold=HSE_ZSCORE_THRESHOLD,
        close_event_threshold=HSE_CLOSE_EVENT_THRESHOLD_SEC,
        use_speed_threshold_for_zscore=False,
    )

    return hse_times, multiunit_firing_rate, multiunit_rate_zscore


def get_multiunit(
    spike_times: dict[str, list[NDArray[np.float64]]], position_info: pd.DataFrame
) -> MultiunitData:
    """Get multiunit HSE data for all brain areas.

    Parameters
    ----------
    spike_times : dict[str, list[NDArray[np.float64]]]
        Dictionary mapping brain area to list of spike time arrays
    position_info : pd.DataFrame
        Position info with head_speed column and time index

    Returns
    -------
    MultiunitData
        TypedDict containing:
        - hse_times: dict mapping brain area to HSE times
        - multiunit_firing_rate: dict mapping brain area to firing rates
        - multiunit_rate_zscore: dict mapping brain area to z-scored rates

    """
    # Get multiunit HSE
    hse_times = {}
    multiunit_firing_rate = {}
    multiunit_rate_zscore = {}

    for brain_area in spike_times:
        (
            hse_times[brain_area],
            multiunit_firing_rate[brain_area],
            multiunit_rate_zscore[brain_area],
        ) = get_multiunit_hse(
            spike_times[brain_area],
            position_info.head_speed.to_numpy(),
            position_info.index.to_numpy(),
        )

    return {
        "hse_times": hse_times,
        "multiunit_firing_rate": multiunit_firing_rate,
        "multiunit_rate_zscore": multiunit_rate_zscore,
    }


def filter_ripple_lfp_time(
    ripple_data: RippleData, position_time: NDArray[np.float64]
) -> RippleData:
    """Filter ripple LFP data to match position time bounds.

    Parameters
    ----------
    ripple_data : RippleData
        Dictionary with ripple data
    position_time : NDArray[np.float64]
        Position time array in seconds (defines time bounds)

    Returns
    -------
    RippleData
        New dictionary with filtered ripple data matching position time bounds

    """
    position_time_slice = slice(position_time[0], position_time[-1])

    ripple_times = ripple_data["ripple_times"]
    ripple_times = ripple_times[
        (ripple_times["start_time"] >= position_time[0])
        & (ripple_times["end_time"] <= position_time[-1])
    ]

    return {
        "ripple_times": ripple_times,
        "ripple_filtered_lfps": ripple_data["ripple_filtered_lfps"].loc[
            position_time_slice
        ],
        "ripple_lfps": ripple_data["ripple_lfps"].loc[position_time_slice],
        "ripple_consensus_trace": ripple_data["ripple_consensus_trace"].loc[
            position_time_slice
        ],
        "zscored_ripple_consensus_trace": ripple_data[
            "zscored_ripple_consensus_trace"
        ].loc[position_time_slice],
    }

"""Spike data loading and processing.

This module handles loading neural spiking data from different brain regions
and processing it for decoding analysis. Supports both clusterless (waveform-based)
and sorted spike data.

Brain Regions
-------------
- **HPC (Hippocampus CA1)**: Clusterless thresholding with waveform features
- **mPFC (Medial Prefrontal Cortex)**: MountainSort4 sorted units
- **OFC (Orbitofrontal Cortex)**: MountainSort4 sorted units

Functions
---------
detect_coincident_spikes
    Identify and remove artifact spikes that fire simultaneously across tetrodes.
get_electrode_group_info
    Extract electrode group metadata from NWB file.
get_hpc_marks
    Load clusterless marks (spike times + waveform features) for hippocampus.
get_pfc_spike_times
    Load sorted unit spike times for prefrontal cortex areas.
get_spike_data
    Main function to load spike data from all available brain regions.
filter_spike_times
    Restrict spike times to match position data time bounds.

Notes
-----
Clusterless decoding for HPC uses spike waveform features rather than
cluster assignments. This captures multiunit activity without the need
for spike sorting.

Coincident spike detection removes electrical artifacts where >33% of
tetrodes fire within 40 microseconds - timing unlikely for genuine neural
activity.

See Also
--------
load_data : Main data loading orchestrator
position : Position data that defines time bounds for spike filtering

"""

from __future__ import annotations

import itertools
import logging
import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.ndimage import label
from spyglass.common import Nwbfile
from spyglass.decoding.v0.clusterless import UnitMarks
from spyglass.spikesorting.v0 import CuratedSpikeSorting, SortGroup
from spyglass.utils.nwb_helper_fn import get_nwb_file

from ..types import SpikeDataDict

logger = logging.getLogger(__name__)

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pynwb")
warnings.filterwarnings("ignore", category=UserWarning, module="hdmf")
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint")


def detect_coincident_spikes(
    spike_times: list[NDArray[np.float64]],
    spike_closeness_threshold: float = 0.00004,
    max_coincident_fraction: float = 0.33,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.intp]]]:
    """Detect and remove coincident spikes across tetrodes.

    Coincident spikes (within threshold across >fraction of tetrodes) are likely
    electrical artifacts rather than biological neural activity.

    Parameters
    ----------
    spike_times : list[NDArray[np.float64]]
        List of arrays, each containing spike times (seconds) for one tetrode.
    spike_closeness_threshold : float, optional
        Time threshold for considering spikes coincident (seconds). Default 40μs
        (0.00004s) based on neural propagation delays being ~1-10 ms; spikes
        within 40μs across multiple channels indicate electrical artifact.
    max_coincident_fraction : float, optional
        Fraction of tetrodes that must fire together to be considered artifact.
        Default 0.33 (33%) based on real neural synchrony rarely exceeding 20%
        of channels simultaneously.

    Returns
    -------
    tuple[list[NDArray[np.float64]], list[NDArray[np.intp]]]
        - filtered_spike_times: list of arrays with coincident spikes removed,
          one per input sort group (preserves 1:1 alignment with input).
          Sort groups that lose all spikes get empty arrays.
          Returns empty list if input is empty or contains only empty arrays.
        - filtered_time_bin_ind: list of arrays with indices of kept spikes,
          one per input sort group (preserves 1:1 alignment with input).
          Returns empty list if input is empty or contains only empty arrays.

    Notes
    -----
    Returns empty lists ([], []) when input is empty or all arrays are empty.
    Downstream code should check for empty results before further processing.

    """
    if not spike_times or all(len(st) == 0 for st in spike_times):
        return [], []

    concat_spike_times = np.concatenate(spike_times)
    sort_group_id = np.concatenate(
        [
            np.ones(len(spike_time), dtype=int) * i
            for i, spike_time in enumerate(spike_times)
        ]
    )
    time_bin_ind = np.concatenate(
        [np.arange(len(spike_time), dtype=int) for spike_time in spike_times]
    )

    sort_ind = np.argsort(concat_spike_times)
    sorted_spike_times = concat_spike_times[sort_ind]
    sort_group_id = sort_group_id[sort_ind]
    time_bin_ind = time_bin_ind[sort_ind]

    is_close = np.diff(sorted_spike_times) < spike_closeness_threshold
    is_close = np.concatenate([[False], is_close])
    labels, _ = label(is_close)

    spike_events = pd.DataFrame(
        {
            "labels": labels,
            "sort_group_id": sort_group_id,
            "time_bin_ind": time_bin_ind,
            "spike_times": sorted_spike_times,
        },
    )
    # Calculate coincident fraction to identify artifacts
    n_sort_groups = len(spike_times)
    coincident_fraction = (
        spike_events.loc[spike_events.labels > 0]
        .groupby("labels")
        .sort_group_id.nunique()
        / n_sort_groups
    )
    artifact_labels = coincident_fraction[coincident_fraction > max_coincident_fraction]

    # Build mask of spikes belonging to artifact groups.
    # scipy.ndimage.label assigns label=0 to the first spike in each cluster
    # (because np.diff only looks backward), so we must also remove the spike
    # immediately preceding each artifact-labeled run.
    is_artifact = spike_events.labels.isin(artifact_labels.index)
    # Shift forward to also flag the preceding (label=0) spike that starts each cluster
    is_artifact = is_artifact | is_artifact.shift(-1, fill_value=False)
    spike_events = spike_events.loc[~is_artifact]

    grouped = spike_events.groupby("sort_group_id")
    grouped_spike_times = grouped.spike_times.apply(np.array)
    grouped_time_bin_ind = grouped.time_bin_ind.apply(np.array)

    # Preserve all original sort group indices so the output aligns 1:1 with the input.
    # Sort groups that lost all spikes get empty arrays.
    filtered_spike_times = [
        grouped_spike_times[i] if i in grouped_spike_times.index else np.array([], dtype=np.float64)
        for i in range(n_sort_groups)
    ]
    filtered_time_bin_ind = [
        grouped_time_bin_ind[i] if i in grouped_time_bin_ind.index else np.array([], dtype=np.intp)
        for i in range(n_sort_groups)
    ]

    return filtered_spike_times, filtered_time_bin_ind


def get_electrode_group_info(nwb_file_name: str) -> pd.DataFrame:
    """Get electrode group information from NWB file.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name

    Returns
    -------
    pd.DataFrame
        Electrode group information with targeted location labels

    """
    nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
    nwb_file = get_nwb_file(nwb_file_abspath)
    electrode_group_df = []
    for electrode_group in nwb_file.electrode_groups.values():
        group_info = {
            "electrode_group_name": electrode_group.name,
            "description": electrode_group.description,
        }
        group_info.update(electrode_group.fields)
        electrode_group_df.append(group_info)

    electrode_group_df = (
        pd.DataFrame(electrode_group_df)
        .drop(columns=["device"])
        .set_index("electrode_group_name")
    )

    is_CA1 = electrode_group_df.targeted_location.str.contains("CA1") & (
        electrode_group_df.location != "CorpusCallosum"
    )
    electrode_group_df.loc[is_CA1, "targeted_location"] = "CA1"

    is_mPFC = electrode_group_df.targeted_location.str.contains("mPFC") & (
        electrode_group_df.location != "CorpusCallosum"
    )
    electrode_group_df.loc[is_mPFC, "targeted_location"] = "mPFC"

    is_OFC = electrode_group_df.targeted_location.str.contains("OFC") & (
        electrode_group_df.location != "CorpusCallosum"
    )
    electrode_group_df.loc[is_OFC, "targeted_location"] = "OFC"

    return electrode_group_df


_MAX_TETRODE_ELECTRODES = 4


def _get_tetrode_sort_group_ids(
    nwb_file_name: str,
    restriction: dict,
) -> set[int]:
    """Get sort group IDs that correspond to tetrodes (<=4 electrodes).

    Non-tetrode sort groups (e.g., 32-channel probes) can be processed with
    tetrode preprocessing parameters but produce high-dimensional waveform
    features that are incompatible with clusterless decoding calibrated for
    tetrode marks.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name.
    restriction : dict
        UnitMarks restriction dict.

    Returns
    -------
    set[int]
        Sort group IDs with <= 4 electrodes.

    """
    unit_mark_keys = pd.DataFrame(
        (UnitMarks & restriction).fetch("sort_group_id", as_dict=True)
    )
    sort_group_ids = unit_mark_keys["sort_group_id"].unique()

    tetrode_ids = set()
    for sg_id in sort_group_ids:
        sg_restriction = {"nwb_file_name": nwb_file_name, "sort_group_id": sg_id}
        n_electrodes = len(SortGroup.SortGroupElectrode & sg_restriction)
        if n_electrodes <= _MAX_TETRODE_ELECTRODES:
            tetrode_ids.add(int(sg_id))
        else:
            logger.warning(
                "Excluding sort group %d from %s: %d electrodes (not a tetrode)",
                sg_id, nwb_file_name, n_electrodes,
            )

    return tetrode_ids


def get_hpc_marks(
    nwb_file_name: str,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Get hippocampus clusterless marks (spike times and waveform features).

    Only includes sort groups with <= 4 electrodes (tetrodes). Sort groups
    with more electrodes (e.g., 32-channel probes) are excluded because their
    high-dimensional waveform features are incompatible with the clusterless
    decoding kernel bandwidth (``waveform_std``) calibrated for tetrode marks.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name

    Returns
    -------
    tuple
        - spike_times: list of arrays with spike times per tetrode
        - spike_waveform_features: list of arrays with waveform features

    """
    # Base restriction shared by both parameter options
    restriction_base = {
        "nwb_file_name": nwb_file_name,
        "curation_id": 0,
        "sort_interval_name": "runs_noPrePostTrialTimes raw data valid times",
        "preproc_params_name": "franklab_tetrode_hippocampus",
        "team_name": "ac_em_xs",
        "sorter": "clusterless_thresholder",
        "mark_param_name": "default",
    }

    # Check which parameter name exists (cheap query)
    restriction_fixed = {**restriction_base, "sorter_params_name": "clusterless_fixed"}
    if len(UnitMarks & restriction_fixed) > 0:
        restriction = restriction_fixed
    else:
        # Fall back to alternative parameter name
        restriction = {**restriction_base, "sorter_params_name": "default_clusterless"}

    # Filter to tetrode sort groups only
    tetrode_sg_ids = _get_tetrode_sort_group_ids(nwb_file_name, restriction)
    if not tetrode_sg_ids:
        raise ValueError(
            f"No tetrode sort groups found for {nwb_file_name!r}"
        )

    tetrode_restriction = [
        {**restriction, "sort_group_id": sg_id} for sg_id in sorted(tetrode_sg_ids)
    ]

    # Fetch data once with the correct parameter
    marks = (UnitMarks & tetrode_restriction).fetch_dataframe()
    if len(marks) == 0:
        raise ValueError(
            f"No UnitMarks found for {nwb_file_name!r} with either "
            "'clusterless_fixed' or 'default_clusterless' sorter_params_name"
        )
    marks = [(mark.index.to_numpy(), mark.to_numpy()) for mark in marks]
    # Unpack list of (spike_times, waveform_features) tuples into separate tuples
    spike_times_tuple, spike_waveform_features_tuple = zip(*marks, strict=True)
    spike_times = list(spike_times_tuple)
    spike_waveform_features = list(spike_waveform_features_tuple)

    spike_times, filtered_time_bin_ind = detect_coincident_spikes(spike_times)
    spike_waveform_features = [
        features[ind]
        for ind, features in zip(
            filtered_time_bin_ind, spike_waveform_features, strict=True
        )
    ]

    return spike_times, spike_waveform_features


def get_pfc_spike_times(
    nwb_file_name: str, brain_area: str
) -> list[NDArray[np.float64]]:
    """Get prefrontal cortex sorted spike times.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name
    brain_area : str
        Brain area name ("mPFC" or "OFC")

    Returns
    -------
    list
        List of arrays with spike times per unit

    """
    restriction = {
        "nwb_file_name": nwb_file_name,
        "preproc_params_name": "default",
        "sort_interval_name": "sleeps_runs_noPrePostTrialTimes raw data valid times",
        "sorter_params_name": "franklab_probe_ctx_30KHz_115rad_new_mountainsort2",
        "team_name": "ac_em_xs",
        "sorter": "mountainsort4",
    }

    curation_ids = (CuratedSpikeSorting & restriction).fetch("curation_id")
    if len(curation_ids) == 0:
        raise ValueError(
            f"No CuratedSpikeSorting rows found for {nwb_file_name!r} "
            f"with brain_area={brain_area!r}"
        )
    max_curation_id = curation_ids.max()
    restriction.update({"curation_id": max_curation_id})

    curated_spikes_info = pd.DataFrame(
        (CuratedSpikeSorting & restriction) * SortGroup.SortGroupElectrode
    )
    electrode_group_df = get_electrode_group_info(nwb_file_name)
    curated_spikes_info = pd.merge(
        curated_spikes_info, electrode_group_df, on="electrode_group_name"
    )
    spikesorting_keys = pd.merge(
        pd.DataFrame(CuratedSpikeSorting() & restriction),
        curated_spikes_info.groupby("sort_group_id").targeted_location.first(),
        on="sort_group_id",
    )
    spikesorting_keys = spikesorting_keys.loc[
        spikesorting_keys.targeted_location == brain_area
    ].to_dict(orient="records")

    nwb_pfc = (CuratedSpikeSorting() & spikesorting_keys).fetch_nwb()

    return list(
        itertools.chain.from_iterable(
            [
                file["units"]["spike_times"].to_list()
                for file in nwb_pfc
                if "units" in file
            ]
        )
    )


def get_spike_data(nwb_file_name: str) -> SpikeDataDict:
    """Load spike data for all available brain areas.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name

    Returns
    -------
    dict
        Dictionary containing:
        - spike_times: dict mapping brain area to list of spike time arrays
        - spike_waveform_features: dict mapping brain area to list of feature arrays

    """
    spike_times = {}
    spike_waveform_features = {}

    try:
        spike_times["HPC"], spike_waveform_features["HPC"] = get_hpc_marks(
            nwb_file_name
        )
    except ValueError as e:
        logger.debug("No HPC data available for %s: %s", nwb_file_name, e)

    for brain_area in ["mPFC", "OFC"]:
        try:
            spike_times[brain_area] = get_pfc_spike_times(nwb_file_name, brain_area)
            if len(spike_times[brain_area]) < 1:
                del spike_times[brain_area]
        except ValueError as e:
            logger.debug("No %s data available for %s: %s", brain_area, nwb_file_name, e)

    return {
        "spike_times": spike_times,
        "spike_waveform_features": spike_waveform_features,
    }


def filter_spike_times(
    spike_times: dict[str, list[NDArray[np.float64]]],
    spike_waveform_features: dict[str, list[NDArray[np.float64]]],
    position_time: NDArray[np.float64],
) -> tuple[dict[str, list[NDArray[np.float64]]], dict[str, list[NDArray[np.float64]]]]:
    """Filter spike times to match position data time bounds.

    Restricts spikes to the position time range and drops tetrodes/units
    that have zero spikes after filtering (empty arrays cause downstream
    errors in the non_local_detector encoding model).

    Parameters
    ----------
    spike_times : dict
        Dictionary mapping brain area to list of spike time arrays
    spike_waveform_features : dict
        Dictionary mapping brain area to list of feature arrays
    position_time : np.ndarray
        Position time array (defines time bounds)

    Returns
    -------
    filtered_spike_times : dict
        Dictionary mapping brain area to list of filtered spike time arrays.
        Tetrodes/units with zero spikes after filtering are excluded.
    filtered_spike_waveform_features : dict
        Dictionary mapping brain area to list of filtered feature arrays.
        Entries correspond 1:1 with filtered_spike_times.

    """
    filtered_spike_times = {}
    filtered_spike_waveform_features = {}

    for brain_area, brain_area_spike_times in spike_times.items():
        brain_area_filtered_times = []
        brain_area_filtered_features = []
        has_features = brain_area in spike_waveform_features
        n_dropped = 0

        for sort_ind, sort_group_spike_times in enumerate(brain_area_spike_times):
            is_in_bounds = np.logical_and(
                sort_group_spike_times >= position_time[0],
                sort_group_spike_times <= position_time[-1],
            )
            filtered = sort_group_spike_times[is_in_bounds]

            # Drop tetrodes/units with no spikes in the time range
            if len(filtered) == 0:
                n_dropped += 1
                continue

            brain_area_filtered_times.append(filtered)

            if has_features:
                brain_area_filtered_features.append(
                    spike_waveform_features[brain_area][sort_ind][is_in_bounds]
                )

        if n_dropped > 0:
            logger.info(
                "Dropped %d/%d %s tetrodes/units with no spikes in time range",
                n_dropped,
                len(brain_area_spike_times),
                brain_area,
            )

        filtered_spike_times[brain_area] = brain_area_filtered_times

        if brain_area_filtered_features:
            filtered_spike_waveform_features[brain_area] = brain_area_filtered_features

    return filtered_spike_times, filtered_spike_waveform_features

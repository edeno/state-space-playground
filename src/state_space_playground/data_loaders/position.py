"""Position data loading and linearization.

This module handles loading position data from Spyglass database and performing
spatial linearization along track graphs. Position data includes raw 2D coordinates,
projected positions on the track, and 1D linear position along graph edges.

Functions
---------
get_interpolated_position_info
    Interpolate position data to new time points and compute linear position.
get_position_info
    Load and process position data for a given session and epoch.

Notes
-----
Position tracking uses DLC (DeepLabCut) when available, falling back to older
tracking methods. All position data is linearized to 1D along the track graph
for decoding analysis.

See Also
--------
load_data : Main data loading orchestrator that uses these functions
trials : Trial construction that depends on position data

"""

from __future__ import annotations

import logging
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from datajoint import DataJointError
from numpy.typing import NDArray
from spyglass.common import IntervalList
from spyglass.common.common_position import IntervalPositionInfo
from spyglass.linearization.v0.main import IntervalLinearizedPosition, TrackGraph
from spyglass.position import PositionOutput
from track_linearization import get_linearized_position

from ..types import PositionInfoDict
from .constants import TRACK_SEGMENT_TO_PATCH

logger = logging.getLogger(__name__)

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pynwb")
warnings.filterwarnings("ignore", category=UserWarning, module="hdmf")
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint")


def get_interpolated_position_info(
    position_info: pd.DataFrame,
    time: NDArray[np.float64],
    track_graph: nx.Graph,
    edge_order: list[tuple[int, int]],
    edge_spacing: list[float],
    position_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Interpolate position info to new time points and add linearization.

    Combines the original position data with new time points via linear
    interpolation, then computes linearized position along the track graph.
    This ensures position data is available at all desired time points
    (e.g., spike times, LFP timestamps).

    Parameters
    ----------
    position_info : pd.DataFrame
        Position data with time index (seconds) and columns for x, y coordinates.
    time : np.ndarray, shape (n_times,)
        New time points (seconds) for interpolation. These will be the final
        time points in the output.
    track_graph : nx.Graph
        NetworkX graph representing the track structure with nodes as wells
        and edges as track segments.
    edge_order : list of tuple of (int, int)
        Ordered list of graph edges (node pairs) defining the linearization path.
    edge_spacing : list of float
        Spacing values along each edge for linearization.
    position_columns : list of str, optional
        Names of columns containing x and y position coordinates.
        Default is ["head_position_x", "head_position_y"].

    Returns
    -------
    pd.DataFrame
        Interpolated position DataFrame with additional columns:
        - Original position columns (interpolated)
        - linear_position : 1D position along track (cm)
        - projected_x_position, projected_y_position : Position projected onto track
        - track_segment_id : ID of current track segment

    Notes
    -----
    The function merges original and new time points, interpolates linearly,
    then extracts only the requested time points. This avoids extrapolation
    artifacts at boundaries.

    See Also
    --------
    get_position_info : Main position loading function that calls this
    track_linearization.get_linearized_position : Underlying linearization function

    """
    if position_columns is None:
        position_columns = ["head_position_x", "head_position_y"]

    new_index = pd.Index(
        np.unique(np.concatenate((position_info.index, time))),
        name="time",
    )

    interpolated_position_info = (
        position_info.reindex(index=new_index)
        .interpolate(method="linear")
        .reindex(index=time)
    )

    linear_position_info = get_linearized_position(
        position=interpolated_position_info[position_columns].to_numpy(),
        track_graph=track_graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    ).set_index(interpolated_position_info.index)

    return pd.concat(
        (
            interpolated_position_info,
            linear_position_info,
        ),
        axis=1,
    )


def get_position_info(
    nwb_file_name: str, epoch_name: str, pos_name: str
) -> PositionInfoDict:
    """Load position data for a given session and epoch.

    Loads raw position tracking data from Spyglass database, performs quality
    control (drops NaN values), restricts to valid trial times, and computes
    linearized position along the track graph. Attempts to use DLC (DeepLabCut)
    tracking when available, falling back to older tracking methods.

    Parameters
    ----------
    nwb_file_name : str
        Name of NWB file (e.g., "chimi20200212_.nwb").
    epoch_name : str
        Epoch identifier in format "NN_rM" (e.g., "02_r1" for epoch 2, replicate 1).
    pos_name : str
        Position interval name from Spyglass PositionIntervalMap table.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'position_info' : pd.DataFrame
            Position data with time index and columns including:
            head_position_x, head_position_y (raw coordinates),
            linear_position (1D position along track),
            projected_x_position, projected_y_position (projected onto track),
            track_segment_id (current track segment),
            head_speed, head_orientation (if available),
            patch_id (current patch ID, 1-3).
        - 'linear_edge_order' : list of tuple of (int, int)
            Ordered edges for linearization.
        - 'linear_edge_spacing' : list of float
            Spacing along each edge.
        - 'track_graph' : nx.Graph
            NetworkX graph of the track with node and edge attributes.

    Notes
    -----
    The function first attempts to load DLC-tracked position from PositionOutput
    table. If unavailable, falls back to IntervalPositionInfo. Position data is
    then restricted to " noPrePostTrialTimes" intervals if available (excludes
    pre-trial and post-trial periods).

    Track graph is loaded based on linearization parameters or inferred from
    the animal name (first part of NWB filename).

    See Also
    --------
    get_interpolated_position_info : Interpolates position to new time points
    load_data : Main orchestrator that calls this function

    Examples
    --------
    >>> pos_data = get_position_info("chimi20200212_.nwb", "02_r1", "02_r1")
    >>> pos_data['position_info'].columns
    Index(['head_position_x', 'head_position_y', 'head_speed', ...])
    >>> pos_data['track_graph'].number_of_nodes()
    6  # Six wells in the environment

    """
    position_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": pos_name,
        "position_info_param_name": "default_decoding",
    }

    linearization_key = {
        "position_info_param_name": "default_decoding",
        "nwb_file_name": nwb_file_name,
        "interval_list_name": pos_name,
        "linearization_param_name": "default",
    }

    try:
        track_graph_name = (IntervalLinearizedPosition() & linearization_key).fetch1(
            "track_graph_name"
        )
    except DataJointError:
        track_graph_name = nwb_file_name.split("_")[0]

    track_graph = (
        TrackGraph() & {"track_graph_name": track_graph_name}
    ).get_networkx_track_graph()
    track_graph_params = (
        TrackGraph() & {"track_graph_name": track_graph_name}
    ).fetch1()
    linear_edge_order = track_graph_params["linear_edge_order"]
    linear_edge_spacing = track_graph_params["linear_edge_spacing"]

    try:
        epoch = int(epoch_name.split("_")[0])
        pos_merge_id = str(
            (
                PositionOutput().merge_restrict()
                & {
                    "nwb_file_name": nwb_file_name,
                    "source": "DLCPosV1",
                    "epoch": epoch,
                }
            ).fetch1("merge_id")
        )
        position_info = (
            (
                (PositionOutput() & {"merge_id": pos_merge_id})
                .fetch1_dataframe()
                .dropna()
            )
            .drop(columns="video_frame_ind")
            .add_prefix("head_")
        )
        time = (IntervalPositionInfo() & position_key).fetch1_dataframe().dropna().index

    except DataJointError:
        position_info = (
            (IntervalPositionInfo() & position_key).fetch1_dataframe().dropna()
        )
        time = position_info.index
    try:
        valid_interval_times = (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": epoch_name + " noPrePostTrialTimes",
            }
        ).fetch1("valid_times")
        position_info = position_info.loc[
            valid_interval_times[0][0] : valid_interval_times[-1][1]
        ]
    except DataJointError as e:
        logger.debug(
            "No noPrePostTrialTimes interval for %s %s: %s", nwb_file_name, epoch_name, e
        )

    position_info = get_interpolated_position_info(
        position_info,
        time,
        track_graph,
        linear_edge_order,
        linear_edge_spacing,
    )
    position_info["patch_id"] = position_info["track_segment_id"].map(
        TRACK_SEGMENT_TO_PATCH
    )

    return {
        "position_info": position_info,
        "linear_edge_order": linear_edge_order,
        "linear_edge_spacing": linear_edge_spacing,
        "track_graph": track_graph,
    }

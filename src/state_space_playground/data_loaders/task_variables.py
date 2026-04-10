"""Task-specific derived variables computation.

This module computes task-relevant behavioral variables for the spatial bandit task,
including trial timing, spatial progress, distance to goals, and turn sequences.
These variables are essential for analyzing neural activity in relation to behavior
and task structure.

Main Functions
--------------
make_task_variables
    Computes all task-specific derived variables for the entire session. This is the
    primary entry point that orchestrates all variable computations and returns a
    comprehensive DataFrame indexed by time.

Helper Functions
----------------
path_progress_for_run_segment
    Computes normalized progress (0-1) along a trial path using the track graph.
dist_to_goal
    Computes distance to goal well for all time points in a trial.
turn_direction
    Determines the sequence of left/right turns taken during a trial.

Computed Variables
------------------
The make_task_variables function computes:

Spatial variables:
    - path_progress: Normalized (0-1) progress along current trial path
    - dist_to_goal: Distance to goal well in graph distance units
    - start_node_id: Starting well ID for current trial
    - goal_node_id: Goal well ID for current trial

Temporal variables:
    - trial_number: Current trial index
    - time_to_goal: Time elapsed since trial start (seconds)

Behavioral variables:
    - turn_direction: Sequence of turns (e.g., "left-right-left")
    - is_no_reward_{N}: Reward omission indicators (N trials back, e.g., is_no_reward_1)
    - is_patch_change{N}: Patch change indicators (N trials relative, e.g., is_patch_change-1)

Notes
-----
- All variables are time-aligned and indexed by seconds
- Path progress and distance calculations use NetworkX graph shortest paths
- Turn directions are computed from track geometry using cross products
- Division by zero is handled when start/end nodes are the same
- Omission and patch change indicators support configurable trial shifts

See Also
--------
continuum_swr_replay.data_loaders.trials : Trial construction from task events
continuum_swr_replay.data_loaders.constants : Track graph structure and mappings
continuum_swr_replay.types : TaskVariables TypedDict

Examples
--------
>>> task_vars = make_task_variables(
...     position_info, trials_df, track_graph,
...     sampling_frequency=500.0
... )
>>> task_vars[["path_progress", "dist_to_goal", "turn_direction"]].head()

"""

from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd

# Opt in to future pandas behavior: fillna on object dtype will no longer
# silently downcast. This suppresses the FutureWarning.
pd.set_option("future.no_silent_downcasting", True)
from numpy.typing import NDArray
from scipy.interpolate import interp1d

# Minimum meaningful distance for path progress calculation (in cm).
# Distances below this threshold are treated as zero to avoid division
# by near-zero values. Set to 1e-10 cm (sub-picometer), which is far below
# any physically meaningful spatial resolution in behavioral tracking.
PATH_PROGRESS_EPSILON: float = 1e-10

# Default trial shifts for task variable indicators
# Omission shifts: look back N trials to check for reward omissions
DEFAULT_OMISSION_SHIFTS: tuple[int, ...] = tuple(range(1, 6))  # (1, 2, 3, 4, 5)
# Patch change shifts: look N trials relative to current (-5 to +5)
DEFAULT_PATCH_CHANGE_SHIFTS: tuple[int, ...] = tuple(range(-5, 6))  # (-5, -4, ..., 4, 5)


def path_progress_for_run_segment(
    pos_info: pd.DataFrame,
    track_graph: nx.Graph,
    start_node: int,
    end_node: int,
) -> np.ndarray:
    """Compute path progress (0-1) along a run segment.

    Parameters
    ----------
    pos_info : pd.DataFrame
        Position info for the segment
    track_graph : nx.Graph
        Track graph
    start_node : int
        Starting well node
    end_node : int
        Ending well node

    Returns
    -------
    np.ndarray
        Path progress values (0 to 1) for each time point

    """
    path = nx.shortest_path(track_graph, source=start_node, target=end_node)
    path_edges = list(zip(path[:-1], path[1:], strict=True))
    path_edge_ids = [track_graph.edges[edge]["edge_id"] for edge in path_edges]

    dist = np.full(pos_info.shape[0], np.nan)
    time = pos_info.index
    add: float = 0.0

    if len(path_edges) == 0:
        return np.zeros(pos_info.shape[0])

    for (start_node, end_node), edge_id in zip(path_edges, path_edge_ids, strict=True):
        is_edge = pos_info.track_segment_id == edge_id
        proj_pos = pos_info.loc[
            is_edge, ["projected_x_position", "projected_y_position"]
        ].to_numpy()
        start_node_pos = np.array(track_graph.nodes[start_node]["pos"])[None, :]
        dist[is_edge] = np.linalg.norm(proj_pos - start_node_pos, axis=1)
        dist[is_edge] += add
        add += float(
            np.linalg.norm(
                np.array(track_graph.nodes[start_node]["pos"])
                - np.array(track_graph.nodes[end_node]["pos"])
            )
        )

    is_nan = np.isnan(dist)
    if np.sum(~is_nan) >= 2:
        interpolater = interp1d(
            time[~is_nan], dist[~is_nan], fill_value="extrapolate", kind="previous"
        )
        dist[is_nan] = interpolater(time[is_nan])

    # Avoid division by zero if start and end nodes are the same
    max_dist = dist.max()
    # Handle both zero and very small values for numerical stability
    if max_dist <= PATH_PROGRESS_EPSILON or np.isnan(max_dist):
        return np.zeros(pos_info.shape[0])
    return np.asarray(np.clip(dist / max_dist, 0.0, 1.0))


def _precompute_segment_distances_to_goal(
    goal_node_id: int,
    track_graph: nx.Graph,
    track_segment_id_to_edge: dict[int, tuple[int, int]],
) -> dict[int, tuple[float, float, float]]:
    """Precompute distances from each segment endpoint to goal.

    For each track segment, computes:
    - Distance from node1 to goal (via shortest path)
    - Distance from node2 to goal (via shortest path)
    - Length of the segment edge

    This allows fast interpolation of distance to goal for any point
    on the segment without recomputing shortest paths.

    Parameters
    ----------
    goal_node_id : int
        Goal well node ID
    track_graph : nx.Graph
        Track graph
    track_segment_id_to_edge : dict
        Mapping from segment ID to edge tuple (node1, node2)

    Returns
    -------
    dict[int, tuple[float, float, float]]
        Mapping from segment_id to (dist_node1_to_goal, dist_node2_to_goal, edge_length)

    """
    segment_distances = {}

    for segment_id, (node1, node2) in track_segment_id_to_edge.items():
        # Compute shortest path distances from each endpoint to goal
        try:
            dist_node1_to_goal = nx.shortest_path_length(
                track_graph, source=node1, target=goal_node_id, weight="distance"
            )
        except nx.NetworkXNoPath:
            # If no path exists, use infinity
            dist_node1_to_goal = float("inf")

        try:
            dist_node2_to_goal = nx.shortest_path_length(
                track_graph, source=node2, target=goal_node_id, weight="distance"
            )
        except nx.NetworkXNoPath:
            dist_node2_to_goal = float("inf")

        # Get edge length
        edge_length = track_graph.edges[(node1, node2)]["distance"]

        segment_distances[segment_id] = (
            dist_node1_to_goal,
            dist_node2_to_goal,
            edge_length,
        )

    return segment_distances


def dist_to_goal(
    position_info: pd.DataFrame,
    trial: pd.Series,
    track_graph: nx.Graph,
    track_segment_id_to_edge: dict[int, tuple[int, int]],
) -> NDArray[np.float64]:
    """Compute distance to goal for all time points in a trial.

    Vectorized implementation that precomputes distances from segment endpoints
    to goal, then computes all timepoint distances using array operations.
    This avoids per-row Python loops and graph copies.

    Parameters
    ----------
    position_info : pd.DataFrame
        Position info for the trial
    trial : pd.Series
        Trial information with to_well attribute
    track_graph : nx.Graph
        Track graph
    track_segment_id_to_edge : dict
        Mapping from segment ID to edge tuple

    Returns
    -------
    np.ndarray
        Distance to goal for each time point

    """
    n = len(position_info)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Precompute distances from each segment endpoint to goal (once per trial)
    segment_distances = _precompute_segment_distances_to_goal(
        trial.to_well, track_graph, track_segment_id_to_edge
    )

    # Build lookup arrays indexed by segment_id for vectorized access
    segment_ids = position_info.track_segment_id.to_numpy()
    unique_segments = np.unique(segment_ids)

    # Precompute node1 positions, distances, and edge lengths per segment
    seg_node1_x = np.empty(n)
    seg_node1_y = np.empty(n)
    seg_dist_node1_to_goal = np.empty(n)
    seg_dist_node2_to_goal = np.empty(n)
    seg_edge_length = np.empty(n)

    for seg_id in unique_segments:
        mask = segment_ids == seg_id
        node1, _node2 = track_segment_id_to_edge[seg_id]
        node1_pos = track_graph.nodes[node1]["pos"]
        d1, d2, el = segment_distances[seg_id]

        seg_node1_x[mask] = node1_pos[0]
        seg_node1_y[mask] = node1_pos[1]
        seg_dist_node1_to_goal[mask] = d1
        seg_dist_node2_to_goal[mask] = d2
        seg_edge_length[mask] = el

    # Vectorized distance from each position to its segment's node1
    proj_x = position_info.projected_x_position.to_numpy()
    proj_y = position_info.projected_y_position.to_numpy()
    dx = proj_x - seg_node1_x
    dy = proj_y - seg_node1_y
    dist_from_node1 = np.sqrt(dx * dx + dy * dy)

    # Clamp to edge bounds for numerical stability
    np.clip(dist_from_node1, 0.0, seg_edge_length, out=dist_from_node1)

    # Two possible paths to goal — take the shorter one
    dist_via_node1 = dist_from_node1 + seg_dist_node1_to_goal
    dist_via_node2 = (seg_edge_length - dist_from_node1) + seg_dist_node2_to_goal

    return np.minimum(dist_via_node1, dist_via_node2)


def turn_direction(
    trial: pd.Series,
    position_info: pd.DataFrame,
    track_graph: nx.Graph,
    track_edges: np.ndarray,
) -> str:
    """Determine turn direction sequence for a trial.

    Parameters
    ----------
    trial : pd.Series
        Trial information
    position_info : pd.DataFrame
        Position info for the trial
    track_graph : nx.Graph
        Track graph
    track_edges : np.ndarray
        Array of track edges

    Returns
    -------
    str
        Turn direction sequence (e.g., "left-right-left")

    """
    trial_segment_ids = position_info.loc[
        trial.start_time : trial.end_time
    ].track_segment_id.unique()  # pandas unique is not sorted
    # Require at least 100 ms of occupancy per segment to filter transient noise.
    # At 500 Hz this is 50 samples.
    sampling_frequency = 1.0 / np.median(np.diff(position_info.index.to_numpy()))
    min_segment_samples = int(0.1 * sampling_frequency)
    is_bad = (
        position_info.loc[trial.start_time : trial.end_time]
        .groupby("track_segment_id")
        .size()
        < min_segment_samples
    )
    trial_segment_ids = trial_segment_ids[
        ~np.isin(trial_segment_ids, is_bad.index[is_bad])
    ]

    if len(trial_segment_ids) == 0:
        return ""

    # Ensure that the trial is in the correct traversal direction
    # track_segment_id does not contain node order information
    trial_nodes = track_edges[trial_segment_ids]

    if trial_nodes[0][0] != trial.from_well:
        trial_nodes = trial_nodes[::-1]

    for ind, (edge1, edge2) in enumerate(
        zip(trial_nodes[:-1], trial_nodes[1:], strict=True)
    ):
        if edge1[-1] != edge2[0]:
            trial_nodes[ind + 1] = trial_nodes[ind + 1][::-1]

    vectors = np.array(
        [
            np.array(track_graph.nodes[node2]["pos"])
            - np.array(track_graph.nodes[node1]["pos"])
            for (node1, node2) in trial_nodes
        ]
    )
    # 2D scalar cross product: positive means counterclockwise (left) turn.
    # Assumes track graph node "pos" coordinates use a standard right-hand
    # system (x increases rightward, y increases upward), as provided by
    # Spyglass position tracking.
    v1, v2 = vectors[:-1], vectors[1:]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    trial_turns = np.where(cross > 0, "left", "right")
    return "-".join(trial_turns)


def _process_single_trial(
    trial_data: tuple[
        dict, pd.DataFrame, nx.Graph, np.ndarray, dict, np.ndarray, np.ndarray, float
    ],
) -> dict:
    """Process a single trial in parallel.

    Parameters
    ----------
    trial_data : tuple
        Contains (trial_dict, position_info, track_graph, track_edges, track_segment_id_to_edge,
                  is_prev_reward_omission_row, is_patch_change_row, sampling_frequency)

    Returns
    -------
    dict
        Dictionary containing trial results with time indices

    """
    (
        trial_dict,
        position_info,
        track_graph,
        track_edges,
        track_segment_id_to_edge,
        is_prev_reward_omission_row,
        is_patch_change_row,
        sampling_frequency,
    ) = trial_data

    # Create a simple trial object with necessary attributes
    # Uses SimpleNamespace for clean dict-to-object conversion
    # Expected attributes: Index, start_time, end_time, goal_arrival_time, from_well, to_well
    trial = SimpleNamespace(**trial_dict)

    time = position_info.index.to_numpy()
    is_in_trial = (time >= trial.start_time) & (time < trial.end_time)
    from_start_time_to_goal = (time >= trial.start_time) & (
        time < trial.goal_arrival_time
    )
    from_goal_to_next_movement = (time >= trial.goal_arrival_time) & (
        time < trial.end_time
    )

    # Get trial position data
    trial_position = position_info.loc[is_in_trial]

    # Compute all variables for this trial
    trial_turn_direction = turn_direction(
        trial, trial_position, track_graph, track_edges
    )

    start_to_goal_position = position_info.loc[from_start_time_to_goal]
    trial_path_progress = path_progress_for_run_segment(
        start_to_goal_position, track_graph, trial.from_well, trial.to_well
    )

    trial_dist_to_goal = dist_to_goal(
        trial_position, trial, track_graph, track_segment_id_to_edge
    )

    trial_time_to_goal_start = (
        np.arange(np.sum(from_start_time_to_goal)) / sampling_frequency
    )

    return {
        "trial_index": trial.Index,
        "time_indices": {
            "is_in_trial": np.where(is_in_trial)[0],
            "from_start_time_to_goal": np.where(from_start_time_to_goal)[0],
            "from_goal_to_next_movement": np.where(from_goal_to_next_movement)[0],
        },
        "values": {
            "trial_number": trial.Index,
            "from_well": trial.from_well,
            "to_well": trial.to_well,
            "turn_direction": trial_turn_direction,
            "path_progress": trial_path_progress,
            "dist_to_goal": trial_dist_to_goal,
            "time_to_goal_start": trial_time_to_goal_start,
        },
        "omission_patch": {
            "is_prev_reward_omission": is_prev_reward_omission_row,
            "is_patch_change": is_patch_change_row,
        },
    }


def make_task_variables(
    position_info: pd.DataFrame,
    trials_df: pd.DataFrame,
    track_graph: nx.Graph,
    sampling_frequency: float = 500.0,
    omission_shifts: list[int] | None = None,
    patch_change_shifts: list[int] | None = None,
    parallel: bool = False,
) -> pd.DataFrame:
    """Compute all task-specific derived variables.

    Parameters
    ----------
    position_info : pd.DataFrame
        Position data
    trials_df : pd.DataFrame
        Trials data
    track_graph : nx.Graph
        Track graph
    sampling_frequency : float, optional
        Sampling frequency in Hz
    omission_shifts : list[int], optional
        Trial shifts for reward omission indicators
    patch_change_shifts : list[int], optional
        Trial shifts for patch change indicators
    parallel : bool, optional
        Whether to use parallel processing across trials. Default False.
        Uses ProcessPoolExecutor to process trials concurrently, which can
        significantly reduce computation time for sessions with many trials.

    Returns
    -------
    pd.DataFrame
        Task variables including:
        - trial_number
        - time_to_goal
        - start_node_id, goal_node_id
        - dist_to_goal
        - path_progress
        - turn_direction
        - is_no_reward_{N} (for each N in omission_shifts, e.g., is_no_reward_1)
        - is_patch_change{N} (for each N in patch_change_shifts, e.g., is_patch_change-1)

    """
    if omission_shifts is None:
        omission_shifts = list(DEFAULT_OMISSION_SHIFTS)
    if patch_change_shifts is None:
        patch_change_shifts = list(DEFAULT_PATCH_CHANGE_SHIFTS)

    time = position_info.index
    n_time = len(time)

    trial_number = np.full((n_time,), np.nan)
    time_to_goal = np.full((n_time), np.nan)
    start_node_id = np.full((n_time), np.nan)
    goal_node_id = np.full((n_time), np.nan)
    dist_to_goal_arr = np.full((n_time), np.nan)
    path_progress = np.full((n_time), np.nan)
    turn_direction_arr = np.full((n_time), pd.NA)
    n_shifts = len(omission_shifts)
    is_prev_reward_omission_pos = np.full((n_time, n_shifts), False)

    track_edges = np.array(track_graph.edges)

    track_segment_id_to_edge = {
        track_graph.edges[edge]["edge_id"]: edge for edge in track_graph.edges
    }
    is_prev_reward_omission = pd.concat(
        [
            (~trials_df.is_reward)
            .shift(shift)
            .infer_objects(copy=False)
            .fillna(False)
            .rename(f"is_no_reward_{shift}")
            for shift in omission_shifts
        ],
        axis=1,
    )
    is_patch_change = pd.concat(
        [
            trials_df.is_patch_change.shift(shift)
            .infer_objects(copy=False)
            .fillna(False)
            .rename(f"is_patch_change_{shift}")
            for shift in patch_change_shifts
        ],
        axis=1,
    )
    n_shifts = len(patch_change_shifts)
    is_patch_change_pos = np.full((n_time, n_shifts), False)

    if parallel:
        # Parallel processing across trials
        # Convert trials to dicts for pickling
        trial_data_list = []
        for trial in trials_df.itertuples():
            trial_dict = trial._asdict()
            trial_data_list.append(
                (
                    trial_dict,
                    position_info,
                    track_graph,
                    track_edges,
                    track_segment_id_to_edge,
                    is_prev_reward_omission.loc[trial.Index].to_numpy(),
                    is_patch_change.loc[trial.Index].to_numpy(),
                    sampling_frequency,
                )
            )

        with ProcessPoolExecutor() as executor:
            trial_results = list(executor.map(_process_single_trial, trial_data_list))

        # Combine results from all trials
        for result in trial_results:
            indices = result["time_indices"]
            values = result["values"]

            # Assign values to output arrays
            trial_number[indices["is_in_trial"]] = values["trial_number"]
            start_node_id[indices["is_in_trial"]] = values["from_well"]
            goal_node_id[indices["is_in_trial"]] = values["to_well"]
            turn_direction_arr[indices["is_in_trial"]] = values["turn_direction"]

            time_to_goal[indices["from_start_time_to_goal"]] = values[
                "time_to_goal_start"
            ]
            time_to_goal[indices["from_goal_to_next_movement"]] = 0.0

            path_progress[indices["from_start_time_to_goal"]] = values["path_progress"]
            path_progress[indices["from_goal_to_next_movement"]] = 0.0

            dist_to_goal_arr[indices["is_in_trial"]] = values["dist_to_goal"]

            is_prev_reward_omission_pos[indices["is_in_trial"]] = result[
                "omission_patch"
            ]["is_prev_reward_omission"]
            is_patch_change_pos[indices["is_in_trial"]] = result["omission_patch"][
                "is_patch_change"
            ]

    else:
        # Sequential processing (original implementation)
        for trial in trials_df.itertuples():
            is_in_trial = (time >= trial.start_time) & (time < trial.end_time)
            from_start_time_to_goal = (time >= trial.start_time) & (
                time < trial.goal_arrival_time
            )
            from_goal_to_next_movement = (time >= trial.goal_arrival_time) & (
                time < trial.end_time
            )
            trial_number[is_in_trial] = trial.Index

            time_to_goal[from_start_time_to_goal] = (
                np.arange(np.sum(from_start_time_to_goal)) / sampling_frequency
            )
            time_to_goal[from_goal_to_next_movement] = 0.0

            start_node_id[is_in_trial] = trial.from_well
            goal_node_id[is_in_trial] = trial.to_well
            turn_direction_arr[is_in_trial] = turn_direction(
                trial, position_info.loc[is_in_trial], track_graph, track_edges
            )
            path_progress[from_start_time_to_goal] = path_progress_for_run_segment(
                position_info.loc[from_start_time_to_goal],
                track_graph,
                trial.from_well,
                trial.to_well,
            )
            path_progress[from_goal_to_next_movement] = 0.0

            dist_to_goal_arr[is_in_trial] = dist_to_goal(
                position_info.loc[is_in_trial],
                trial,
                track_graph,
                track_segment_id_to_edge,
            )
            is_prev_reward_omission_pos[is_in_trial] = is_prev_reward_omission.loc[
                trial.Index
            ].to_numpy()
            is_patch_change_pos[is_in_trial] = is_patch_change.loc[
                trial.Index
            ].to_numpy()

    return pd.DataFrame(
        {
            "trial_number": trial_number,
            "time_to_goal": time_to_goal,
            "start_node_id": start_node_id,
            "goal_node_id": goal_node_id,
            "dist_to_goal": dist_to_goal_arr,
            "path_progress": path_progress,
            "turn_direction": turn_direction_arr,
            **{
                f"is_no_reward_{shift}": is_reward_omission
                for is_reward_omission, shift in zip(
                    is_prev_reward_omission_pos.T, omission_shifts, strict=True
                )
            },
            **{
                f"is_patch_change{shift}": is_patch_change
                for is_patch_change, shift in zip(
                    is_patch_change_pos.T, patch_change_shifts, strict=True
                )
            },
        },
        index=time,
    ).ffill()

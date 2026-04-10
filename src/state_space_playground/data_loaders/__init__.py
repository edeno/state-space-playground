"""Data loading module for the spatial bandit task.

This module provides functions to load neural data, position data, task events,
and compute derived task variables for the continuum replay analysis.
"""

# Main data loading functions
from .bandit_task import load_data

# Constants
from .constants import (
    NWB_FILES,
    TRACK_SEGMENT_TO_EDGE_ID,
    TRACK_SEGMENT_TO_PATCH,
    WELL_PATCH_MAPPING,
)
from .utils import get_epoch_info, get_labels, get_training_timepoints

__all__ = [
    # Main functions
    "load_data",
    "get_epoch_info",
    "get_training_timepoints",
    "get_labels",
    # Constants
    "NWB_FILES",
    "WELL_PATCH_MAPPING",
    "TRACK_SEGMENT_TO_PATCH",
    "TRACK_SEGMENT_TO_EDGE_ID",
]

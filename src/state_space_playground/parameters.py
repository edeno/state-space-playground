"""Global parameters and configuration for neural decoding analysis.

This module provides centralized configuration for the continuum-swr-replay
project, including spatial binning parameters, decoding model parameters,
file paths, and analysis constants.

Constants
---------
SAMPLING_FREQUENCY : int
    Decoding sampling frequency (500 Hz). All temporal analyses use this rate.
MAX_SPEED_THRESHOLD_CM_PER_SEC : float
    Maximum speed threshold (4.0 cm/s) for filtering non-local events. Events
    with speeds above this are likely not genuine replay events.
ROOT_DIR : Path
    Root directory of the project (parent of src/).
PROCESSED_DATA_DIR : Path
    Directory for saving analysis results and processed data.
FIGURE_DIR : Path
    Directory for saving generated figures.
PLACE_BIN_SIZE_1D : float
    Spatial bin size for 1D decoding (2.0 cm).
PLACE_BIN_SIZE_2D : float
    Spatial bin size for 2D decoding (3.5 cm).
POSITION_STD : float
    Spatial uncertainty standard deviation (~3.536 cm, derived from sqrt(12.5)).
WAVEFORM_STD : float
    Waveform feature space standard deviation (24.0 µV) for clusterless decoding.
BLOCK_SIZE : int
    GPU processing block size (1024) for memory management.
DETECTOR_CONFIG : dict
    Maps brain area names to detector class names and types.
DECODING_PARAMS : DecodingParameters
    Singleton instance with default decoding parameters.
PROBABILITY_THRESHOLDS : list
    Probability thresholds for posterior analysis [0.50, 0.80, 0.95].
COLORS : dict
    Color scheme for plotting different event types.

Classes
-------
DecodingParameters
    Dataclass containing validated decoding parameters.
TrainingType
    Enum for training data selection strategies (ALL or NO_RIPPLE).

Notes
-----
All constants use type hints with Final[] to indicate they should not be
modified. The DecodingParameters dataclass is frozen for immutability.

See Also
--------
continuum_swr_replay.types : TypedDict definitions for data structures
continuum_swr_replay.data_loaders.constants : Task-specific constants

Examples
--------
>>> from continuum_swr_replay.parameters import DECODING_PARAMS, TrainingType
>>> DECODING_PARAMS.place_bin_size
2.0
>>> TrainingType.NO_RIPPLE.value
'no_ripple'

"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final

import numpy as np

# Decoding sampling frequency in samples per second
SAMPLING_FREQUENCY = 500

# Maximum speed threshold for filtering non-local events (cm/s)
# Events with speeds above this are likely not genuine replay events
MAX_SPEED_THRESHOLD_CM_PER_SEC = 4.0

# Data directories and definitions
# ROOT_DIR is the project root (continuum-swr-replay/), not src/
ROOT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = ROOT_DIR / "Processed-Data"
FIGURE_DIR = ROOT_DIR / "paper" / "figures"

# Multiunit High Synchrony Event (HSE) detection parameters
HSE_MINIMUM_DURATION_SEC: Final[float] = 0.015  # 15 ms minimum event duration
HSE_ZSCORE_THRESHOLD: Final[float] = 2.0  # Z-score threshold for event detection
HSE_CLOSE_EVENT_THRESHOLD_SEC: Final[float] = 0.0  # Gap to merge adjacent events (0 = no merging)

# Spatial binning parameters (in cm)
PLACE_BIN_SIZE_1D: Final[float] = 2.0
PLACE_BIN_SIZE_2D: Final[float] = 3.5

# Decoding model parameters
POSITION_STD: Final[float] = float(np.sqrt(12.5))  # ~3.536 cm spatial uncertainty
WAVEFORM_STD: Final[float] = 24.0  # Waveform feature space standard deviation (µV)
BLOCK_SIZE: Final[int] = 2**10  # GPU processing block size (1024)

# Detector configuration by brain area
# Maps brain area to (detector_class_name, detector_type_string)
# Actual detector classes must be imported from non_local_detector in analysis scripts
DETECTOR_CONFIG: Final[dict[str, tuple[str, str]]] = {
    "HPC": ("NonLocalClusterlessDetector", "clusterless"),
    "mPFC": ("NonLocalSortedSpikesDetector", "sorted_spikes"),
    "OFC": ("NonLocalSortedSpikesDetector", "sorted_spikes"),
}


@dataclass(frozen=True)
class DecodingParameters:
    """Parameters for neural decoding analysis.

    Attributes
    ----------
    place_bin_size : float
        Spatial bin size in cm for position discretization. Smaller bins
        provide higher spatial resolution but require more computation.
        Default is 2.0 cm for 1D decoding. Must be positive.
    waveform_std : float
        Standard deviation for waveform feature likelihood in µV. Controls
        how tightly spikes must match the waveform features to contribute
        to the posterior. Default is 24.0 µV. Must be positive.
    position_std : float
        Standard deviation for position likelihood in cm. Derived from
        sqrt(12.5) ≈ 3.536 cm, which corresponds to uniform spatial uncertainty.
        Must be positive.
    block_size : int
        Block size for GPU memory management during decoding. Controls
        how many time bins are processed simultaneously. Must be a power
        of 2 for optimal GPU performance. Default is 1024 (2^10).

    Notes
    -----
    These parameters control the non-local detector's decoding algorithm.
    The geometric mean of state duration is 1 / (1 - p), where:
    - p = 1 - (1 / n_time_steps)
    - n_time_steps should equal half a theta cycle
    - Theta cycles are ~8 Hz or 125 ms per cycle
    - Half a theta cycle is 62.5 ms
    - If timestep is 2 ms, then n_time_steps = 31.25
    - So p = 0.968

    The dataclass is frozen (immutable) to prevent accidental modification
    of these critical analysis parameters.

    Raises
    ------
    ValueError
        If block_size is not a power of 2, or if any standard deviation
        or bin size parameter is not positive.

    """

    place_bin_size: float = PLACE_BIN_SIZE_1D
    waveform_std: float = WAVEFORM_STD
    position_std: float = POSITION_STD
    block_size: int = BLOCK_SIZE

    def __post_init__(self) -> None:
        """Validate parameter values after initialization.

        Raises
        ------
        ValueError
            If block_size is not a power of 2, or if any parameter
            that should be positive is not.

        """
        # Validate block_size is a power of 2
        # Powers of 2 have exactly one bit set, so n & (n-1) == 0
        if self.block_size <= 0 or (self.block_size & (self.block_size - 1)) != 0:
            raise ValueError(f"block_size must be power of 2, got {self.block_size}")

        # Validate positive values
        if self.position_std <= 0:
            raise ValueError(f"position_std must be positive, got {self.position_std}")

        if self.waveform_std <= 0:
            raise ValueError(f"waveform_std must be positive, got {self.waveform_std}")

        if self.place_bin_size <= 0:
            raise ValueError(
                f"place_bin_size must be positive, got {self.place_bin_size}"
            )


# Singleton instance of decoding parameters
DECODING_PARAMS: Final = DecodingParameters()

PROBABILITY_THRESHOLDS: Final[list[float]] = [0.50, 0.80, 0.95]


class TrainingType(str, Enum):
    """Types of training data selection for decoder.

    Attributes
    ----------
    ALL : str
        Use all time points for training (no exclusions)
    NO_RIPPLE : str
        Exclude ripple times from training data

    Notes
    -----
    Inherits from str to allow direct string comparisons and JSON serialization.
    Use .value to get the underlying string value.

    Examples
    --------
    >>> TrainingType.ALL
    <TrainingType.ALL: 'all'>
    >>> TrainingType.ALL.value
    'all'
    >>> TrainingType.ALL == "all"
    True

    """

    ALL = "all"
    NO_RIPPLE = "no_ripple"

    def __str__(self) -> str:
        """Return string value for easy formatting."""
        return self.value


COLORS = {
    "SWR": "#d95f02",
    "MUA": "#7570b3",
    "Non-Local": "#1b9e77",
    "Local": "#add8e6",  # Soft Blue for Local
    "No-Spike": "#b0b0b0",  # Light Gray for No-Spike
    "All Events": "#d9d9d9",  # Light Gray for background/all-events reference
}

__all__ = [
    # Sampling and thresholds
    "SAMPLING_FREQUENCY",
    "MAX_SPEED_THRESHOLD_CM_PER_SEC",
    "PROBABILITY_THRESHOLDS",
    # HSE detection parameters
    "HSE_MINIMUM_DURATION_SEC",
    "HSE_ZSCORE_THRESHOLD",
    "HSE_CLOSE_EVENT_THRESHOLD_SEC",
    # Directories
    "ROOT_DIR",
    "PROCESSED_DATA_DIR",
    "FIGURE_DIR",
    # Spatial binning parameters
    "PLACE_BIN_SIZE_1D",
    "PLACE_BIN_SIZE_2D",
    # Decoding model parameters
    "POSITION_STD",
    "WAVEFORM_STD",
    "BLOCK_SIZE",
    "DETECTOR_CONFIG",
    # Decoding parameters (dataclass)
    "DecodingParameters",
    "DECODING_PARAMS",
    # Enums
    "TrainingType",
    # Color schemes
    "COLORS",
]

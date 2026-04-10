"""DIO event data loading.

This module loads digital I/O (DIO) event data from Spyglass/NWB files. DIO events
capture hardware-triggered signals from the behavioral apparatus, including beam
breaks (well visits), reward pump activations, and environmental lights.

These events are essential for constructing the task structure: identifying when the
animal visits wells, receives rewards, and transitions between behavioral states.

Main Functions
--------------
load_dios
    Primary entry point that loads all DIO event types (beam breaks, pump events,
    light events) for a given session and time range. Returns organized event data
    as a TypedDict.

Helper Functions
----------------
convert_dio_events_to_start_stop_times
    Converts raw DIO binary signals to discrete events with start/end times.
    Handles edge cases where events occur at data boundaries.

Event Types
-----------
Beam Breaks (Poke events):
    Triggered when animal's nose breaks infrared beams at well locations. Used to
    identify well visits and construct trial structure. Event names match pattern
    "Poke%".

Pump Events:
    Triggered when reward pump activates to deliver liquid reward. Used to identify
    rewarded trials. Event names match pattern "Pump%".

Light Events:
    Environmental lighting changes during the task. Event names match pattern
    "Light%".

Data Structure
--------------
All events are returned as pandas DataFrames with:
    - start_time: Event onset time (seconds)
    - end_time: Event offset time (seconds)
    - event_name: String identifier for the event type
    - event_number: Sequential numbering within each event type
    - Multi-index: (event_name, event_number)

Notes
-----
- All times are in seconds
- Events are filtered to valid_times range (typically epoch boundaries)
- DIO data is binary (0/1) and converted to discrete intervals
- Queries use Spyglass DIOEvents table with SQL-like LIKE pattern matching
- Edge case: if event starts at last time point, end_time is clamped to last index

See Also
--------
continuum_swr_replay.data_loaders.trials : Trial construction from DIO events
continuum_swr_replay.types : DIOEventData TypedDict

Examples
--------
>>> dio_data = load_dios("chimi20200212_.nwb", (100.0, 1000.0))
>>> beam_breaks = dio_data["beam_breaks"]
>>> pump_events = dio_data["pump_events"]

"""

import warnings

import numpy as np
import pandas as pd
from spyglass.common import DIOEvents

from ..types import DIOEventData

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pynwb")
warnings.filterwarnings("ignore", category=UserWarning, module="hdmf")
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint")


def convert_dio_events_to_start_stop_times(
    event: pd.DataFrame, event_name: str
) -> pd.DataFrame:
    """Convert DIO events to start/stop time format.

    Parameters
    ----------
    event : pd.DataFrame
        Event data with time index
    event_name : str
        Name of the event

    Returns
    -------
    pd.DataFrame
        DataFrame with start_time, end_time, event_name, and event_number columns

    """
    start_indices = event.astype(bool).to_numpy().nonzero()[0]
    end_indices = start_indices + 1
    end_indices[end_indices >= len(event)] = len(event) - 1

    event_times = pd.DataFrame(
        {
            "start_time": event.iloc[start_indices].index,
            "end_time": event.iloc[end_indices].index,
            "event_name": event_name,
            "event_number": np.arange(len(start_indices)) + 1,
        },
    )
    return event_times.set_index(["event_name", "event_number"])


def _load_dio_event_type(
    nwb_file_name: str, pattern: str, valid_times: tuple[float, float]
) -> pd.DataFrame:
    """Load and process a specific type of DIO event.

    Helper function to eliminate code duplication when loading different
    DIO event types (beam breaks, pump events, light events).

    Parameters
    ----------
    nwb_file_name : str
        NWB file name
    pattern : str
        SQL LIKE pattern for event names (e.g., "Poke%", "Pump%", "Light%")
    valid_times : tuple[float, float]
        (start_time, end_time) tuple defining valid time range in seconds

    Returns
    -------
    pd.DataFrame
        Processed DIO events with start/end times and event numbers

    See Also
    --------
    load_dios : Main function that uses this helper for all event types
    convert_dio_events_to_start_stop_times : Converts raw events to intervals

    """
    events = {
        nwb_file["dio_event_name"]: pd.DataFrame(nwb_file["dio"].data)
        .set_index(pd.Index(nwb_file["dio"].timestamps, name="time"))
        .loc[valid_times[0] : valid_times[1]]  # type: ignore[misc]
        for nwb_file in (
            DIOEvents
            & {"nwb_file_name": nwb_file_name}
            & f'dio_event_name LIKE "{pattern}"'
        ).fetch_nwb()
    }
    return pd.concat(
        [
            convert_dio_events_to_start_stop_times(event, event_name)
            for event_name, event in events.items()
        ],
        axis=0,
    )


def load_dios(nwb_file_name: str, valid_times: tuple[float, float]) -> DIOEventData:
    """Load DIO events (beam breaks, pump events, lights).

    Parameters
    ----------
    nwb_file_name : str
        NWB file name
    valid_times : tuple[float, float]
        (start_time, end_time) tuple defining valid time range in seconds

    Returns
    -------
    DIOEventData
        TypedDict containing:
        - beam_breaks: pd.DataFrame with poke events
        - pump_events: pd.DataFrame with reward pump events
        - light_events: pd.DataFrame with light events

    """
    beam_breaks = _load_dio_event_type(nwb_file_name, "Poke%", valid_times)
    pump_events = _load_dio_event_type(nwb_file_name, "Pump%", valid_times)
    light_events = _load_dio_event_type(nwb_file_name, "Light%", valid_times)

    return {
        "beam_breaks": beam_breaks,
        "pump_events": pump_events,
        "light_events": light_events,
    }

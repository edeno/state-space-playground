"""Constants for the bandit task data loading.

This module contains all constant values used across the data loading pipeline,
including NWB file lists, spatial mappings, and task structure definitions.

Attributes
----------
NWB_FILES : list of str
    Complete list of NWB file names for all animals and sessions in the dataset.
    Includes data from 5 animals: chimi, j16, peanut, senor, and wilbur.
WELL_PATCH_MAPPING : dict of {int: int}
    Maps well node IDs (0-5) to patch IDs (1-3).
    Wells 0-1 are in patch 1, wells 2-3 in patch 2, wells 4-5 in patch 3.
TRACK_SEGMENT_TO_PATCH : dict of {int: int}
    Maps track segment IDs to patch IDs (1-3).
TRACK_SEGMENT_TO_EDGE_ID : dict of {int: int}
    Maps track segment indices to edge IDs in the track graph.

Notes
-----
The spatial bandit task uses a 6-well environment organized into 3 patches.
Each patch contains 2 wells, and the animal navigates between wells along
track segments (edges in the track graph).

See Also
--------
get_epoch_info : Get information about all available epochs
load_data : Main data loading function that uses these constants

Examples
--------
>>> from continuum_swr_replay.data_loaders.constants import WELL_PATCH_MAPPING
>>> WELL_PATCH_MAPPING[0]  # Well 0 is in patch 1
1
>>> WELL_PATCH_MAPPING[4]  # Well 4 is in patch 3
3

"""

NWB_FILES = [
    "chimi20200212_.nwb",
    "chimi20200213_.nwb",
    "chimi20200214_.nwb",
    "chimi20200216_.nwb",
    "chimi20200217_.nwb",
    "chimi20200218_.nwb",
    "chimi20200220_.nwb",
    "chimi20200221_.nwb",
    "chimi20200222_.nwb",
    "chimi20200223_.nwb",
    "chimi20200224_.nwb",
    "chimi20200225_.nwb",
    "chimi20200227_.nwb",
    "chimi20200228_.nwb",
    "chimi20200304_.nwb",
    "chimi20200305_.nwb",
    "chimi20200306_.nwb",
    "chimi20200308_.nwb",
    "chimi20200310_.nwb",
    "chimi20200311_.nwb",
    "chimi20200312_.nwb",
    "chimi20200313_.nwb",
    "j1620210706_.nwb",
    "j1620210707_.nwb",
    "j1620210708_.nwb",
    "j1620210709_.nwb",
    "j1620210710_.nwb",
    "j1620210711_.nwb",
    "j1620210712_.nwb",
    "j1620210713_.nwb",
    "j1620210714_.nwb",
    "j1620210715_.nwb",
    "j1620210716_.nwb",
    "j1620210717_.nwb",
    "j1620210718_.nwb",
    "j1620210719_.nwb",
    "j1620210720_.nwb",
    "j1620210721_.nwb",
    "peanut20201125_.nwb",
    "peanut20201127_.nwb",
    "peanut20201128_.nwb",
    "peanut20201129_.nwb",
    "peanut20201201_.nwb",
    "peanut20201202_.nwb",
    "peanut20201203_.nwb",
    "peanut20201204_.nwb",
    "peanut20201205_.nwb",
    "peanut20201206_.nwb",
    "peanut20201208_.nwb",
    "peanut20201209_.nwb",
    "senor20201027_.nwb",
    "senor20201028_.nwb",
    "senor20201030_.nwb",
    "senor20201101_.nwb",
    "senor20201102_.nwb",
    "senor20201103_.nwb",
    "senor20201104_.nwb",
    "senor20201105_.nwb",
    "senor20201106_.nwb",
    "senor20201107_.nwb",
    "senor20201109_.nwb",
    "senor20201110_.nwb",
    "senor20201111_.nwb",
    "senor20201112_.nwb",
    "senor20201113_.nwb",
    "senor20201114_.nwb",
    "senor20201115_.nwb",
    "senor20201116_.nwb",
    "senor20201117_.nwb",
    "senor20201118_.nwb",
    "senor20201120_.nwb",
    "senor20201121_.nwb",
    "wilbur20210326_.nwb",
    "wilbur20210328_.nwb",
    "wilbur20210329_.nwb",
    "wilbur20210330_.nwb",
    "wilbur20210331_.nwb",
    "wilbur20210402_.nwb",
    "wilbur20210403_.nwb",
    "wilbur20210404_.nwb",
    "wilbur20210405_.nwb",
    "wilbur20210406_.nwb",
    "wilbur20210407_.nwb",
    "wilbur20210408_.nwb",
    "wilbur20210409_.nwb",
    "wilbur20210410_.nwb",
    "wilbur20210411_.nwb",
    "wilbur20210412_.nwb",
    "wilbur20210413_.nwb",
    "wilbur20210414_.nwb",
    "wilbur20210415_.nwb",
]
"""Complete list of 91 NWB files from 5 animals performing the spatial bandit task.

Animals and session counts:
- chimi: 22 sessions (Feb-Mar 2020)
- j16: 16 sessions (Jul 2021)
- peanut: 12 sessions (Nov-Dec 2020)
- senor: 22 sessions (Oct-Nov 2020)
- wilbur: 19 sessions (Mar-Apr 2021)

Excluded sessions (no noPrePostTrialTimes intervals due to trial parsing issues):
chimi20200215, chimi20200219, chimi20200226, chimi20200301, chimi20200302,
chimi20200303, chimi20200307, peanut20201124, peanut20201126, peanut20201130,
peanut20201207, senor20201029, senor20201031, senor20201108, senor20201119,
wilbur20210327, wilbur20210401, wilbur20210416

Epochs with missing Spyglass pipeline entries (fail during HPC decoding):
- senor20201104_06_r3: missing LFPBandV1 entry (no ripple detection data)
- senor20201104_08_r4: missing LFPBandV1 entry (no ripple detection data)
- senor20201105_09_r4: missing TrackGraph entry (no linearized position data)
"""

WELL_PATCH_MAPPING = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 3,
    5: 3,
}
"""Mapping from well node IDs to patch IDs.

The 6-well environment is organized into 3 patches:
- Patch 1: Wells 0, 1
- Patch 2: Wells 2, 3
- Patch 3: Wells 4, 5
"""

TRACK_SEGMENT_TO_PATCH = {
    0: 1,
    1: 1,
    6: 1,
    2: 2,
    3: 2,
    7: 2,
    4: 3,
    5: 3,
    8: 3,
}
"""Mapping from track segment IDs to patch IDs.

Track segments are edges in the graph connecting wells. Each segment
is associated with one of the 3 patches based on its location.
Segments 0, 1, 6 are in patch 1; segments 2, 3, 7 in patch 2;
segments 4, 5, 8 in patch 3.
"""

TRACK_SEGMENT_TO_EDGE_ID = {
    0: 1,
    1: 2,
    2: 4,
    3: 5,
    4: 7,
    5: 8,
    6: 0,
    7: 3,
    8: 6,
}
"""Mapping from track segment indices to NetworkX graph edge IDs.

This mapping translates between sequential segment indices (0-8) and
the actual edge IDs used in the NetworkX track graph representation.
Required for correctly indexing into graph edge attributes and positions.
"""

"""Prime the session cache for j1620210710_.nwb / 02_r1 (sorted HPC).

Run once to avoid the ~2 minute reload cost on every notebook restart:

    uv run python scripts/save_session.py

On cache hit the script is a quick sanity check; on miss it calls
``load_data`` and pickles the result to ``cache/sessions/``.
"""

from __future__ import annotations

import sys

from state_space_playground.session import load_session


def main() -> int:
    data = load_session(
        nwb_file_name="j1620210710_.nwb",
        epoch_name="02_r1",
        use_sorted_hpc=True,
        force=False,
    )
    keys = sorted(data.keys())
    print(f"Loaded session with {len(keys)} keys.")

    # Spot-check the most important entries.
    print()
    print("=== Spike data ===")
    spike_times = data["spike_times"]
    for area in sorted(spike_times.keys()):
        units = spike_times[area]
        n_spikes = sum(len(u) for u in units)
        print(f"  {area}: {len(units)} units, {n_spikes:,} spikes")

    print()
    print("=== Position ===")
    pi = data["position_info"]
    dt_min = (pi.index.max() - pi.index.min()) / 60.0
    print(f"  position_info: {pi.shape[0]:,} rows, {dt_min:.1f} min span")

    print()
    print("=== Task ===")
    print(f"  trials:      {len(data['trials'])}")
    print(f"  ripple_times: {len(data['ripple_times'])}")
    print(f"  pump_events:  {len(data['pump_events'])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

import json
from typing import Dict, List, Optional


def parse_msprof_json(trace_path: str) -> List[Dict]:
    """Parse msprof Chrome trace JSON and extract all ``KERNEL_*`` events.

    msprof outputs a Chrome-format trace JSON where device-side kernel
    executions are recorded as events with names beginning with ``KERNEL_``.
    This function reads the file, filters those events, and returns them
    sorted by timestamp.

    Parameters
    ----------
    trace_path : str
        Path to the msprof Chrome trace JSON file.

    Returns
    -------
    List[Dict]
        A chronologically-sorted list of ``KERNEL_*`` event dicts.
        Each dict contains at least:
          - ``name`` (str): event name, e.g. ``KERNEL_RMSNorm``.
          - ``ts`` (float): start timestamp in microseconds.
          - ``dur`` (float): duration in microseconds.
          - ``ph`` (str): event phase (``"X"`` for complete events).
          - ``pid`` / ``tid`` (int): process / thread identifiers.
        Returns an empty list if no kernel events are found or the file
        cannot be read.

    Raises
    ------
    FileNotFoundError
        If *trace_path* does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    with open(trace_path, "r") as f:
        data = json.load(f)

    events: List[Dict] = data.get("traceEvents", data if isinstance(data, list) else [])

    kernel_events = [ev for ev in events if isinstance(ev, dict) and ev.get("name", "").startswith("KERNEL_")]

    kernel_events.sort(key=lambda e: e.get("ts", 0.0))

    return kernel_events


def aggregate_device_kernels(events: List[Dict], logical_call_marker: str = "") -> Dict:
    """Aggregate all device kernel durations that belong to one logical call.

    A single logical operator invocation (e.g. one call to an Ascend C
    kernel via ``<<<>>>`` or one PyPTO trace) may produce multiple device
    kernel events.  This function aggregates them.

    Parameters
    ----------
    events : List[Dict]
        A list of ``KERNEL_*`` event dicts, typically from
        :func:`parse_msprof_json`.  If *logical_call_marker* is supplied,
        only events whose ``name`` contains that substring are considered;
        otherwise all events are aggregated.
    logical_call_marker : str
        Optional substring filter on event ``name``.  For example, pass
        ``"rms_norm"`` to aggregate only kernels whose name contains
        ``"rms_norm"``.

    Returns
    -------
    Dict
        - all_device_kernels_us (float): sum of *dur* across all matching
          kernel events.
        - primary_compute_kernel_us (float): *dur* of the longest single
          matching kernel event.
        - kernel_count (int): number of matching kernel events.
        - kernel_names (List[str]): sorted list of unique kernel names.
    """
    filtered = events
    if logical_call_marker:
        filtered = [ev for ev in events if logical_call_marker in ev.get("name", "")]

    if not filtered:
        return {
            "all_device_kernels_us": 0.0,
            "primary_compute_kernel_us": 0.0,
            "kernel_count": 0,
            "kernel_names": [],
        }

    total_us = sum(ev.get("dur", 0.0) for ev in filtered)
    primary = max(filtered, key=lambda e: e.get("dur", 0.0))
    names = sorted({ev.get("name", "") for ev in filtered})

    return {
        "all_device_kernels_us": total_us,
        "primary_compute_kernel_us": primary.get("dur", 0.0),
        "kernel_count": len(filtered),
        "kernel_names": names,
    }

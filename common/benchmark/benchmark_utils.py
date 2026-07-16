import math
import numpy as np
from typing import Dict, List, Optional


def compute_statistics(latencies_us: List[float]) -> Dict:
    """Compute descriptive statistics from a list of raw latency samples.

    Parameters
    ----------
    latencies_us : List[float]
        One or more latency measurements in microseconds.

    Returns
    -------
    Dict
        - count (int): number of valid samples.
        - median_us (float): median latency.
        - mean_us (float): mean latency.
        - min_us (float): minimum latency.
        - max_us (float): maximum latency.
        - p90_us (float): 90th percentile latency.
        - std_us (float): population standard deviation.
        - cv (float): coefficient of variation (std / mean).

    Raises
    ------
    ValueError
        If *latencies_us* is empty.
    """
    if not latencies_us:
        raise ValueError("latencies_us must contain at least one sample")

    arr = np.array(latencies_us, dtype=np.float64)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    cv = std / mean if mean != 0.0 else 0.0

    sorted_arr = np.sort(arr)

    def percentile(p: float) -> float:
        idx = int(math.ceil(p * n)) - 1
        idx = max(0, min(idx, n - 1))
        return float(sorted_arr[idx])

    return {
        "count": n,
        "median_us": float(np.median(arr)),
        "mean_us": mean,
        "min_us": float(arr.min()),
        "max_us": float(arr.max()),
        "p90_us": percentile(0.90),
        "std_us": std,
        "cv": cv,
    }


def compute_effective_bandwidth(
    input_bytes: int, output_bytes: int, latency_us: float
) -> float:
    """Compute effective memory bandwidth in GB/s.

    Effective bandwidth is defined as total bytes transferred
    (input + output) divided by latency.

    Parameters
    ----------
    input_bytes : int
        Total bytes read from device memory (inputs).
    output_bytes : int
        Total bytes written to device memory (outputs).
    latency_us : float
        Measured kernel latency in microseconds.

    Returns
    -------
    float
        Effective bandwidth in gigabytes per second (GB/s).
        Returns 0.0 if *latency_us* is zero or negative.
    """
    if latency_us <= 0.0:
        return 0.0
    total_bytes = input_bytes + output_bytes
    if total_bytes <= 0:
        return 0.0
    seconds = latency_us * 1e-6
    return total_bytes / seconds / (1024 ** 3)

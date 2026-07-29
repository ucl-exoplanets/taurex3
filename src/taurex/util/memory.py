"""Memory management utilities.

Provides helpers for releasing freed memory back to the operating system.
Python and numpy use internal memory pools that hold onto freed pages,
which causes RSS to grow even when actual allocations are stable.
"""

import ctypes
import gc
import sys


def trim_memory() -> int:
    """Release freed memory back to the OS.

    Python's pymalloc and numpy's internal allocator hold onto
    freed memory pages for reuse, which can cause RSS to grow
    over time even when actual allocations are stable.

    This function:
    1. Runs gc.collect() to free any unreachable objects
    2. On Linux, calls malloc_trim(0) to release free heap pages
       back to the OS, which reduces RSS.

    Returns
    -------
    int
        Number of objects collected by gc.collect().
    """
    collected = gc.collect()

    # On Linux, trim the malloc heap to return freed pages to the OS.
    # This is critical for long-running retrieval jobs where
    # numpy temporarily allocates large arrays (sigma_xsec, etc.)
    # in the log-likelihood hot path.
    if sys.platform == "linux":
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.malloc_trim(0)
        except (OSError, AttributeError):
            pass

    return collected


def memory_usage_mb() -> float:
    """Return current RSS in MB (Linux only, else 0)."""
    if sys.platform != "linux":
        return 0.0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # Format: "VmRSS:    12345 kB"
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return 0.0
    return 0.0

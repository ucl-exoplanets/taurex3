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
    if sys.platform == "linux":
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.malloc_trim(0)
        except (OSError, AttributeError):
            pass

    return collected

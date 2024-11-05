"""Module for wrapping MPI functions.

Most functions will do nothing if mpi4py is not present and will simply
replicate expected behaviour. This allows for MPI and non-MPI code to be
run without any changes (In theory...).
"""
import enum
import logging
import typing as t
from functools import lru_cache, wraps

import numpy as np

from .types import AnyValType

T = t.TypeVar("T", bound=AnyValType)


class Ops(str, enum.Enum):
    """Available MPI operations"""

    SUM = "sum"
    PROD = "prod"
    MAX = "max"
    MIN = "min"


def has_mpi() -> bool:
    """Checks if mpi4py is installed

    Returns
    -------
    bool:
        True if mpi4py is installed, False otherwise

    """
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        return False
    return True


def convert_op(operation: Ops) -> t.Any:
    """Converts string to MPI operation."""
    from mpi4py import MPI

    return getattr(MPI, str(operation.upper()))


@lru_cache(maxsize=10)
def shared_comm() -> t.Any:
    """Return shared memory communicator.

    Returns the process id within a node. Used for shared memory.
    """
    from mpi4py import MPI

    return MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)


@lru_cache(maxsize=10)
def nprocs() -> int:
    """Gets number of processes or returns 1 if mpi is not installed

    Returns
    -------
    int:
        Rank of process or 1 if MPI is not installed

    """
    try:
        from mpi4py import MPI
    except ImportError:
        return 1

    comm = MPI.COMM_WORLD

    return comm.Get_size()


def allgather(value: T) -> t.List[T]:
    """Gathers all values from all processes or returns value if mpi is not installed

    Parameters
    ----------
    value:
        Value to gather

    Returns
    -------
    list:
        List of gathered values or list containing value if mpi is not installed

    """
    try:
        from mpi4py import MPI
    except ImportError:
        return [value]

    comm = MPI.COMM_WORLD
    data = value
    data = comm.allgather(data)

    return data


def allreduce(value: T, op: Ops) -> T:
    """Reduces all values from all processes or returns value if mpi is not installed

    Parameters
    ----------
    value:
        Value to reduce

    op: :class:`~taurex.mpi.Ops`
        Operation to perform

    Returns
    -------
    result:
        Reduced value or value if mpi is not installed

    """

    try:
        from mpi4py import MPI
    except ImportError:
        return value

    comm = MPI.COMM_WORLD
    data = value
    data = comm.allreduce(value, op=convert_op(op))

    return data


def broadcast(array: T, rank: t.Optional[int] = 0) -> T:
    """Broadcasts array from rank or returns array if mpi is not installed

    Parameters
    ----------
    array:
        Array to broadcast

    rank: int, optional
        Rank to broadcast from, default is 0

    Returns
    -------
    array:
        Broadcasted array or array if mpi is not installed

    """
    import numpy as np

    try:
        from mpi4py import MPI
    except ImportError:
        return array
    comm = MPI.COMM_WORLD
    if isinstance(array, np.ndarray):
        data = None
        if get_rank() == rank:
            data = np.copy(array)
        else:
            data = np.zeros_like(array)
        comm.Bcast(data, root=rank)
    else:
        data = comm.bcast(array, root=rank)

    return data


@lru_cache(maxsize=10)
def get_rank(comm: t.Any = None) -> int:
    """Gets rank or returns 0 if mpi is not installed

    Parameters
    ----------

    comm: int, optional
        MPI communicator, default is MPI_COMM_WORLD


    Returns
    -------
    int:
        Rank of process in communitor or 0 if MPI is not installed

    """
    try:
        from mpi4py import MPI
    except ImportError:
        return 0

    comm = comm or MPI.COMM_WORLD
    rank = comm.Get_rank()

    return rank


def barrier(comm: t.Any = None) -> None:
    """Waits for all processes to finish.

    Does nothing if mpi4py not present

    Parameters
    ----------

    comm: int, optional
        MPI communicator, default is MPI_COMM_WORLD

    """

    try:
        from mpi4py import MPI
    except ImportError:
        return

    comm = comm or MPI.COMM_WORLD
    comm.Barrier()


def only_master_rank(f) -> t.Callable:
    """A decorator to ensure only the master MPI rank can run it."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        if get_rank() == 0:
            return f(*args, **kwargs)

    return wrapper


@lru_cache(maxsize=10)
def shared_rank() -> int:
    """Gets rank within shared memory communicator. (MPI only)"""
    return shared_comm().Get_rank()


def allocate_as_shared(
    arr: np.ndarray,
    logger: t.Optional[logging.Logger] = None,
    force_shared: t.Optional[bool] = False,
):
    """Converts a numpy array into an MPI shared memory.

    This allow for things like opacities to be loaded only
    once per node when using MPI. Only activates if mpi4py
    installed and when enabled via the ``mpi_use_shared`` input::

        [Global]
        mpi_use_shared = True

    or ``force_shared=True`` otherwise does nothing and
    returns the same array back

    Parameters
    ----------

    arr: numpy array
        Array to convert

    logger: :class:`~taurex.log.logger.Logger`
        Logger object to print outputs

    force_shared: bool
        Force conversion to shared memory


    Returns
    -------
    array:
        If enabled and MPI present, shared memory version of array
        otherwise the original array

    """
    try:
        from mpi4py import MPI
    except ImportError:
        return arr
    from taurex.cache import GlobalCache

    if GlobalCache()["mpi_use_shared"] or force_shared:
        if logger is not None:
            logger.info("Moving to shared memory")
        comm = shared_comm()
        nbytes = arr.size * arr.itemsize

        window = MPI.Win.Allocate_shared(nbytes, arr.itemsize, comm=comm)
        buf, itemsize = window.Shared_query(0)
        if itemsize != arr.itemsize:
            raise Exception(
                f"Shared memory size {itemsize} != array itemsize {arr.itemsize}"
            )

        shared_array = np.ndarray(buffer=buf, dtype=arr.dtype, shape=arr.shape)

        if shared_rank() == 0:
            np.copyto(shared_array, arr)

        comm.Barrier()

        return shared_array
    else:
        return arr

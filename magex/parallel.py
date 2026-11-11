"""Thin MPI layer with a serial fallback (mpi4py optional)."""
from __future__ import annotations

import numpy as np

try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK, SIZE = COMM.Get_rank(), COMM.Get_size()
except Exception:  # pragma: no cover
    MPI, COMM, RANK, SIZE = None, None, 0, 1


def split(n, rank=None, size=None):
    """Indices of a block-cyclic share of range(n) for this rank."""
    rank = RANK if rank is None else rank
    size = SIZE if size is None else size
    return np.arange(rank, n, size)


def allreduce_sum(arr):
    if SIZE == 1:
        return arr
    out = np.empty_like(arr)
    COMM.Allreduce(np.ascontiguousarray(arr), out, op=MPI.SUM)
    return out


def allgather_rows(local, idx, n):
    """Assemble an array whose first axis has n rows; this rank owns rows idx."""
    full = np.zeros((n,) + local.shape[1:], dtype=local.dtype)
    full[idx] = local
    return allreduce_sum(full)


def bcast(obj):
    return obj if SIZE == 1 else COMM.bcast(obj, root=0)


def is_root():
    return RANK == 0


def log(*a, **k):
    if RANK == 0:
        print(*a, **k, flush=True)

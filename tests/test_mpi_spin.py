"""MPI domain decomposition (needs mpi4py and mpirun)."""
import os
import shutil
import subprocess
import sys

import pytest


def test_distributed_llg_equals_serial(tmp_path):
    pytest.importorskip("mpi4py")
    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpirun is None:
        pytest.skip("no mpirun")
    here = os.path.dirname(__file__)
    env = dict(os.environ, NUMBA_NUM_THREADS="1")
    r = subprocess.run([mpirun, "-n", "2", sys.executable, os.path.join(here, "mpi_spin_check.py"), str(tmp_path)],
                       env=env, capture_output=True, text=True, timeout=1200)
    assert "MPI-SPIN OK 2" in r.stdout, r.stdout[-2000:] + r.stderr[-3000:]

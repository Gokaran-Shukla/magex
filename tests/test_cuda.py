"""CUDA back end, checked in numba's CUDA simulator (no GPU needed)."""
import os
import subprocess
import sys

import pytest


def test_cuda_kernels_in_simulator(tmp_path):
    pytest.importorskip("numba")
    env = dict(os.environ, NUMBA_ENABLE_CUDASIM="1", NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING="1",
               NUMBA_DISABLE_PERFORMANCE_WARNINGS="1", PYTHONWARNINGS="ignore")
    here = os.path.dirname(__file__)
    r = subprocess.run([sys.executable, os.path.join(here, "cuda_sim_check.py"), str(tmp_path)], env=env,
                       capture_output=True, text=True, timeout=1200)
    assert "CUDA-SIM OK" in r.stdout, r.stdout[-2000:] + r.stderr[-3000:]

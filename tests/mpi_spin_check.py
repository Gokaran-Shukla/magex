"""mpirun -n 2 python tests/mpi_spin_check.py: distributed LLG equals serial at T = 0."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mpi4py import MPI  # noqa: E402

from magex.llg import LLGSolver  # noqa: E402
from magex.models import write_model  # noqa: E402
from magex.mpi_spin import DistributedLLG, DistributedMC  # noqa: E402
from magex.spinmodel import SpinSystem, load_results  # noqa: E402
from magex.structure import apply_structure  # noqa: E402

comm = MPI.COMM_WORLD
p = os.path.join(sys.argv[1], "m.json")
if comm.Get_rank() == 0:
    write_model(p, lattice="sc", J=[1.0], D=0.1, K=0.05)
comm.Barrier()
s = SpinSystem(load_results(p), (12, 8, 6), field=(0.1, 0, 0.4))
apply_structure(s, dict(materials=[dict(name="bulk", Kc=0.02, K2=0.01)]))
e0 = s.init_spins("random", seed=3)
for it in ("heun", "depondt"):
    d = DistributedLLG(s, 0.2, 1e-3, 0.0, 1, it)
    e = d.d.scatter(e0)
    for _ in range(100):
        e = d.step(e)
    eg = d.d.gather(e)
    ref = LLGSolver(s, 0.2, 1e-3, 0.0, 1, integrator=it, backend="numpy")
    er = e0.copy()
    for _ in range(100):
        er = ref.step(er)
    assert np.abs(eg - er).max() < 1e-12, it
    assert abs(d.d.energy(e) - s.energy(er)) < 1e-9 * abs(s.energy(er))
mc = DistributedMC(s, 20.0, 5)
e = mc.d.scatter(s.init_spins("ferro"))
for _ in range(20):
    e = mc.sweep(e)
assert np.all(np.isfinite(mc.d.gather(e)))
if comm.Get_rank() == 0:
    print("MPI-SPIN OK", comm.Get_size())

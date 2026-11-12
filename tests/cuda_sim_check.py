"""Run under NUMBA_ENABLE_CUDASIM=1 (tests/test_cuda.py): CUDA kernels vs CPU kernels and exact statistics."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numba import cuda  # noqa: E402

from magex.cuda_backend import CudaLLG, CudaMC  # noqa: E402
from magex.llg import LLGSolver  # noqa: E402
from magex.models import write_model  # noqa: E402
from magex.spinmodel import K_B, MU_B, SpinSystem, load_results  # noqa: E402
from magex.structure import apply_structure  # noqa: E402

tmp = sys.argv[1]
p = os.path.join(tmp, "m.json")
write_model(p, lattice="sc", J=[1.0], D=0.1, K=0.05)
s = SpinSystem(load_results(p), (4, 4, 3), field=(0.1, 0, 0.5))
apply_structure(s, dict(materials=[dict(name="bulk", Kc=0.02, K2=0.01)]))
e0 = s.init_spins("random", seed=1)
for it in ("heun", "depondt"):
    eg = CudaLLG(s, 0.2, 1e-3, 0.0, 1, it).run(e0, 20)
    c = LLGSolver(s, 0.2, 1e-3, 0.0, 1, integrator=it)
    ec = e0.copy()
    for _ in range(20):
        ec = c.step(ec)
    assert np.abs(eg - ec).max() < 1e-12, it
p2 = os.path.join(tmp, "f.json")
write_model(p2, lattice="sc", J=[0.0], K=1.0)
f = SpinSystem(load_results(p2), (5, 5, 4), (False, False, False), field=(0.2, 0, 2.0))
T = 10.0
th = np.linspace(0, np.pi, 601)
ph = np.linspace(0, 2 * np.pi, 601)
TH, PH = np.meshgrid(th, ph, indexing="ij")
E3 = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1)
En = np.einsum("...a,ab,...b->...", E3, f.A[0], E3) - 2 * MU_B * (E3 @ np.array([0.2, 0, 2.0]))
w = np.exp(-(En - En.min()) / (K_B * T)) * np.sin(TH)
I = lambda q: np.trapezoid(np.trapezoid(q, ph, axis=1), th)  # noqa: E731
mz = I(w * E3[..., 2]) / I(w)
for mv in ("gaussian", "heatbath"):
    mc = CudaMC(f, T, 3, 0.6, mv)
    ed = cuda.to_device(f.init_spins("ferro"))
    ms = []
    for n in range(300):
        mc.sweep_device(ed)
        if n > 30:
            ms.append(ed.copy_to_host()[:, 2].mean())
    err = np.std(ms) / np.sqrt(len(ms) / 3)
    assert abs(np.mean(ms) - mz) < 4 * err + 2e-3, (mv, np.mean(ms), mz, err)
print("CUDA-SIM OK")

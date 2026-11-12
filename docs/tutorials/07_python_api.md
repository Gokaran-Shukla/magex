# 7 · Python API

Everything the command line does is available as library calls.

## Atomistic

```python
import numpy as np
from magex.models import write_model
from magex.spinmodel import load_results, SpinSystem, mean_field_tc
from magex.llg import LLGSolver
from magex.montecarlo import MetropolisMC
from magex import texture

write_model("tri.json", lattice="triangular", J=[2.0], D=1.0, moment=3.0)
uc = load_results("tri.json")
sys = SpinSystem(uc, supercell=(48, 48, 1), pbc=(True, True, False), field=(0, 0, 2.0))

e = sys.init_spins("random", seed=1)
llg = LLGSolver(sys, alpha=0.5, dt=2e-3, temperature=0.0)
e, table, snaps = llg.run(e, steps=20000, output_every=200, snapshots=10)
print("Q =", texture.topological_charge(sys, e))
texture.plot_texture(sys, e, "skx.png")

mc = MetropolisMC(sys, temperature=20.0, seed=3)
for _ in range(1000):
    e = mc.sweep(e)
```

`SpinSystem` exposes `energy(e)`, `gradient(e)`, `effective_field(e)` (tesla), `order_parameter(e)`,
`set_field(B)`, and the sparse coupling matrix `Jsp` (3N×3N), so custom integrators or analysis
(spin–spin correlations, structure factors) are a few lines.

## Micromagnetic

```python
import numpy as np
from magex.micromag import Mesh, Micromagnet, geometry_mask, material, RK45, relax, hysteresis
from magex.micromag.plot import plot_state, plot_table

mesh = Mesh(n=(100, 100, 1), d=(1e-9, 1e-9, 0.6e-9))
mm = Micromagnet(mesh, demag=True)
mm.add_region(geometry_mask(mesh, "disk", diameter_nm=100), "CoPt", **material("Co/Pt"))

m = mm.init("skyrmion", radius_nm=10)
m = relax(mm, m)
plot_state(mm, m, "relaxed.png")

# time-dependent field: any callable t -> (3,) tesla
mm.B_func = lambda t: np.array([0, 0, -0.05]) * (t < 0.1e-9)
m, table = RK45(mm, tol=1e-5).run(m, 0.3e-9, record_every=5e-12)
plot_table(table, "dynamics.png")

# non-uniform static field, e.g. from a nearby tip
X, Y, Z = mesh.coords()
mm.B_ext_cell = np.zeros(mesh.n + (3,))
mm.B_ext_cell[..., 2] = 0.1 * np.exp(-(X**2 + Y**2) / (20e-9) ** 2)

# SOT with an arbitrary current waveform
mm.sot = dict(J=1e12, theta=0.1, sigma=[0, 1, 0], J_func=lambda t: np.sin(2 * np.pi * 5e9 * t))
```

`Micromagnet.fields(m)` returns each field term separately, `energies(m)` the energy per term
(J), `average(m)` the M_s-weighted ⟨m⟩; `magex.micromag.write_ovf / read_ovf` exchange data with
OOMMF and mumax.

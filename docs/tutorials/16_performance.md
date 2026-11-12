# 16 · Performance: threads, MPI, GPU and restarts

| engine | parallelism |
|--------|-------------|
| `magex run`, `magex damping` | MPI over k-points (`mpirun -n 40 magex run ...`) |
| `magex mc` | one process per temperature point (`nproc`); parallel tempering runs replicas in one process |
| `magex llg`, `magex sqw`, `magex gneb` | sparse matrix–vector products (multithreaded BLAS) |
| `magex micromag` | multithreaded FFTs (`scipy.fft`, all cores); optional GPU |
| Wang–Landau | numba-compiled kernel (`pip install magex[fast]`) |

## GPU (experimental)

```python
import cupy as cp
from magex.micromag.driver import build
from magex.micromag.solver import RK45

sysm = build(cfg, base)
m = sysm.init("skyrmion", radius_nm=15)
sysm.to_gpu()                     # parameters, demag kernel, caches -> device; scipy.fft -> cupyx
m, table = RK45(sysm).run(cp.asarray(m), 1e-9)
```

The solver is written against the NumPy API; CuPy arrays dispatch through NumPy's
`__array_function__` protocol.  This path has not yet been validated on GPU hardware: run
`python -m magex.micromag.gpu` once on a GPU node — it compares one field evaluation and 20 ps of
RK45 dynamics between CPU and GPU and prints the timings — before using it for production.

## Checkpoints and restarts

```toml
[[stage]]
type = "run"
time_ns = 20
checkpoint_ps = 500             # write output/checkpoint.npz every 0.5 ns during the stage

[output]
dir = "long_run"
restart = true                  # on start: skip finished stages and resume the interrupted one
```

A checkpoint is also written after every finished stage.  Restarted runs reproduce uninterrupted
runs exactly (same final ⟨m⟩ to all printed digits in the check).  Atomistic LLG runs write
`spins_*.npz`, which `init = "file"` reloads.

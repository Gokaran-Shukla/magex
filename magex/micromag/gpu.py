"""Optional CuPy (NVIDIA GPU) backend for the micromagnetic solver.

    sysm = build(cfg)          # or Micromagnet(...)
    sysm.to_gpu()              # parameters, demag kernel and caches -> device
    m = cupy.asarray(m)
    m, table = RK45(sysm).run(m, 1e-9)

The solver code is written against the NumPy API; CuPy arrays dispatch through NumPy's
__array_function__ protocol and scipy.fft is switched to the cupyx.scipy.fft backend.
Status: experimental — the NumPy path is covered by the test-suite; the CuPy path was written
without access to a GPU and should be validated with ``python -m magex.micromag.gpu`` (compares one
field evaluation and 100 RK45 steps on CPU and GPU) before production use.
"""
from __future__ import annotations

import numpy as np

_ARRAYS = ("Ms", "A", "D_int", "D_bulk", "Ku1", "Ku2", "Kc1", "alpha", "anisU", "anisC1", "anisC2", "region")
_OPTIONAL = ("B_ext_cell", "oersted", "grain")


def _cupy():
    try:
        import cupy
        import cupyx.scipy.fft as cufft
        import scipy.fft
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("CuPy is not installed (pip install cupy-cuda12x)") from exc
    scipy.fft.set_global_backend(cufft)
    return cupy


def to_gpu(system):
    cp = _cupy()
    for name in _ARRAYS:
        setattr(system, name, cp.asarray(getattr(system, name)))
    for name in _OPTIONAL:
        if getattr(system, name, None) is not None:
            setattr(system, name, cp.asarray(getattr(system, name)))
    system.B_local = [(cp.asarray(mk), fn) for mk, fn in system.B_local]
    if system.use_demag:
        d = system.demag                      # build on CPU, then move the kernel
        d.K = cp.asarray(d.K)
        d.workers = None
    system.xp = cp
    system.invalidate()
    return system


def to_cpu(system):
    for name in _ARRAYS + _OPTIONAL:
        a = getattr(system, name, None)
        if a is not None and hasattr(a, "get"):
            setattr(system, name, a.get())
    system.B_local = [(mk.get() if hasattr(mk, "get") else mk, fn) for mk, fn in system.B_local]
    if system.use_demag and system._demag is not None and hasattr(system._demag.K, "get"):
        system._demag.K = system._demag.K.get()
        system._demag.workers = -1
    system.xp = np
    import scipy.fft
    try:
        scipy.fft.set_global_backend("scipy")
    except Exception:  # pragma: no cover
        pass
    system.invalidate()
    return system


def _selfcheck():  # pragma: no cover - needs a GPU
    import time
    import cupy as cp
    from .mesh import Mesh
    from .materials import material
    from .solver import RK45
    from .system import Micromagnet
    mesh = Mesh((128, 128, 1), (2e-9, 2e-9, 1e-9))
    s = Micromagnet(mesh)
    s.add_region(np.ones(mesh.n, bool), **material("Co/Pt"))
    m = s.init("skyrmion", radius_nm=15)
    B_cpu = s.effective_field(m)
    t = time.time()
    m1, _ = RK45(s).run(m.copy(), 20e-12)
    tc = time.time() - t
    s.to_gpu()
    mg = cp.asarray(m)
    B_gpu = cp.asnumpy(s.effective_field(mg))
    t = time.time()
    m2, _ = RK45(s).run(mg, 20e-12)
    cp.cuda.Stream.null.synchronize()
    tg = time.time() - t
    print("max |B_cpu - B_gpu| =", float(np.abs(B_cpu - B_gpu).max()), "T")
    print("max |m_cpu - m_gpu| after 20 ps =", float(np.abs(m1 - cp.asnumpy(m2)).max()))
    print(f"CPU {tc:.2f} s, GPU {tg:.2f} s")


if __name__ == "__main__":  # pragma: no cover
    _selfcheck()

"""Distributed-memory spin dynamics and Monte Carlo (MPI domain decomposition, mpi4py).

The supercell is cut into slabs along a cell axis; every rank owns the sites of its slab and keeps ghost
copies of the neighbours it needs.  The local coupling matrix contains only the rows of owned sites
(columns in the local numbering owned + ghost), so the inner kernels are the same multithreaded numba
kernels as the shared-memory path (ghost rows are marked inactive).  One halo exchange per Heun stage
(two per step) and one per Monte Carlo colour keep the ghosts current, so the algorithm is the serial one:
at T = 0 the distributed trajectory equals the serial trajectory to round-off (tests/test_mpi_spin.py).

    mpirun -n 4 python -m magex.mpi_spin llg input.toml     (or use DistributedLLG in a script)

Scope: exchange tensors, DMI, single-ion and on-site cubic / higher uniaxial terms, Zeeman, pinning, site
temperatures.  Long-range dipoles and multi-site (biquadratic, four-spin) terms are not distributed.
The SpinSystem is built on every rank (set-up memory O(N) per rank); the dynamical state, the work and
the communication are distributed.
"""
from __future__ import annotations

import numpy as np

from .spinmodel import GAMMA, K_B, MU_B


def _comm():
    from mpi4py import MPI
    return MPI.COMM_WORLD


class Decomposition:
    def __init__(self, system, comm=None, axis=None):
        from . import fast
        from .terms import CubicAnisotropy, UniaxialHigher
        if any(not isinstance(t, (CubicAnisotropy, UniaxialHigher)) for t in system.terms):
            raise NotImplementedError("MPI decomposition: long-range / multi-site terms are not distributed")
        if system.has_higher:
            raise NotImplementedError("MPI decomposition: biquadratic / four-spin terms are not distributed")
        self.comm = comm or _comm()
        self.rank, self.size = self.comm.Get_rank(), self.comm.Get_size()
        self.sys = system
        N = system.N
        cells = system.cells[system.cell_index]
        if axis is None:
            axis = int(np.argmax(system.L))
        self.axis = axis
        L = system.L[axis]
        bounds = np.linspace(0, L, self.size + 1).round().astype(int)
        owner = np.searchsorted(bounds, cells[:, axis], side="right") - 1
        self.owner = owner
        own = np.where(owner == self.rank)[0]
        J = system.Jsp.tocsr()
        rows = (3 * own[:, None] + np.arange(3)[None, :]).ravel()
        Jr = J[rows]
        cols_sites = np.unique(Jr.indices // 3)
        ghost = np.setdiff1d(cols_sites, own)
        self.own, self.ghost = own, ghost
        self.gl = np.concatenate([own, ghost])                       # local -> global
        self.n_own, self.n_loc = len(own), len(own) + len(ghost)
        g2l = -np.ones(N, np.int64)
        g2l[self.gl] = np.arange(self.n_loc)
        self.g2l = g2l
        # local square coupling matrix (ghost rows empty)
        import scipy.sparse as sp
        Jc = Jr.tocoo()
        lc = 3 * g2l[Jc.col // 3] + Jc.col % 3
        Jloc = sp.csr_matrix((Jc.data, (Jc.row, lc)), shape=(3 * self.n_loc, 3 * self.n_loc))
        self.ptr, self.nbr, self.Jb = fast.neighbour_blocks(Jloc, self.n_loc)
        self.Jrows = Jloc[:3 * self.n_own].tocsr()
        self.A = np.ascontiguousarray(system.A[self.gl])
        self.mu = np.ascontiguousarray(system.mu[self.gl])
        fm = fast.FastModel(system)
        self.Kc, self.C, self.K2, self.K3, self.u = fm.Kc[self.gl].copy(), fm.C, fm.K2[self.gl].copy(), \
            fm.K3[self.gl].copy(), fm.u
        self.has_c, self.has_u = fm.has_c, fm.has_u
        pin = np.zeros(self.n_loc, np.bool_)
        if system.pinned is not None:
            pin |= system.pinned[self.gl]
        self.pinned_own = pin.copy()
        pin[self.n_own:] = True                                       # ghosts are never updated locally
        self.inactive = pin
        # halo plan: who needs which of my sites
        need = [ghost[owner[ghost] == r] for r in range(self.size)]
        recv_lists = need
        send_lists = self.comm.alltoall(recv_lists)
        self.send = [(r, g2l[np.asarray(lst, np.int64)]) for r, lst in enumerate(send_lists) if len(lst) and r != self.rank]
        self.recv = [(r, g2l[np.asarray(lst, np.int64)]) for r, lst in enumerate(recv_lists) if len(lst) and r != self.rank]

    def args(self):
        A = self.A if (self.sys.has_A or self.has_c or self.has_u) else np.zeros((0, 3, 3))
        return (self.ptr, self.nbr, self.Jb, A, self.mu, np.asarray(self.sys.B, float), self.Kc, self.C, self.K2,
                self.K3, self.u, self.has_c, self.has_u, self.inactive)

    def halo(self, x):
        """Update the ghost rows of the local array x (n_loc, 3)."""
        from mpi4py import MPI
        reqs, bufs = [], []
        for r, idx in self.recv:
            b = np.empty((len(idx), 3))
            bufs.append((idx, b))
            reqs.append(self.comm.Irecv(b, source=r, tag=7))
        for r, idx in self.send:
            reqs.append(self.comm.Isend(np.ascontiguousarray(x[idx]), dest=r, tag=7))
        MPI.Request.Waitall(reqs)
        for idx, b in bufs:
            x[idx] = b

    def scatter(self, e_global):
        return np.ascontiguousarray(e_global[self.gl], float)

    def gather(self, e_local):
        """Full configuration on every rank."""
        parts = self.comm.allgather((self.own, e_local[:self.n_own]))
        out = np.empty((self.sys.N, 3))
        for idx, v in parts:
            out[idx] = v
        return out

    def magnetization(self, e_local):
        m = e_local[:self.n_own].sum(0)
        return self.comm.allreduce(m) / self.sys.N

    def energy(self, e_local):
        """Total energy (meV): owned rows of the bilinear form + on-site + Zeeman, summed over ranks."""
        n = self.n_own
        e = e_local[:n]
        h = (self.Jrows @ e_local.reshape(-1)).reshape(-1, 3)
        E = -np.sum(e * h) - MU_B * np.sum(self.mu[:n] * (e @ np.asarray(self.sys.B, float)))
        if self.sys.has_A:
            E += np.einsum("ia,iab,ib->", e, self.A[:n], e)
        if self.has_c:
            a2 = (e @ self.C.T) ** 2
            E += np.sum(self.Kc[:n] * (a2[:, 0] * a2[:, 1] + a2[:, 1] * a2[:, 2] + a2[:, 2] * a2[:, 0]))
        if self.has_u:
            c = e @ self.u
            E += np.sum(-self.K2[:n] * c**4 - self.K3[:n] * c**6)
        return self.comm.allreduce(float(E))


class DistributedLLG:
    """Stochastic Heun / Depondt LLG with domain decomposition (same equations and noise as LLGSolver)."""

    def __init__(self, system, alpha=0.1, dt=1e-3, temperature=0.0, seed=1, integrator="heun", comm=None, axis=None):
        from . import fast
        self.d = Decomposition(system, comm, axis)
        self.sys = system
        self.alpha = np.broadcast_to(np.asarray(alpha, float), (system.N,))[self.d.gl].copy()
        self.dt = float(dt)
        self.T = float(temperature)
        self.depondt = integrator == "depondt"
        self.streams = fast.seed_streams((int(seed) + 7919 * self.d.rank) % (2**31))
        n = self.d.n_loc
        self.ep, self.k1, self.b = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
        self.time = 0.0
        self._sig = None

    def _sigma(self):
        from .structure import site_temperature
        T = np.broadcast_to(site_temperature(self.sys, self.T), (self.sys.N,))[self.d.gl]
        return np.ascontiguousarray(np.sqrt(2.0 * self.alpha * K_B * T / (GAMMA * self.d.mu * MU_B * self.dt)))

    def step(self, e):
        from . import fast
        d = self.d
        noisy = self.T > 0 or getattr(self.sys, "site_T", None) is not None
        if noisy and self._sig is None:
            self._sig = self._sigma()
        sig = self._sig if noisy else np.zeros(d.n_loc)
        f1 = fast.pick("_fused1", d.n_loc)
        f2 = fast.pick("_fused2", d.n_loc)
        f1(e, self.ep, self.k1, self.b, self.streams, sig, self.alpha, self.dt, True, self.depondt, noisy, *d.args())
        d.halo(self.ep)
        f2(e, self.ep, self.k1, self.b, self.alpha, self.dt, True, self.depondt, *d.args())
        d.halo(e)
        self.time += self.dt
        return e

    def run(self, e_global, steps, every=100):
        e = self.d.scatter(e_global)
        rows = []
        for n in range(steps + 1):
            if n % every == 0:
                rows.append([self.time, *self.d.magnetization(e)])
            if n < steps:
                e = self.step(e)
        return e, np.array(rows)


class DistributedMC:
    """Metropolis / heat-bath Monte Carlo over the global colour groups, each split among the ranks."""

    def __init__(self, system, temperature, seed=1, sigma=0.6, move="gaussian", comm=None, axis=None):
        from . import fast
        from .montecarlo import greedy_coloring
        self.d = Decomposition(system, comm, axis)
        self.sys = system
        self.T = float(temperature)
        self.sigma = float(sigma)
        self.move = {"gaussian": 0, "uniform": 1, "heatbath": 2}[move]
        col = greedy_coloring(system.N, system.bonds)
        d = self.d
        owned = np.zeros(system.N, bool)
        owned[d.own] = True
        self.groups = []
        for c in range(col.max() + 1):
            g = np.where((col == c) & owned)[0]
            if system.pinned is not None:
                g = g[~system.pinned[g]]
            self.groups.append(np.ascontiguousarray(d.g2l[g], np.int64))
        self.streams = fast.seed_streams((int(seed) + 104729 * d.rank) % (2**31))
        self.accepted = 0
        self.tried = 0

    def sweep(self, e):
        from . import fast
        d = self.d
        from .structure import site_temperature
        T = np.broadcast_to(site_temperature(self.sys, self.T), (self.sys.N,))[d.gl]
        beta = np.where(T > 0, 1.0 / (K_B * np.maximum(T, 1e-300)), np.inf)
        mv = self.move if (self.move != 2 or np.all(np.isfinite(beta))) else 0
        args = d.args()[:-1]
        for idx in self.groups:
            k = fast.pick("_mc_group", len(idx))
            self.accepted += int(k(e, idx, self.streams, *args, beta, self.sigma, mv, False, self.sys.has_A))
            self.tried += len(idx)
            d.halo(e)
        return e

    def acceptance(self):
        a = self.d.comm.allreduce(self.accepted) / max(self.d.comm.allreduce(self.tried), 1)
        self.accepted = self.tried = 0
        return a


def _main(argv=None):  # pragma: no cover - exercised through mpirun in tests
    import argparse
    import os
    import tomllib
    import time
    ap = argparse.ArgumentParser(prog="python -m magex.mpi_spin")
    ap.add_argument("task", choices=["llg", "mc"])
    ap.add_argument("config")
    a = ap.parse_args(argv)
    from .spinmodel import SpinSystem, build_from_config
    from .structure import apply_structure
    cfg = tomllib.load(open(a.config, "rb"))
    base = os.path.dirname(os.path.abspath(a.config))
    uc, sm = build_from_config(cfg, base)
    c = dict(cfg.get(a.task, {}))
    system = SpinSystem(uc, sm.get("supercell", [32, 32, 32]), sm.get("pbc", [True, True, True]), c.get("field", [0, 0, 0]))
    apply_structure(system, cfg.get("structure"))
    e0 = system.init_spins(c.get("init", "ferro"), c.get("init_axis", [0, 0, 1]), c.get("seed", 1))
    comm = _comm()
    t0 = time.time()
    if a.task == "llg":
        s = DistributedLLG(system, c.get("alpha", 0.1), c.get("dt", 1e-3), c.get("temperature", 0.0), c.get("seed", 1),
                           c.get("integrator", "heun"))
        e, tab = s.run(e0, int(c.get("steps", 1000)), int(c.get("output_every", 100)))
        if comm.Get_rank() == 0:
            for r in tab:
                print(f"t = {r[0]:9.3f} ps  m = ({r[1]: .4f}, {r[2]: .4f}, {r[3]: .4f})")
    else:
        s = DistributedMC(system, c.get("temperature", 10.0), c.get("seed", 1), c.get("sigma", 0.6), c.get("move", "gaussian"))
        e = s.d.scatter(e0)
        ms = []
        for n in range(int(c.get("sweeps", 1000))):
            e = s.sweep(e)
            ms.append(np.linalg.norm(s.d.magnetization(e)))
        if comm.Get_rank() == 0:
            print(f"<|m|> = {np.mean(ms[len(ms) // 2:]):.5f}, acceptance {s.acceptance():.3f}")
    if comm.Get_rank() == 0:
        print(f"{comm.Get_size()} ranks, {system.N} spins, wall {time.time() - t0:.1f} s")


if __name__ == "__main__":  # pragma: no cover
    _main()

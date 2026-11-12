"""Coupled spin-lattice dynamics (SLD), in the spirit of Ma, Woo & Dudarev, PRB 78, 024434 (2008) and
the UppASD / LAMMPS-SPIN implementations.

Hamiltonian (meV, A, ps, amu):

    H = sum_i p_i^2 / 2m + (k/2) sum_<ij> (r_ij - r0_ij)^2
        + E_spin[e]  -  2 sum_<ij> [J(r_ij) - J(r0_ij)] e_i . e_j

E_spin is the full static magex spin model (all exchange tensors, DMI, anisotropy, field); the
nearest-neighbour exchange is modulated by the bond length,

    J(r) = J0 exp(-(r - r0)/lambda)            (form = "exp"),   or
    J(r) = J0 (1 - r/rc)^3 Theta(rc - r) / (1 - r0/rc)^3   (form = "cubic", Ma et al.),

which gives magnetoelastic forces F = 2 J'(r) (e_i.e_j) r_hat.  Integration: velocity Verlet for the
lattice (optionally Langevin, BAOAB splitting) wrapped around one Heun/Depondt LLG step for the spins
(Strang splitting).  Without damping the total energy is conserved to O(dt^2).
"""
from __future__ import annotations

import numpy as np

from .spinmodel import GAMMA, K_B

AMU = 0.103642696            # meV ps^2 / A^2 per amu


class SpinLattice:
    def __init__(self, system, J0=None, lam=0.5, form="exp", rc=None, k_spring=1000.0, mass=55.845, potential=None,
                 alpha=0.0, T_spin=0.0, eta=0.0, T_lattice=0.0, dt=1e-3, seed=1, integrator="auto", shells="nn"):
        """integrator: "symplectic" (default when alpha = 0 and T_spin = 0: time-reversible splitting with
        exact per-spin implicit-midpoint rotations over non-interacting colour groups, so that the coupled
        spin-lattice energy is conserved without drift), or any LLG integrator ("heun", "depondt", ...).
        shells: "nn" (nearest-neighbour bonds carry J(r)) or "all" (every isotropic bond of the model)."""
        from .llg import LLGSolver
        self.sys = system
        P = system.positions
        b = system.bonds
        b = b[b[:, 0] < b[:, 1]]
        d = system._minimg(P[b[:, 1]] - P[b[:, 0]])
        r = np.linalg.norm(d, axis=1)
        nn = np.abs(r - r.min()) < 1e-3 if shells == "nn" else np.ones(len(r), bool)
        self.pairs = b[nn]
        self.r0 = r[nn]
        if J0 is None:                                   # isotropic part of the static coupling of every bond
            from .fast import neighbour_blocks
            ptr, nbr, (typ, table) = neighbour_blocks(system.Jsp, system.N)
            iso = table[:, 0] if table.shape[1] in (1, 4) else (table[:, 0] + table[:, 4] + table[:, 8]) / 3
            src = np.repeat(np.arange(system.N), np.diff(ptr))
            key = src.astype(np.int64) * system.N + nbr
            lut = dict(zip(key.tolist(), iso[typ].tolist()))
            J0 = np.array([lut.get(int(a) * system.N + int(c), 0.0) for a, c in self.pairs])
        self.J0 = np.broadcast_to(np.asarray(J0, float), (len(self.pairs),)).copy()
        self.lam, self.form = float(lam), form
        self.rc = float(rc) if rc else 1.5 * float(self.r0.max())
        self.k = float(k_spring)
        # lattice pair potential V(r) per bond: {"type": "harmonic"} (k_spring), {"type": "morse", "D": meV, "a": 1/A},
        # {"type": "lj", "eps": meV} (sigma = r0 / 2^(1/6): minimum at the reference bond length),
        # {"type": "fpu", "k3": meV/A^3, "k4": meV/A^4} (k/2 x^2 + k3/3 x^3 + k4/4 x^4, x = r - r0)
        self.pot = dict(type="harmonic") if potential is None else dict(potential)
        self.m = float(mass) * AMU
        self.u = np.zeros_like(P)                         # displacements (A)
        self.v = np.zeros_like(P)                         # velocities (A/ps)
        self.dt = float(dt)
        self.eta, self.TL = float(eta), float(T_lattice)
        self.rng = np.random.default_rng(seed)
        if integrator == "auto":
            integrator = "symplectic" if (alpha == 0 and T_spin == 0) else "heun"
        self.integrator = integrator
        self.spin = LLGSolver(system, alpha, dt, T_spin, seed, integrator="heun" if integrator == "symplectic"
                              else integrator)
        if integrator == "symplectic":
            from .montecarlo import greedy_coloring
            bb = np.concatenate([system.bonds, self.pairs, self.pairs[:, ::-1]])
            col = greedy_coloring(system.N, bb)
            self.groups = [np.where(col == c)[0] for c in range(col.max() + 1)]
        self._J0full = self._J(self.r0)
        self._dJsp = None

    # ----------------------------------------------------------------- exchange(r)
    def _J(self, r):
        if self.form == "exp":
            return self.J0 * np.exp(-(r - self.r0) / self.lam)
        x = np.clip(1 - r / self.rc, 0, None)
        return self.J0 * x**3 / (1 - self.r0 / self.rc) ** 3

    def _dJ(self, r):
        if self.form == "exp":
            return -self._J(r) / self.lam
        x = np.clip(1 - r / self.rc, 0, None)
        return -3 * self.J0 * x**2 / self.rc / (1 - self.r0 / self.rc) ** 3

    def pair(self, r):
        """Lattice pair energy V(r) and dV/dr (meV, meV/A) for every bond."""
        x = r - self.r0
        t = self.pot.get("type", "harmonic")
        if t == "harmonic":
            return 0.5 * self.k * x**2, self.k * x
        if t == "fpu":
            k3, k4 = float(self.pot.get("k3", 0.0)), float(self.pot.get("k4", 0.0))
            return 0.5 * self.k * x**2 + k3 / 3 * x**3 + k4 / 4 * x**4, self.k * x + k3 * x**2 + k4 * x**3
        if t == "morse":
            D, a = float(self.pot["D"]), float(self.pot["a"])
            q = np.exp(-a * x)
            return D * (1 - q) ** 2, 2 * D * a * q * (1 - q)
        if t == "lj":
            eps = float(self.pot["eps"])
            sig = self.r0 / 2 ** (1 / 6)
            s6 = (sig / r) ** 6
            return 4 * eps * (s6 * s6 - s6) + eps, 4 * eps * (-12 * s6 * s6 + 6 * s6) / r
        raise ValueError(f"unknown lattice potential '{t}'")

    def _geom(self):
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        P = self.sys.positions + self.u
        d = self.sys._minimg(P[j] - P[i])
        r = np.linalg.norm(d, axis=1)
        return i, j, d, r

    # ----------------------------------------------------------------- energies and forces
    def magnetoelastic_field(self, e):
        """Extra -dE/de (meV) from the bond-length dependence of J."""
        i, j, d, r = self._geom()
        dJ = (self._J(r) - self._J0full)[:, None]
        h = np.zeros_like(e)
        np.add.at(h, i, 2 * dJ * e[j])
        np.add.at(h, j, 2 * dJ * e[i])
        return h

    def forces(self, e):
        i, j, d, r = self._geom()
        rh = d / r[:, None]
        ee = np.einsum("ia,ia->i", e[i], e[j])
        # dV/dr = k (r - r0) - 2 J'(r) e_i.e_j ;  F_j = -dV/dr rhat, F_i = +dV/dr rhat
        dV = self.pair(r)[1] - 2 * self._dJ(r) * ee
        F = np.zeros_like(self.u)
        np.add.at(F, i, dV[:, None] * rh)
        np.add.at(F, j, -dV[:, None] * rh)
        return F

    def energy(self, e):
        i, j, d, r = self._geom()
        ee = np.einsum("ia,ia->i", e[i], e[j])
        Vl = float(np.sum(self.pair(r)[0]))
        Epot = Vl - 2 * np.sum((self._J(r) - self._J0full) * ee)
        Ekin = 0.5 * self.m * np.sum(self.v**2)
        return dict(kinetic=float(Ekin), lattice=Vl,
                    spin=float(self.sys.energy(e)), coupling=float(Epot - Vl),
                    total=float(Ekin + Epot + self.sys.energy(e)))

    def temperature_lattice(self):
        return float(self.m * np.sum(self.v**2) / (3 * len(self.v) * K_B))

    # ----------------------------------------------------------------- integrator
    def step(self, e):
        dt = self.dt
        sp = self.spin
        # palindromic splitting  K(h/2) D(h/2) [O(h)] S(h) D(h/2) K(h/2): second order, symplectic without baths
        F = self.forces(e)
        self.v += 0.5 * dt * F / self.m
        self.u += 0.5 * dt * self.v
        if self.eta > 0:                                  # O step (Ornstein-Uhlenbeck, exact)
            c = np.exp(-self.eta * dt)
            self.v = c * self.v + np.sqrt((1 - c * c) * K_B * self.TL / self.m) * self.rng.normal(size=self.v.shape)
        # spin step in the exchange field of the mid-step lattice
        mu_muB = (self.sys.mu * 5.7883818060e-2)[:, None]
        if self.integrator == "symplectic":
            e = self._spin_symplectic(e, dt)
            sp.time += dt
        else:
            sp.extra_field = lambda e_: self.magnetoelastic_field(e_) / mu_muB
            e = sp.step(e)
        self.u += 0.5 * dt * self.v
        F = self.forces(e)
        self.v += 0.5 * dt * F / self.m
        return e

    def _bfield(self, e):
        """Precession field (T) of the spin model plus the bond-length dependent exchange."""
        return self.sys.effective_field(e) + self.magnetoelastic_field(e) / (self.sys.mu * 5.7883818060e-2)[:, None]

    def _spin_symplectic(self, e, dt):
        """Palindromic sweep over colour groups (h/2 forward, h/2 backward); every spin of a group is rotated
        by the implicit midpoint rule e' = e + h gamma B(e_mid) x e_mid, solved by fixed-point iteration
        (exact for fields that do not depend on the spin itself)."""
        e = e.copy()
        order = self.groups + self.groups[::-1]
        h = 0.5 * dt
        for idx in order:
            eo = e[idx]
            en = eo.copy()
            for _ in range(8):
                mid = 0.5 * (eo + en)
                e[idx] = mid / np.linalg.norm(mid, axis=1, keepdims=True)
                B = self._bfield(e)[idx]
                w = h * GAMMA * B                          # rotation vector: de/dt = gamma B x e
                ax = np.cross(w, eo)
                new = eo + (ax + 0.5 * np.cross(w, ax)) / (1 + 0.25 * np.einsum("ia,ia->i", w, w))[:, None]
                if np.max(np.abs(new - en)) < 1e-14:
                    en = new
                    break
                en = new
            e[idx] = en
        return e

    def run(self, e, steps, every=10):
        rows = []
        for n in range(steps + 1):
            if n % every == 0:
                En = self.energy(e)
                rows.append([self.spin.time, En["total"], En["spin"], En["lattice"], En["kinetic"], En["coupling"],
                             self.temperature_lattice(), *e.mean(0)])
            if n < steps:
                e = self.step(e)
        return e, np.array(rows)

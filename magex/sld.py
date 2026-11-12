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

from .spinmodel import K_B

AMU = 0.103642696            # meV ps^2 / A^2 per amu


class SpinLattice:
    def __init__(self, system, J0=None, lam=0.5, form="exp", rc=None, k_spring=1000.0, mass=55.845,
                 alpha=0.0, T_spin=0.0, eta=0.0, T_lattice=0.0, dt=1e-3, seed=1, integrator="heun"):
        from .llg import LLGSolver
        self.sys = system
        P = system.positions
        b = system.bonds
        b = b[b[:, 0] < b[:, 1]]
        d = system._minimg(P[b[:, 1]] - P[b[:, 0]])
        r = np.linalg.norm(d, axis=1)
        nn = np.abs(r - r.min()) < 1e-3
        self.pairs = b[nn]
        self.r0 = r[nn]
        if J0 is None:                                   # isotropic nn exchange of the static model
            J = system.Jsp.tocsr()
            i, j = self.pairs[:, 0], self.pairs[:, 1]
            J0 = np.array([np.trace(J[3 * a:3 * a + 3, 3 * c:3 * c + 3].toarray()) / 3 for a, c in zip(i, j)])
        self.J0 = np.broadcast_to(np.asarray(J0, float), (len(self.pairs),)).copy()
        self.lam, self.form = float(lam), form
        self.rc = float(rc) if rc else 1.5 * float(self.r0.max())
        self.k = float(k_spring)
        self.m = float(mass) * AMU
        self.u = np.zeros_like(P)                         # displacements (A)
        self.v = np.zeros_like(P)                         # velocities (A/ps)
        self.dt = float(dt)
        self.eta, self.TL = float(eta), float(T_lattice)
        self.rng = np.random.default_rng(seed)
        self.spin = LLGSolver(system, alpha, dt, T_spin, seed, integrator=integrator)
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
        dV = self.k * (r - self.r0) - 2 * self._dJ(r) * ee
        F = np.zeros_like(self.u)
        np.add.at(F, i, dV[:, None] * rh)
        np.add.at(F, j, -dV[:, None] * rh)
        return F

    def energy(self, e):
        i, j, d, r = self._geom()
        ee = np.einsum("ia,ia->i", e[i], e[j])
        Epot = 0.5 * self.k * np.sum((r - self.r0) ** 2) - 2 * np.sum((self._J(r) - self._J0full) * ee)
        Ekin = 0.5 * self.m * np.sum(self.v**2)
        return dict(kinetic=float(Ekin), lattice=float(0.5 * self.k * np.sum((r - self.r0) ** 2)),
                    spin=float(self.sys.energy(e)), coupling=float(Epot - 0.5 * self.k * np.sum((r - self.r0) ** 2)),
                    total=float(Ekin + Epot + self.sys.energy(e)))

    def temperature_lattice(self):
        return float(self.m * np.sum(self.v**2) / (3 * len(self.v) * K_B))

    # ----------------------------------------------------------------- integrator
    def step(self, e):
        dt = self.dt
        sp = self.spin
        F = self.forces(e)
        self.v += 0.5 * dt * F / self.m
        self.u += 0.5 * dt * self.v
        if self.eta > 0:                                  # O step (Ornstein-Uhlenbeck, exact)
            c = np.exp(-self.eta * dt)
            self.v = c * self.v + np.sqrt((1 - c * c) * K_B * self.TL / self.m) * self.rng.normal(size=self.v.shape)
        self.u += 0.5 * dt * self.v
        # spin step in the exchange field of the current lattice
        mu_muB = (self.sys.mu * 5.7883818060e-2)[:, None]
        sp.extra_field = lambda e_: self.magnetoelastic_field(e_) / mu_muB
        e = sp.step(e)
        F = self.forces(e)
        self.v += 0.5 * dt * F / self.m
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

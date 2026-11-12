"""Langevin spin dynamics with longitudinal fluctuations (Ma & Dudarev, PRB 86, 054416 (2012)).

The moments are vectors M_i = s_i e_i (s in units of the reference moment) with the Hamiltonian of
``LSFMonteCarlo`` (exchange tensors, single-ion anisotropy, Zeeman, Landau a s^2 + b s^4):

    dM_i/dt = -gamma M_i x B_i + lambda H_i + xi_i,     H_i = -dE/dM_i (meV),  B_i = H_i / (mu_i mu_B)
    <xi_ia(t) xi_jb(t')> = 2 lambda k_B T delta_ij delta_ab delta(t - t')

The precession conserves the energy, the longitudinal + transverse relaxation (lambda, 1/(ps meV)) and the
additive noise drive the system to the Boltzmann distribution exp(-E/k_BT) d^3M, i.e. the same ensemble
as ``LSFMonteCarlo(measure="d3M")`` (checked in tests/test_fast.py).  Stochastic Heun integrator.
"""
from __future__ import annotations

import numpy as np

from .spinmodel import GAMMA, K_B, MU_B


class LongitudinalSpinDynamics:
    def __init__(self, system, temperature=0.0, lam=0.01, dt=1e-3, b=1.0, a=None, seed=1):
        if system.has_higher or system.terms:
            raise NotImplementedError("longitudinal spin dynamics supports bilinear exchange, SIA and field")
        self.sys = system
        self.T = float(temperature)
        self.lam = float(lam)
        self.dt = float(dt)
        ref = system.ref
        h = (system.Jsp @ ref.reshape(-1)).reshape(-1, 3)
        self.b = np.broadcast_to(np.asarray(b, float), (system.N,)).copy()
        self.a = (np.einsum("ia,ia->i", h, ref) - 2 * self.b) if a is None else np.broadcast_to(a, (system.N,)).copy()
        self.rng = np.random.default_rng(seed)
        self.time = 0.0

    def energy(self, M):
        s2 = np.einsum("ia,ia->i", M, M)
        return float(self.sys.energy(M) + np.sum(self.a * s2 + self.b * s2**2))

    def H(self, M):
        """-dE/dM (meV)."""
        s2 = np.einsum("ia,ia->i", M, M)[:, None]
        g = self.sys.gradient(M) + (2 * self.a[:, None] + 4 * self.b[:, None] * s2) * M
        return -g

    def rhs(self, M, xi):
        H = self.H(M)
        B = H / (self.sys.mu[:, None] * MU_B)
        return -GAMMA * np.cross(M, B) + self.lam * H + xi

    def step(self, M):
        xi = 0.0
        if self.T > 0:
            xi = np.sqrt(2 * self.lam * K_B * self.T / self.dt) * self.rng.normal(size=M.shape)
        k1 = self.rhs(M, xi)
        Mp = M + self.dt * k1
        k2 = self.rhs(Mp, xi)
        self.time += self.dt
        return M + 0.5 * self.dt * (k1 + k2)

    def run(self, M, steps, every=10):
        rows = []
        for n in range(steps + 1):
            if n % every == 0:
                s = np.linalg.norm(M, axis=1)
                rows.append([self.time, self.energy(M) / self.sys.N, s.mean(),
                             np.linalg.norm((self.sys.signs[:, None] * M).mean(0))])
            if n < steps:
                M = self.step(M)
        return M, np.array(rows)

"""Tests for the atomistic spin model, LLG, Monte Carlo and textures."""
import numpy as np
import pytest

from magex.llg import LLGSolver
from magex.models import write_model
from magex.montecarlo import MetropolisMC
from magex.spinmodel import GAMMA, K_B, MU_B, SpinSystem, jq_stability, load_results, mean_field_tc
from magex import texture


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


def _sys(tmp, L=(4, 4, 4), field=(0, 0, 0), **kw):
    p = tmp / "m.json"
    write_model(str(p), **kw)
    return SpinSystem(load_results(str(p)), L, field=field)


def test_gradient_matches_finite_difference(tmp):
    s = _sys(tmp, lattice="triangular", L=(5, 5, 1), J=[1.0, 0.2], D=0.3, K=0.1, field=(0.1, 0.2, 0.5))
    e = s.init_spins("random", seed=1)
    g = s.gradient(e)
    h = 1e-6
    for i, a in [(0, 0), (3, 2), (7, 1)]:
        ep, em = e.copy(), e.copy()
        ep[i, a] += h
        em[i, a] -= h
        fd = (s.energy(ep) - s.energy(em)) / (2 * h)
        assert abs(fd - g[i, a]) < 1e-6 * max(1, abs(fd))


def test_larmor_precession(tmp):
    s = _sys(tmp, lattice="sc", J=[0.0], L=(1, 1, 1), field=(0, 0, 2.0))
    e = np.array([[1.0, 0, 0]])
    sol = LLGSolver(s, alpha=0.0, dt=1e-3)
    ph = []
    for _ in range(1000):
        e = sol.step(e)
        ph.append(np.arctan2(e[0, 1], e[0, 0]))
    w = np.polyfit(np.arange(1, 1001) * 1e-3, np.unwrap(ph), 1)[0]
    assert abs(abs(w) - GAMMA * 2.0) / (GAMMA * 2.0) < 1e-6


def test_energy_conserved_without_damping(tmp):
    s = _sys(tmp, lattice="sc", J=[1.0], K=0.05, L=(4, 4, 4))
    e = s.init_spins("random", seed=3)
    sol = LLGSolver(s, alpha=0.0, dt=2e-4)
    E0 = s.energy(e)
    for _ in range(500):
        e = sol.step(e)
    assert abs(s.energy(e) - E0) / abs(E0) < 1e-4


def test_damping_reaches_ground_state(tmp):
    s = _sys(tmp, lattice="sc", J=[-1.0], L=(3, 3, 3), order="afm")
    e = s.init_spins("reference", cone=0.3, seed=2)
    sol = LLGSolver(s, alpha=0.5, dt=2e-3)
    e, tab, _ = sol.run(e, 4000, output_every=500)
    assert s.order_parameter(e) > 0.999


def test_monte_carlo_limits(tmp):
    s = _sys(tmp, lattice="sc", J=[1.0], L=(4, 4, 4))
    Tmf = mean_field_tc(s.uc)
    for T, lo, hi in ((0.2 * Tmf, 0.8, 1.01), (2.0 * Tmf, 0.0, 0.35)):
        mc = MetropolisMC(s, T, seed=1)
        e = s.init_spins("reference")
        ms = []
        for k in range(300):
            e = mc.sweep(e)
            if k > 100:
                ms.append(s.order_parameter(e))
        assert lo < np.mean(ms) < hi


def test_monte_carlo_langevin(tmp):
    """Free spins in a field: <m_z> = L(mu mu_B B / k_B T)."""
    s = _sys(tmp, lattice="sc", J=[0.0], L=(6, 6, 6), field=(0, 0, 5.0), moment=2.0)
    T = 5.0
    mc = MetropolisMC(s, T, seed=3, overrelax=0)
    e = s.init_spins("random", seed=1)
    acc = []
    for k in range(600):
        e = mc.sweep(e)
        if k > 100:
            acc.append(e[:, 2].mean())
    x = 2.0 * MU_B * 5.0 / (K_B * T)
    assert abs(np.mean(acc) - (1 / np.tanh(x) - 1 / x)) < 0.01


def test_jq_stability_ferromagnet(tmp):
    p = tmp / "fm.json"
    write_model(str(p), lattice="bcc", J=[1.0, 0.3])
    lam0, lmax, q = jq_stability(load_results(str(p)), nq=8)
    assert lmax <= lam0 + 1e-9


def test_topological_charge_of_skyrmion(tmp):
    s = _sys(tmp, lattice="triangular", J=[1.0], L=(30, 30, 1))
    c = s.positions[:, :2].mean(0)
    r = s.positions[:, :2] - c
    R = np.linalg.norm(r, axis=1)
    th = np.pi * np.exp(-(R / 12.0) ** 2)
    ph = np.arctan2(r[:, 1], r[:, 0])
    e = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
    assert abs(abs(texture.topological_charge(s, e)) - 1.0) < 0.02
    assert abs(texture.topological_charge(s, np.tile([0, 0, 1.0], (s.N, 1)))) < 1e-9

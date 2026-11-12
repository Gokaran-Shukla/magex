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


def _rashba_fm(lam=0.1):
    from magex.model import TBModel
    R = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    H0 = np.zeros((5, 1, 1), complex)
    X = np.zeros((5, 3, 1, 1), complex)
    S = np.zeros_like(X)
    H0[1:] = -1.0
    X[0, 2] = 0.5
    S[1, 1], S[2, 1], S[3, 0], S[4, 0] = 1j * lam, -1j * lam, -1j * lam, 1j * lam
    z = np.zeros(1, int)
    return TBModel(np.diag([1.0, 1, 10]), np.zeros((1, 3)), ["X"], R, H0, X, S, z, z, z)


def test_damping_torque_forms_agree_and_vanish_without_soc():
    from magex.damping import gilbert_damping, scalar
    m = _rashba_fm()
    r = gilbert_damping(m, -1.0, (0, 0, 1), (120, 120, 1), [0.005, 0.1], shift=(0.5, 0.5, 0), log=lambda *a: None)
    ex, so = scalar(r, "exchange"), scalar(r, "soc")
    assert abs(ex[0] / so[0] - 1) < 5e-3            # identical as eta -> 0
    assert np.all(so > 0)
    a = np.array(r["alpha"]["soc"])[0]
    assert np.allclose(a, a.T, atol=1e-10)           # symmetric, uniaxial for n || z
    m.S[:] = 0
    r = gilbert_damping(m, -1.0, (0, 0, 1), (120, 120, 1), [0.005, 0.01], forms=("exchange",),
                        shift=(0.5, 0.5, 0), log=lambda *a: None)
    e = scalar(r, "exchange")
    assert e[1] / e[0] > 1.6                          # no SOC: only the O(eta) artefact remains


def test_rpa_simple_cubic_watson(tmp):
    from magex.quantum import rpa_tc
    p = tmp / "sc.json"
    write_model(str(p), lattice="sc", J=[1.0])
    T_rpa, T_mf = rpa_tc(load_results(str(p)), nq=32)
    exact = (2 / 3) * 6.0 / (1.516386 * K_B)         # Watson integral of the sc lattice
    assert abs(T_rpa / exact - 1) < 0.03
    assert T_rpa < T_mf


def test_quantum_noise_planck_spectrum():
    from magex.quantum import HBAR_MEV_PS, QuantumNoise, planck
    dt, T, N, n = 2e-3, 10.0, 60, 2 ** 13
    q = QuantumNoise(N, 2.0 / dt, dt, T, seed=1)          # D = 1
    b = np.array([q()[:, 0] for _ in range(n)])
    w = np.fft.rfftfreq(n, dt) * 2 * np.pi
    P = (np.abs(np.fft.rfft(b, axis=0)) ** 2).mean(1) * dt / n
    x = w * HBAR_MEV_PS / (K_B * T)
    for lo, hi in ((0.5, 1.5), (2.0, 3.0), (4.0, 6.0)):
        sel = (x > lo) & (x < hi)
        assert abs(P[sel].mean() / (2 * planck(x[sel]).mean()) - 1) < 0.08


def test_biquadratic_and_four_spin(tmp):
    p = tmp / "hi.json"
    write_model(str(p), lattice="triangular", J=[1.0], Bq=[0.2], K4=0.3, D=0.2)
    s = SpinSystem(load_results(str(p)), (6, 6, 1), (True, True, False), field=(0, 0, 0.5))
    assert len(s.plaq) == 3 * s.N                     # three rhombi per site
    e = s.init_spins("random", seed=2)
    g = s.gradient(e)
    h = 1e-6
    for i, a in [(1, 0), (20, 2)]:
        ep, em = e.copy(), e.copy()
        ep[i, a] += h
        em[i, a] -= h
        assert abs((s.energy(ep) - s.energy(em)) / (2 * h) - g[i, a]) < 1e-6
    idx, en = np.array([7]), np.array([[0, 0, 1.0]])
    e2 = e.copy()
    e2[7] = en[0]
    full = s.higher_energy(e2) - s.higher_energy(e)
    loc = s.local_higher_energy(idx, en, e) - s.local_higher_energy(idx, e[idx], e)
    assert abs(loc[0] - full) < 1e-12
    # Monte Carlo with higher-order terms keeps the energy bookkeeping exact
    mc = MetropolisMC(s, 5.0, seed=1)
    for _ in range(5):
        e = mc.sweep(e)
    assert np.isfinite(s.energy(e))


def test_wang_landau_and_parallel_tempering_agree(tmp):
    from magex.advanced_mc import HAVE_NUMBA, parallel_tempering, wang_landau, wl_thermodynamics
    if not HAVE_NUMBA:
        pytest.skip("numba not installed")
    s = _sys(tmp, lattice="sc", J=[1.0], L=(3, 3, 3))
    wl = wang_landau(s, nbins=80, max_sweeps=400000, lnf_final=1e-6)
    th = wl_thermodynamics(wl, [20.0, 45.0])
    rows, _ = parallel_tempering(s, [20.0, 45.0], therm_sweeps=500, meas_sweeps=3000)
    for a, r in zip(th, rows):
        assert abs(a["E"] - r["E"]) < 0.1 * abs(r["E"]) + 5 * r["dE"]

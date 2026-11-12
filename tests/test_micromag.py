"""Tests for the micromagnetic solver."""
import numpy as np

from magex.micromag import Mesh, Micromagnet, minimize, read_ovf, write_ovf
from magex.micromag.demag import Demag, tensor_at
from magex.micromag.plot import topological_charge


def test_demag_sum_rules():
    N = tensor_at(np.zeros(1), np.zeros(1), np.zeros(1), (1e-9, 1e-9, 1e-9))[:, 0]
    assert np.allclose(N[:3], 1 / 3) and np.allclose(N[3:], 0)
    N = tensor_at(np.zeros(1), np.zeros(1), np.zeros(1), (5e-9, 3e-9, 1e-9))[:, 0]
    assert abs(N[:3].sum() - 1) < 1e-12
    mesh = Mesh((8, 8, 8), (1e-9,) * 3)
    M = np.zeros((8, 8, 8, 3))
    M[..., 2] = 1.0
    assert abs(-Demag(mesh).field(M)[..., 2].mean() - 1 / 3) < 1e-9


def test_thin_film_demag_periodic():
    mesh = Mesh((32, 32, 1), (2e-9, 2e-9, 1e-9), pbc=(True, True, False))
    D = Demag(mesh)
    M = np.zeros((32, 32, 1, 3))
    M[..., 2] = 1.0
    assert abs(D.field(M)[..., 2].mean() + 1) < 2e-3


def _wire(D=0.0, phi=0):
    mesh = Mesh((400, 1, 1), (0.5e-9, 1e-9, 1e-9))
    s = Micromagnet(mesh, demag=False)
    s.add_region(np.ones(mesh.n, bool), Ms=5.8e5, A=1.5e-11, Ku1=8e5, D=D)
    return s


def _wall_energy(D, phi):
    s = _wire(D)
    m, _, _ = minimize(s, s.init("domain_wall", width_nm=4, phi_deg=phi), tol=1e-6)
    u, _, _ = minimize(s, s.init("uniform"), tol=1e-6)
    return (s.energies(m)["total"] - s.energies(u)["total"]) / 1e-18


def test_bloch_wall_energy():
    assert abs(_wall_energy(0.0, 90) / (4 * np.sqrt(1.5e-11 * 8e5)) - 1) < 2e-3


def test_dmi_wall_energy():
    s = _wall_energy(1.5e-3, 180)
    assert abs(s / (4 * np.sqrt(1.5e-11 * 8e5) - np.pi * 1.5e-3) - 1) < 2e-3


def test_ovf_roundtrip(tmp_path):
    mesh = Mesh((5, 4, 3), (1e-9, 2e-9, 3e-9))
    m = np.random.default_rng(0).normal(size=(5, 4, 3, 3))
    for binary in (True, False):
        p = str(tmp_path / f"m{binary}.ovf")
        write_ovf(p, m, mesh, binary=binary)
        r, h = read_ovf(p)
        assert np.allclose(r, m, atol=1e-7)
        assert int(h["xnodes"]) == 5


def test_skyrmion_charge_and_relaxation():
    mesh = Mesh((48, 48, 1), (1e-9, 1e-9, 0.6e-9))
    s = Micromagnet(mesh, demag=False)
    s.add_region(np.ones(mesh.n, bool), Ms=5.8e5, A=1.5e-11, Ku1=8e5, D=3e-3)
    m = s.init("skyrmion", radius_nm=6)
    assert abs(abs(topological_charge(s, m)) - 1) < 0.05
    E0 = s.energies(m)["total"]
    m, _, _ = minimize(s, m, tol=1e-3, max_iter=3000)
    assert s.energies(m)["total"] < E0
    assert abs(abs(topological_charge(s, m)) - 1) < 0.05


def test_sot_damping_like_drives_towards_sigma():
    from magex.micromag.solver import llg_rhs
    mesh = Mesh((2, 2, 1), (2e-9, 2e-9, 1e-9))
    s = Micromagnet(mesh, demag=False)
    s.add_region(np.ones(mesh.n, bool), Ms=1e6, A=1e-11, alpha=0.0)
    s.sot = dict(J=1e12, theta=0.2, sigma=[0, 1, 0])
    m = s.init("uniform", direction=(0, 0, 1))
    dm, _ = llg_rhs(s, m, 0.0)
    assert np.all(dm[..., 1] > 0)


def test_bridge_spiral_energy(tmp_path):
    from magex.micromag.bridge import micromagnetic_parameters
    from magex.models import write_model
    from magex.spinmodel import load_results
    p = str(tmp_path / "tri.json")
    write_model(p, lattice="triangular", a=3.0, J=[2.0], D=1.0, moment=3.0)
    par = micromagnetic_parameters(p)
    uc = load_results(p)
    V = par["V_cell_A3"] * 1e-30

    def E(q):
        tot = 0.0
        for (i, j, R, J) in uc.couplings:
            r = uc.positions[j] + np.asarray(R) @ uc.cell - uc.positions[i]
            ph = q * r[0] * 1e-10
            tot -= np.array([0, 0, 1.0]) @ np.asarray(J) @ np.array([np.sin(ph), 0, np.cos(ph)])
        return tot * 1.602176634e-22
    for q in (2e8, -2e8):
        assert abs((E(q) - E(0)) / V / (par["A"] * q * q + par["D_int"] * q) - 1) < 2e-3


def test_spin_wave_dispersion_matches_analytic():
    from magex.spectra import analytic_film_dispersion, micromag_dispersion
    mesh = Mesh((64, 1, 1), (1e-9, 1e-9, 1e-9), pbc=(True, False, False))
    s = Micromagnet(mesh, demag=False)
    s.add_region(np.ones(mesh.n, bool), Ms=5.8e5, A=1.5e-11, Ku1=8e5, alpha=0.002)
    k, f, P, _ = micromag_dispersion(s, s.init("uniform"), time_ns=0.1, fmax_GHz=800, B_amp=1e-3)
    fa = analytic_film_dispersion(s, k)
    df = f[1] - f[0]
    for kk in (0.1, 0.3, 0.5):
        i = np.argmin(abs(k - kk))
        assert abs(f[np.argmax(P[i])] - fa[i]) < 1.5 * df


def test_sqw_peaks_follow_lswt(tmp_path):
    from magex.llg import LLGSolver
    from magex.models import write_model
    from magex.spectra import commensurate_path, lswt_bands, structure_factor
    from magex.spinmodel import SpinSystem, load_results
    p = str(tmp_path / "chain.json")
    write_model(p, lattice="chain", J=[1.0], K=0.05, moment=2.0)
    uc = load_results(p)
    sysm = SpinSystem(uc, (64, 1, 1), (True, False, False))
    q, _ = commensurate_path(["G", "X"], dict(G=[0, 0, 0], X=[0.5, 0, 0]), (64, 1, 1), 32)
    sol = LLGSolver(sysm, alpha=0.005, dt=2e-3, temperature=1.0, seed=1)
    e = sysm.init_spins("reference")
    for _ in range(2000):
        e = sol.step(e)
    E, Sp, _, _ = structure_factor(sysm, sol, e, q, 16384, 16, 2)
    bands = lswt_bands(uc, q)[:, 0]
    pos = E >= 0
    peak = E[pos][np.argmax(Sp[:, pos], axis=1)]
    dE = E[1] - E[0]
    assert np.median(np.abs(peak - bands)) < 2 * dE


def test_dmi_nonreciprocal_spin_waves():
    from magex.spectra import analytic_film_dispersion, micromag_dispersion
    mesh = Mesh((96, 1, 1), (2e-9, 2e-9, 1e-9), pbc=(True, False, False))
    s = Micromagnet(mesh, demag=False)
    s.add_region(np.ones(mesh.n, bool), Ms=8e5, A=1.3e-11, D=1.5e-3, alpha=0.002)
    s.set_field([0, 0.2, 0])
    m, _, _ = minimize(s, s.init("uniform", direction=(0, 1, 0)))
    k, f, P, _ = micromag_dispersion(s, m, time_ns=0.2, fmax_GHz=300)
    fa = analytic_film_dispersion(s, k)
    df = f[1] - f[0]
    for kk in (-0.3, 0.3):
        i = np.argmin(abs(k - kk))
        assert abs(f[np.argmax(P[i])] - fa[i]) < 2 * df


def test_gneb_stoner_wohlfarth_barrier():
    from magex.gneb import GNEB, MicromagAdapter, minimize as mn, slerp_path
    mesh = Mesh((1, 1, 1), (5e-9,) * 3)
    s = Micromagnet(mesh, demag=False)
    s.add_region(np.ones(mesh.n, bool), Ms=1e6, A=1e-11, Ku1=1e5)
    K, V, Ms = 1e5, 1.25e-25, 1e6
    for B in (0.0, 0.08):
        s.set_field([0, 0, B])
        ad = MicromagAdapter(s)
        a, _, _ = mn(ad, np.array([[0.01, 0, -1.0]]), 1e-30)
        b, _, _ = mn(ad, np.array([[0, 0, 1.0]]), 1e-30)
        r = GNEB(ad, slerp_path(a, b, 9)).run(0.0, 3000, rel_tol=1e-6).result()
        h = Ms * B / (2 * K)
        assert abs(r["barrier_forward"] / (K * V * (1 - h) ** 2) - 1) < 1e-3


def test_afmr_frequency_and_altermagnet_splitting():
    from magex.micromag.afm import TwoSublatticeMagnet, afm_analytic, afm_dispersion
    from magex.micromag.constants import GAMMA_LL
    from magex.micromag.solver import Heun
    mesh = Mesh((1, 1, 1), (2e-9,) * 3)
    s = TwoSublatticeMagnet(mesh, demag=False)
    Ms, K, J = 4e5, 1e5, 4e7
    s.add_region(np.ones(mesh.n, bool), Ms=Ms, A=1e-11, Ku1=K, alpha=0.0, J_AF=J)
    m = s.init("neel")
    m[0, ..., 0] += 0.02
    m = s.normalize(m)
    tr = []
    Heun(s, dt=2e-15).run(m, 10e-12, 5e-15, lambda t, mm: tr.append(mm[0, 0, 0, 0, 0]))
    z = np.where(np.diff(np.sign(tr)))[0]
    f = (len(z) - 1) / 2 / ((z[-1] - z[0]) * 5e-15)
    BK, BE = 2 * K / Ms, J / Ms
    assert abs(f / (GAMMA_LL / (2 * np.pi) * np.sqrt(BK * (2 * BE + BK))) - 1) < 2e-3
    mesh = Mesh((64, 1, 1), (1e-9,) * 3, pbc=(True, False, False))
    s = TwoSublatticeMagnet(mesh, demag=False)
    s.add_region(np.ones(mesh.n, bool), Ms=Ms, A=1e-11, Ku1=K, alpha=0.001, J_AF=J)
    s.set_altermagnet(0.3)
    k, f, P = afm_dispersion(s, s.init("neel"), time_ns=0.05, fmax_GHz=3000)
    fp, fm = afm_analytic(s, k)
    df = f[1] - f[0]
    i = np.argmin(abs(k - 0.4))
    pos, neg = f > 0, f < 0
    assert abs(f[pos][np.argmax(P[i, pos])] - fp[i]) < 2 * df
    assert abs(-f[neg][np.argmax(P[i, neg])] - fm[i]) < 2 * df
    assert fp[i] - fm[i] > 5 * df                  # chirality splitting resolved


def test_zhang_li_wall_velocity_and_slonczewski_switching():
    from magex.micromag.solver import RK45
    from magex.micromag.stt import zhang_li_u
    mesh = Mesh((300, 1, 1), (1e-9,) * 3)
    s = Micromagnet(mesh, demag=False)
    s.add_region(np.ones(mesh.n, bool), Ms=5.8e5, A=1.5e-11, Ku1=8e5, alpha=0.1)
    m, _, _ = minimize(s, s.init("domain_wall", width_nm=4, phi_deg=90, x0_nm=-80), 1e-6)
    s.stt = dict(type="zhang-li", J=[1e12, 0, 0], P=0.5, beta=0.1)
    u = zhang_li_u(s, [1e12, 0, 0], 0.5, 0.1)[0, 0, 0, 0]
    X = mesh.coords()[0][:, 0, 0]

    def pos(mm):
        mz = mm[:, 0, 0, 2]
        i = np.where(np.diff(np.sign(mz)))[0][0]
        return X[i] - mz[i] * (X[i + 1] - X[i]) / (mz[i + 1] - mz[i])
    ts, xs = [], []
    RK45(s).run(m, 0.3e-9, 10e-12, lambda t, mm: (ts.append(t), xs.append(pos(mm))))
    ts, xs = np.array(ts), np.array(xs)
    v = np.polyfit(ts[ts > 0.05e-9], xs[ts > 0.05e-9], 1)[0]
    assert abs(v / u - 1) < 0.03                     # beta = alpha: v = u
    # Slonczewski: macrospin PMA free layer, polariser +z, start near -z; switches only above J_c
    from magex.micromag.constants import E_CHARGE, HBAR
    mesh = Mesh((1, 1, 1), (5e-9, 5e-9, 1e-9))
    Ms, K, a, P = 1e6, 8e5, 0.02, 0.5
    BK = 2 * K / Ms
    Jc = 2 * E_CHARGE * a * Ms * 1e-9 * BK / (HBAR * P / 2 * 1)   # eps(m.p = -1) = P/2 for Lambda = 1
    for J, flips in ((0.7 * Jc, False), (1.5 * Jc, True)):
        s = Micromagnet(mesh, demag=False)
        s.add_region(np.ones(mesh.n, bool), Ms=Ms, A=1e-11, Ku1=K, alpha=a)
        s.stt = dict(type="slonczewski", J=J, P=P, Lambda=1.0, p=[0, 0, 1], thickness=1e-9)
        m = s.init("uniform", direction=(0.05, 0, -1))
        m, _ = RK45(s).run(m, 3e-9, 0.1e-9)
        assert (m[0, 0, 0, 2] > 0) == flips


def test_first_principles_sot_and_spin_pumping():
    from magex.micromag.fp_torques import (pumped_spin_current, spin_pumping_damping,
                                           theta_from_conductivities)
    assert abs(theta_from_conductivities(5e4, sigma_SH=2000) - 0.08) < 1e-12
    assert theta_from_conductivities(5e4, 2000, nm_thickness_nm=1.0, lambda_s_nm=1.5) < 0.08
    assert abs(spin_pumping_damping(2.5e19, 8e5, 5e-9) - 9.225e-3) < 1e-5
    # uniform precession about x at angle th: DC j_s = (hbar/4pi) g w sin^2(th) along x
    w, th, g = 2 * np.pi * 1e10, 0.1, 1e19
    t = np.linspace(0, 2e-9, 4001)
    m = np.stack([np.cos(th) * np.ones_like(t), np.sin(th) * np.cos(w * t), np.sin(th) * np.sin(w * t)], 1)
    jdc, _ = pumped_spin_current(np.column_stack([t, m]), g)
    from magex.micromag.constants import HBAR
    assert abs(jdc[0] / (HBAR / (4 * np.pi) * g * w * np.sin(th) ** 2) - 1) < 1e-3


def test_llb_mean_field_and_relaxation():
    from magex.micromag.llb import LLB, langevin, m_eq, scaled_parameters
    for T in (100.0, 400.0, 600.0):
        me = m_eq(T, 630.0)
        assert abs(me - float(langevin(3 * 630.0 * me / T))) < 1e-10
    assert m_eq(700.0, 630.0) == 0.0
    sc = scaled_parameters(dict(Ms=1e6, A=1e-11, Ku1=1e5), 300.0, 630.0)
    assert abs(sc["Ku1"] / 1e5 - sc["m_e"] ** 3) < 1e-12
    mesh = Mesh((1, 1, 1), (2e-9,) * 3)
    s = Micromagnet(mesh, demag=False)
    s.add_region(np.ones(mesh.n, bool), Ms=4.9e5, A=9e-12, Ku1=1e4)
    llb = LLB(s, Tc=630.0, lam=0.05, mu_s=0.6)
    llb.set_temperature(400.0)
    m = np.zeros(mesh.n + (3,))
    m[..., 2] = 1.0
    m, _ = llb.run(m, 3e-12, 2e-16)
    assert abs(np.linalg.norm(m[0, 0, 0]) - m_eq(400.0, 630.0)) < 1e-4

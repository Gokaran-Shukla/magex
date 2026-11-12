"""Tests of the VAMPIRE / UppASD / Spirit / mumax3-equivalent features (v0.4)."""
import numpy as np
import pytest

from magex.models import write_model
from magex.spinmodel import GAMMA, MU_B, SpinSystem, load_results, normalize


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


def _uc(tmp, name="m.json", **kw):
    p = tmp / name
    write_model(str(p), **kw)
    return load_results(str(p))


def _fd_err(s, e, sites, h=1e-6):
    g = s.gradient(e)
    err = 0.0
    for i in sites:
        for a in range(3):
            ep, em = e.copy(), e.copy()
            ep[i, a] += h
            em[i, a] -= h
            err = max(err, abs((s.energy(ep) - s.energy(em)) / (2 * h) - g[i, a]))
    return err


# ----------------------------------------------------------------------------- structure + terms
def test_structure_terms_gradients_and_mc_local_energy(tmp):
    from magex.montecarlo import MetropolisMC
    from magex.structure import apply_structure
    s = SpinSystem(_uc(tmp, lattice="sc", J=[1.0], K=0.05), (8, 8, 8), (False, False, False))
    apply_structure(s, dict(shape="sphere", size=10.0, core_shell=dict(radius=6.0, core="Fe", shell="Co"),
                            materials=[dict(name="Fe", Kc=0.02, K2=0.01), dict(name="Co", moment_scale=0.8, ks=0.1)],
                            exchange_scale=[[1, 1, 1], [1, 1, 0.5], [1, 0.5, 1]], dipolar=True,
                            quadruplets=dict(K=0.03, list=[[0, 1, 2, 3]]), vacancies=0.05))
    assert 100 < s.N < 512 and len(s.terms) == 4
    rng = np.random.default_rng(0)
    e = normalize(rng.normal(size=(s.N, 3)))
    assert _fd_err(s, e, [0, s.N // 3, s.N - 1]) < 1e-6
    mc = MetropolisMC(s, 50.0, 1)
    mc._e = e
    for t in s.terms:
        if t.long_range:
            t.refresh(e)
    for i in (5, 40):
        en = normalize(rng.normal(size=(1, 3)))
        e2 = e.copy()
        e2[i] = en[0]
        true = sum(t.energy(e2) - t.energy(e) for t in s.terms)
        loc = sum((t.local_energy(np.array([i]), en, e) - t.local_energy(np.array([i]), e[i:i + 1], e))[0] for t in s.terms)
        assert abs(true - loc) < 1e-10


def test_dipolar_direct_sum_and_thin_film_shape_anisotropy(tmp):
    from magex.terms import DIP, Dipolar
    uc = _uc(tmp, lattice="sc", J=[1.0])
    s = SpinSystem(uc, (4, 3, 2), (False, False, False))
    d = Dipolar(s)
    e = normalize(np.random.default_rng(1).normal(size=(s.N, 3)))
    P = s.positions
    B = np.zeros((s.N, 3))
    for i in range(s.N):
        for j in range(s.N):
            if i != j:
                R = P[j] - P[i]
                r = np.linalg.norm(R)
                B[i] += DIP * s.mu[j] * (3 * R * (R @ e[j]) / r**2 - e[j]) / r**3
    assert np.abs(B - d.field(e)).max() < 1e-12
    s = SpinSystem(uc, (16, 16, 4), (True, True, False))
    d = Dipolar(s, images=6)
    dE = (d.energy(np.tile([0, 0, 1.0], (s.N, 1))) - d.energy(np.tile([1.0, 0, 0], (s.N, 1)))) / s.N
    cont = 2 * np.pi * DIP * 2.0**2 * MU_B / 27.0          # mu0 Ms^2 V / 2
    assert abs(dE / cont - 1) < 0.02


def test_textures_charge(tmp):
    from magex import texture
    s = SpinSystem(_uc(tmp, lattice="triangular", J=[1.0], D=0.3), (40, 40, 1), (True, True, False), field=(0, 0, 1.0))
    Q = lambda e: texture.topological_charge(s, e, "xy", 0)
    assert abs(abs(Q(s.init_spins("skyrmion", radius=8.0))) - 1) < 1e-6
    assert abs(abs(Q(s.init_spins("skyrmions", centers=[[30, 30], [80, 60]], radius=6.0))) - 2) < 1e-6
    q1 = Q(s.init_spins("skyrmion", radius=8.0, helicity=0.0))
    q2 = Q(s.init_spins("skyrmion", radius=8.0, helicity=0.0, vorticity=-1))
    assert abs(q1 + q2) < 1e-6
    assert np.linalg.norm(s.init_spins("spiral", q_frac=[0.1, 0, 0]).mean(0)) < 1e-6


def test_pinning_and_temperature_profile(tmp):
    from magex.montecarlo import MetropolisMC
    from magex.structure import apply_structure
    s = SpinSystem(_uc(tmp, lattice="sc", J=[1.0]), (4, 4, 12), (True, True, False))
    apply_structure(s, dict(pinned=[dict(region="bottom", direction=[1, 0, 0])],
                            temperature_profile=dict(axis="z", T0=1.0, T1=200.0)))
    e = s.init_spins("ferro", (0, 0, 1))
    mc = MetropolisMC(s, 100.0, 3)
    for _ in range(200):
        e = mc.sweep(e)
    cz = s.cells[s.cell_index][:, 2]
    assert np.allclose(e[cz == 0], [1, 0, 0])
    assert np.linalg.norm(e[cz == 1].mean(0)) > np.linalg.norm(e[cz == 11].mean(0)) + 0.3


# ----------------------------------------------------------------------------- dynamics
@pytest.mark.parametrize("method", ["heun", "rk4", "depondt", "sib", "euler"])
def test_integrators_larmor(tmp, method):
    from magex.llg import LLGSolver
    s = SpinSystem(_uc(tmp, lattice="sc", J=[0.0]), (1, 1, 1), (False, False, False), field=(0, 0, 1.0))
    e = normalize(np.array([[1.0, 0, 0.3]]))
    sv = LLGSolver(s, 0.0, 1e-3, integrator=method)
    for _ in range(1000):
        e = sv.step(e)
    assert abs(np.arctan2(e[0, 1], e[0, 0]) - GAMMA * 1.0) < 1e-4
    assert abs(e[0, 2] - 0.3 / np.sqrt(1.09)) < 1e-4


def test_zhang_li_wall_velocity(tmp):
    from magex.llg import LLGSolver
    from magex.structure import apply_structure
    s = SpinSystem(_uc(tmp, lattice="chain", J=[5.0], K=0.5), (120, 1, 1), (False, False, False))
    apply_structure(s, dict(materials=[dict(name="bulk", Ku=[-0.1, [0, 1, 0]])]))
    e = s.init_spins("domain", normal="x", position=180.0, width=5.0)
    sv = LLGSolver(s, 0.5, 1e-3)
    for _ in range(2000):
        e = sv.step(e)
    wall = lambda e: np.interp(0, -e[:, 2], s.positions[:, 0])
    sv = LLGSolver(s, 0.1, 4e-3, stt=dict(u=[40.0, 0, 0], beta=0.2))
    ts, xs = [], []
    for n in range(40000):
        e = sv.step(e)
        if n % 2000 == 0:
            ts.append(sv.time)
            xs.append(wall(e))
    v = np.polyfit(ts[10:], xs[10:], 1)[0] * 100
    assert abs(v / 80.0 - 1) < 0.05


@pytest.mark.parametrize("method", ["lbfgs", "ncg", "vp"])
def test_minimizers_agree(tmp, method):
    from magex import texture
    from magex.minimize import minimize
    s = SpinSystem(_uc(tmp, lattice="triangular", J=[1.0], D=0.6), (24, 24, 1), (True, True, False), field=(0, 0, 6.0))
    e, info = minimize(s, s.init_spins("skyrmion", radius=5.0), method, tol=1e-7)
    ref, _ = minimize(s, s.init_spins("skyrmion", radius=5.0), "lbfgs", tol=1e-8)
    assert info["max_torque"] < 1e-7
    assert abs(s.energy(e) - s.energy(ref)) < 1e-6
    assert abs(abs(texture.topological_charge(s, e, "xy", 0)) - 1) < 1e-3


# ----------------------------------------------------------------------------- Monte Carlo
def test_mc_moves_agree(tmp):
    from magex.montecarlo import MetropolisMC
    uc = _uc(tmp, lattice="sc", J=[1.0], K=0.3)
    out = {}
    for mv in ("gaussian", "heatbath"):
        s = SpinSystem(uc, (6, 6, 6), field=(0, 0, 0.5))
        mc = MetropolisMC(s, 20.0, 1, 0.6, 0, mv)
        e = s.init_spins("ferro")
        Es = []
        for n in range(1500):
            e = mc.sweep(e)
            if n > 300 and n % 5 == 0:
                Es.append(s.energy(e, True))
        out[mv] = np.mean(Es)
    assert abs(out["gaussian"] - out["heatbath"]) < 0.04


def test_constrained_mc_callen_callen(tmp):
    from magex.cmc import anisotropy_vs_T
    s = SpinSystem(_uc(tmp, lattice="sc", J=[1.0], K=0.1), (6, 6, 6))
    R = anisotropy_vs_T(s, [1.0, 10.0], therm=300, meas=800)
    assert abs(R[0, 2] / 0.1 - R[0, 1] ** 3) < 0.02
    assert abs(R[1, 2] / 0.1 - R[1, 1] ** 3) < 0.03


def test_lsf_low_temperature(tmp):
    from magex.montecarlo import LSFMonteCarlo
    s = SpinSystem(_uc(tmp, lattice="sc", J=[1.0]), (4, 4, 4))
    mc = LSFMonteCarlo(s, 1.0, 1, 0.1, b=2.0)
    M = s.ref.copy()
    for _ in range(300):
        M = mc.sweep(M)
    assert abs(np.linalg.norm(M, axis=1).mean() - 1) < 0.02


# ----------------------------------------------------------------------------- harmonic analysis
def test_htst_and_eigenmodes(tmp):
    from magex.htst import htst_rate, llg_frequencies, mmf
    from magex.structure import apply_structure
    s = SpinSystem(_uc(tmp, lattice="sc", J=[0.0], K=0.5), (1, 1, 1), (False, False, False))
    apply_structure(s, dict(materials=[dict(name="bulk", Ku=[-0.2, [0, 1, 0]])]))
    f = llg_frequencies(s, np.array([[0, 0, 1.0]]))
    fan = GAMMA / (2 * MU_B) * np.sqrt(2 * 0.5 * (2 * 0.5 + 2 * 0.2)) / (2 * np.pi) * 1e3
    assert abs(f[0] / fan - 1) < 1e-9
    r = htst_rate(s, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]), 50.0)
    assert abs(r["prefactor"] / 1e9 / fan - 1) < 1e-9 and abs(r["dE"] - 0.5) < 1e-12
    e, info = mmf(s, normalize(np.array([[0.01, 0, 1.0]])))
    assert abs(abs(e[0, 0]) - 1) < 1e-5 and info["lambda_1"] < 0
    s3 = SpinSystem(_uc(tmp, "ring.json", lattice="chain", J=[1.0], K=0.1), (8, 1, 1))
    f = llg_frequencies(s3, np.tile([0, 0, 1.0], (8, 1)))
    k = 2 * np.pi * np.arange(8) / 8
    fan = np.sort(GAMMA / (2 * MU_B) * (4 * (1 - np.cos(k)) + 0.2) / (2 * np.pi) * 1e3)
    assert np.allclose(f, fan, rtol=1e-8)


def test_spin_lattice_energy_conservation(tmp):
    from magex.sld import SpinLattice
    s = SpinSystem(_uc(tmp, lattice="sc", J=[1.0]), (3, 3, 3))
    rng = np.random.default_rng(0)
    drift = []
    for dt in (2e-3, 1e-3):
        sl = SpinLattice(s, lam=0.5, k_spring=500.0, dt=dt, integrator="depondt")
        sl.v = rng.normal(size=sl.v.shape) * 2.0
        e = normalize(s.ref + 0.4 * rng.normal(size=s.ref.shape))
        e, tab = sl.run(e, int(1.0 / dt), every=int(0.05 / dt))
        drift.append(np.ptp(tab[:, 1]))
        assert np.ptp(tab[:, 5]) > 10 * drift[-1]          # energy really flows between spins and lattice
    assert drift[1] < 0.6 * drift[0]


def test_observables_and_writers(tmp):
    from magex.observables import (chirality, correlation_function, resolved_magnetisation, sq_grid,
                                   static_structure_factor, write_vtk, write_xyz)
    s = SpinSystem(_uc(tmp, lattice="triangular", J=[1.0]), (12, 12, 1), (True, True, False))
    e = s.init_spins("spiral", q_frac=[1 / 3, 1 / 3, 0])          # 120-degree state
    chi, vec = chirality(s, e)
    assert len(chi) == 2 * 144
    qf, S = sq_grid(s, e)
    k = np.unravel_index(np.argmax(S), S.shape)
    assert np.allclose(np.abs(qf[k][:2]), [1 / 3, 1 / 3], atol=1e-9)
    Sq = static_structure_factor(s, e, (2 * np.pi * qf[k] @ np.linalg.inv(s.uc.cell).T)[None])
    assert abs(Sq[0] - S[k]) < 1e-6 * S[k]
    r, C, n = correlation_function(s, s.init_spins("ferro"))
    assert np.allclose(C, 1)
    assert set(resolved_magnetisation(s, e)) == {0}
    write_vtk(str(tmp / "a.vtk"), s, e)
    write_xyz(str(tmp / "a.xyz"), s, e)
    assert open(tmp / "a.xyz").readline().strip() == str(s.N)


# ----------------------------------------------------------------------------- micromagnetics (mumax3 set)
def test_micromag_new_terms_energy_consistency():
    from magex.micromag.custom import custom_energy
    from magex.micromag.mesh import Mesh, geometry_mask
    from magex.micromag.system import Micromagnet
    mesh = Mesh((10, 8, 2), (2e-9, 2e-9, 2e-9), (True, True, False))
    s = Micromagnet(mesh, demag=True)
    s.add_region(geometry_mask(mesh, "rectangle"), "a", Ms=8e5, A=1e-11, D=2e-3, Kc1=3e4, Kc2=2e4, Kc3=1e4, B1=-1e7, B2=5e6)
    s.add_region(geometry_mask(mesh, "rectangle", size_nm=[8, 30]), "b", Ms=6e5, A=1e-11, D=-1e-3, Ku1=1e5)
    s.inter_dmi[(0, 1)] = 3e-3
    s.strain = np.array([1e-3, -2e-3, 5e-4, 1e-3, -1e-3, 2e-4])
    s.custom_energies.append(custom_energy(dict(density="-K4*(mx**4+my**4+mz**4)", constants=dict(K4=2e4)), mesh))
    m = s.normalize(np.random.default_rng(0).normal(size=mesh.n + (3,)))
    F = s.fields(m)
    h = 1e-6
    for term in ("exchange", "dmi", "cubic", "cubic2", "cubic3", "magnetoelastic", "custom", "demag", "anisotropy"):
        for idx in [(3, 4, 1), (6, 5, 0), (0, 0, 1)]:
            for a in range(3):
                mp, mm = m.copy(), m.copy()
                mp[idx + (a,)] += h
                mm[idx + (a,)] -= h
                dE = (s.energies(mp)[term] - s.energies(mm)[term]) / (2 * h)
                pred = -s.Ms[idx] * mesh.V * F[term][idx + (a,)]
                tol = 1e-4 if term == "demag" else 1e-6          # FFT round-off in the demag energy
                assert abs(dE - pred) <= tol * max(abs(pred), 1e-30) + 1e-30, term


@pytest.mark.parametrize("method", ["rk45", "rk23", "rk4", "heun", "rk2", "euler"])
def test_micromag_solvers_precession(method):
    from magex.micromag.constants import GAMMA_LL
    from magex.micromag.mesh import Mesh
    from magex.micromag.solver import make_solver
    from magex.micromag.system import Micromagnet
    mesh = Mesh((1, 1, 1), (5e-9,) * 3)
    s = Micromagnet(mesh, demag=False)
    s.add_region(np.ones((1, 1, 1), bool), "x", Ms=8e5, alpha=0.0)
    s.set_field([0, 0, 0.1])
    m0 = np.zeros((1, 1, 1, 3))
    m0[..., 0] = 1
    m, _ = make_solver(s, method, 1e-7, dt=1e-13).run(m0, 1e-10, 1e-11)
    exp = (GAMMA_LL * 0.1 * 1e-10 + np.pi) % (2 * np.pi) - np.pi
    assert abs(np.arctan2(m[0, 0, 0, 1], m[0, 0, 0, 0]) - exp) < 1e-4


def test_csg_volume_fraction_and_moving_frame():
    from magex.micromag.constants import GAMMA_LL
    from magex.micromag.custom import MovingFrame, custom_energy
    from magex.micromag.mesh import Mesh, geometry_mask, volume_fraction
    from magex.micromag.solver import RK45, relax
    from magex.micromag.system import Micromagnet
    f = volume_fraction(Mesh((20, 20, 20), (1e-9,) * 3), "sphere", 4, diameter_nm=14)
    assert abs(f.sum() / (np.pi / 6 * 14**3) - 1) < 0.005
    u = geometry_mask(Mesh((20, 20, 4), (1e-9,) * 3), "union", shapes=[dict(shape="disk", diameter_nm=10, center_nm=[-5, 0]),
                                                                    dict(shape="disk", diameter_nm=10, center_nm=[5, 0])])
    i = geometry_mask(Mesh((20, 20, 4), (1e-9,) * 3), "intersect", shapes=[dict(shape="disk", diameter_nm=10, center_nm=[-5, 0]),
                                                                        dict(shape="disk", diameter_nm=10, center_nm=[5, 0])])
    d = geometry_mask(Mesh((20, 20, 4), (1e-9,) * 3), "disk", diameter_nm=10)
    assert u.sum() + i.sum() == 2 * d.sum()
    mesh = Mesh((80, 1, 1), (2e-9,) * 3)
    s = Micromagnet(mesh, demag=False)
    s.add_region(np.ones(mesh.n, bool), "w", Ms=8e5, A=1.3e-11, Ku1=5e5, alpha=0.5)
    s.custom_energies.append(custom_energy(dict(density="Kh*my**2", constants=dict(Kh=1e5)), mesh))
    X, _, _ = mesh.coords()
    th = 2 * np.arctan(np.exp(-X / 5e-9))
    m = relax(s, np.stack([np.sin(th), 0 * th, np.cos(th)], -1), 1e-4)
    s.set_field([0, 0, 0.02])
    fr = MovingFrame(s, "z", "x")
    m, _ = RK45(s, tol=1e-6).run(m, 1.5e-9, 5e-12, fr)
    P = np.array(fr.positions)
    sel = P[:, 0] > 0.5e-9
    v = np.polyfit(P[sel, 0], P[sel, 1], 1)[0]
    assert fr.shift < -15 and abs(abs(v) / (GAMMA_LL * np.sqrt(1.3e-11 / 5e5) * 0.02 / 0.5) - 1) < 0.05


def test_exporters_translate_structure_and_programs(tmp):
    from magex.interop import export_mumax3, export_spirit, export_uppasd, export_vampire
    p = tmp / "m.json"
    write_model(str(p), lattice="sc", J=[1.0], K=0.1)
    cfg = dict(structure=dict(shape="sphere", size=20.0, dipolar=True),
               program=dict(type="hysteresis", Bmax=2.0, npts=21, direction=[0, 0, 1]),
               llg=dict(alpha=0.1, dt=1e-3, steps=100, field=[0, 0, 1.0], stt=dict(u=[10, 0, 0], beta=0.1)))
    v = open(export_vampire(str(p), str(tmp / "v"), cfg=cfg) + "/input").read()
    assert "create:sphere" in v and "dimensions:particle-size = 4.0000 !nm" in v
    assert "dipole:solver = tensor" in v and "sim:program = hysteresis-loop" in v
    u = open(export_uppasd(str(p), str(tmp / "u"), cfg=cfg) + "/inpsd.dat.template").read()
    assert "do_dip 1" in u and "stt A" in u
    sp = open(export_spirit(str(p), str(tmp / "s"), cfg=cfg) + "/input.cfg").read()
    assert "ddi_method fft" in sp and "external_field_magnitude 1.0" in sp
    mm = dict(mesh=dict(n=[32, 32, 2], cell_nm=[2, 2, 1]),
              region=[dict(Ms=8e5, A=1e-11, Kc2=1e3, shape="union",
                           shapes=[dict(shape="disk", diameter_nm=20, center_nm=[-5, 0]),
                                   dict(shape="sphere", diameter_nm=10)]),
                      dict(Ms=6e5, A=1e-11, shape="cuboid", size_nm=[4, 4, 1], frozen=True)],
              inter_dmi=[dict(regions=[0, 1], D=1e-3)],
              stage=[dict(type="run", time_ns=0.1, solver="rk23", moving_frame=True, autosave_ps=10)])
    t = open(export_mumax3(mm, str(tmp / "a.mx3"))).read()
    for key in ("Circle(20e-9).Transl(-5e-9, 0e-9, 0e-9).Add(Ball(10e-9))", "Kc2.SetRegion(0, 1000)",
                "FrozenSpins.SetRegion(1, 1)", "ext_InterDind(0, 1, 0.001)", "SetSolver(3)", "ext_centerWall(2)",
                "AutoSave(m, 10e-12)", "Cuboid(4e-9, 4e-9, 1e-9)"):
        assert key in t, key

"""Fast kernels, exactness upgrades and large-system harmonic analysis (v0.4)."""
import numpy as np
import pytest

from magex.models import write_model
from magex.spinmodel import K_B, MU_B, SpinSystem, load_results, normalize


def _uc(tmp, name="m.json", **kw):
    p = tmp / name
    write_model(str(p), **kw)
    return load_results(str(p))


@pytest.mark.parametrize("integrator", ["heun", "depondt"])
def test_fast_llg_matches_numpy(tmp_path, integrator):
    from magex.llg import LLGSolver
    from magex.structure import apply_structure
    s = SpinSystem(_uc(tmp_path, lattice="sc", J=[1.0], D=0.1, K=0.05), (6, 6, 6), field=(0.1, 0, 0.5))
    apply_structure(s, dict(materials=[dict(name="bulk", Kc=0.02, K2=0.01)], dipolar=True))
    e0 = s.init_spins("random", seed=1)
    a = LLGSolver(s, 0.1, 1e-3, integrator=integrator, backend="numpy")
    b = LLGSolver(s, 0.1, 1e-3, integrator=integrator)
    assert b.fast is not None
    ea, eb = e0.copy(), e0.copy()
    for _ in range(30):
        ea, eb = a.step(ea), b.step(eb)
    assert np.abs(ea - eb).max() < 1e-12
    s2 = SpinSystem(_uc(tmp_path, "b.json", lattice="sc", J=[1.0], D=0.2, K=0.05), (40, 40, 2), field=(0, 0, 0.3))
    e0 = s2.init_spins("random", seed=2)                    # parallel fused path, DMI storage
    a = LLGSolver(s2, 0.2, 1e-3, integrator=integrator, backend="numpy")
    b = LLGSolver(s2, 0.2, 1e-3, integrator=integrator)
    ea, eb = e0.copy(), e0.copy()
    for _ in range(20):
        ea, eb = a.step(ea), b.step(eb)
    assert np.abs(ea - eb).max() < 1e-12


def test_ziggurat_normal_moments():
    from magex import fast
    b = np.empty((300000, 3))
    fast._gauss3(b, fast.seed_streams(7), np.ones(len(b)), np.zeros(len(b), bool))
    x = b.ravel()
    assert abs(x.mean()) < 0.006 and abs(x.var() - 1) < 0.006
    assert abs((x**4).mean() / x.var() ** 2 - 3) < 0.03
    assert abs((np.abs(x) > 3).mean() / 0.0026998 - 1) < 0.08


@pytest.mark.parametrize("move", ["gaussian", "heatbath", "uniform"])
def test_fast_mc_exact_single_spin(tmp_path, move):
    from magex.montecarlo import MetropolisMC
    s = SpinSystem(_uc(tmp_path, lattice="sc", J=[0.0], K=1.0), (16, 16, 16), (False, False, False), field=(0.2, 0, 2.0))
    T = 10.0
    th = np.linspace(0, np.pi, 601)
    ph = np.linspace(0, 2 * np.pi, 601)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    E3 = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1)
    En = np.einsum("...a,ab,...b->...", E3, s.A[0], E3) - 2 * MU_B * (E3 @ np.array([0.2, 0, 2.0]))
    w = np.exp(-(En - En.min()) / (K_B * T)) * np.sin(TH)
    I = lambda f: np.trapezoid(np.trapezoid(f, ph, axis=1), th)
    mz = I(w * E3[..., 2]) / I(w)
    mc = MetropolisMC(s, T, 3, 0.6, 1, move)
    assert mc.fast is not None
    e = s.init_spins("ferro")
    ms = []
    for n in range(400):
        e = mc.sweep(e)
        if n > 50:
            ms.append(e[:, 2].mean())
    assert abs(np.mean(ms) - mz) < 4 * np.std(ms) / np.sqrt(len(ms) / 3) + 1e-3


def test_dipolar_two_sublattices_and_exact_mc(tmp_path):
    import itertools
    from magex.montecarlo import MetropolisMC
    from magex.terms import DIP, Dipolar
    uc = _uc(tmp_path, lattice="sc", J=[-1.0], order="afm")
    s = SpinSystem(uc, (3, 3, 2), (False, False, False))
    d = Dipolar(s)
    e = normalize(np.random.default_rng(1).normal(size=(s.N, 3)))
    P = s.positions
    B = np.zeros((s.N, 3))
    for i, j in itertools.product(range(s.N), range(s.N)):
        if i != j:
            R = P[j] - P[i]
            r = np.linalg.norm(R)
            B[i] += DIP * s.mu[j] * (3 * R * (R @ e[j]) / r**2 - e[j]) / r**3
    assert np.abs(B - d.field(e)).max() < 1e-12
    # exact sequential dipolar MC agrees with brute-force Metropolis on full energies (periodic film)
    uc2 = _uc(tmp_path, "w.json", lattice="sc", a=2.0, J=[0.02], moment=3.0)

    def mk():
        s_ = SpinSystem(uc2, (3, 3, 2), (True, True, False))
        s_.terms.append(Dipolar(s_, images=2))
        return s_
    T = 0.5
    s1 = mk()
    rng = np.random.default_rng(3)
    e = s1.init_spins("random", seed=2)
    E = s1.energy(e)
    ref = []
    for sw in range(1200):
        for i in range(s1.N):
            en = e.copy()
            en[i] = normalize((e[i] + 0.6 * rng.normal(size=3))[None])[0]
            En_ = s1.energy(en)
            if En_ <= E or rng.random() < np.exp(-(En_ - E) / (K_B * T)):
                e, E = en, En_
        if sw > 200:
            ref.append(E / s1.N)
    s2 = mk()
    mc = MetropolisMC(s2, T, 5, 0.6, 0)
    assert mc.dip is not None
    e = s2.init_spins("random", seed=2)
    out = []
    for sw in range(5000):
        e = mc.sweep(e)
        if sw > 300:
            out.append(s2.energy(e, True))
    err = np.hypot(np.std(ref) / np.sqrt(len(ref) / 20), np.std(out) / np.sqrt(len(out) / 20))
    assert abs(np.mean(ref) - np.mean(out)) < 4 * err


def test_film_dipolar_tail_convergence(tmp_path):
    from magex.terms import Dipolar
    s = SpinSystem(_uc(tmp_path, lattice="sc", J=[1.0]), (12, 12, 1), (True, True, False))
    vals = []
    for im in (2, 8):
        d = Dipolar(s, images=im)
        vals.append((d.energy(np.tile([0, 0, 1.0], (s.N, 1))) - d.energy(np.tile([1.0, 0, 0], (s.N, 1)))) / s.N)
    assert abs(vals[0] / vals[1] - 1) < 3e-4


def test_sparse_hessian_and_frequencies(tmp_path):
    from magex.htst import hessian, hessian_operator, llg_frequencies, llg_frequencies_sparse
    from magex.minimize import minimize
    from magex.structure import apply_structure
    uc = _uc(tmp_path, lattice="triangular", J=[1.0], D=0.6)
    s = SpinSystem(uc, (10, 10, 1), (True, True, False), field=(0, 0, 6.0))
    apply_structure(s, dict(materials=[dict(name="bulk", Kc=0.01, K2=0.02)], dipolar=True))
    e, _ = minimize(s, s.init_spins("skyrmion", radius=5.0), "lbfgs", tol=1e-9)
    Hd, _ = hessian(s, e)
    H, _ = hessian_operator(s, e)
    X = np.random.default_rng(0).normal(size=(2 * s.N, 3))
    assert max(np.abs(H @ X[:, k] - Hd @ X[:, k]).max() for k in range(3)) < 1e-7 * np.abs(Hd).max()
    s2 = SpinSystem(uc, (10, 10, 1), (False, False, False), field=(0, 0, 6.0))
    e2, _ = minimize(s2, s2.init_spins("skyrmion", radius=5.0), "lbfgs", tol=1e-9)
    fd = llg_frequencies(s2, e2)[:5]
    fs = llg_frequencies_sparse(s2, e2, k=5)
    assert np.allclose(np.unique(np.round(fd, 6))[:3], np.unique(np.round(fs, 6))[:3], rtol=1e-6)


def test_htst_zero_mode_bookkeeping(tmp_path):
    from magex.htst import htst_rate, htst_rate_zero_modes
    from magex.structure import apply_structure
    uc = _uc(tmp_path, lattice="chain", a=50.0, J=[0.0], K=0.5)
    s = SpinSystem(uc, (2, 1, 1), (False, False, False))
    apply_structure(s, dict(materials=[dict(name="bulk", Ku=[-0.2, [0, 1, 0]])]))
    s.A[1] = 0.0                                      # decoupled isotropic spin: 2 zero modes everywhere
    s1 = SpinSystem(uc, (1, 1, 1), (False, False, False))
    apply_structure(s1, dict(materials=[dict(name="bulk", Ku=[-0.2, [0, 1, 0]])]))
    r1 = htst_rate(s1, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]), 50.0)
    r2 = htst_rate_zero_modes(s, np.array([[0, 0, 1.0], [0, 0, 1.0]]), np.array([[1.0, 0, 0], [0, 0, 1.0]]), 50.0,
                              n_zero=(2, 2), volumes=(4 * np.pi, 4 * np.pi))
    assert abs(r2["prefactor"] / r1["prefactor"] - 1) < 1e-9


def test_cmc_dipolar_shape_anisotropy(tmp_path):
    from magex.cmc import constrained_mc
    from magex.structure import apply_structure
    f = SpinSystem(_uc(tmp_path, lattice="sc", J=[1.0]), (6, 6, 2), (True, True, False))
    apply_structure(f, dict(dipolar=dict(images=4)))
    d = f.terms[0]
    dE = (d.energy(np.tile([0, 0, 1.0], (f.N, 1))) - d.energy(np.tile([1.0, 0, 0], (f.N, 1)))) / f.N
    r = constrained_mc(f, 0.3, np.pi / 4, therm=200, meas=400)
    assert abs(r["torque"] / dE - 1) < 0.04            # torque = -dF/dtheta = +dE sin(2 theta) (hard z axis)


def test_longitudinal_spin_dynamics_vs_lsf_mc(tmp_path):
    from magex.lsd import LongitudinalSpinDynamics
    from magex.montecarlo import LSFMonteCarlo
    s = SpinSystem(_uc(tmp_path, lattice="sc", J=[1.0]), (5, 5, 5))
    T = 15.0
    mc = LSFMonteCarlo(s, T, 1, 0.15, b=2.0)
    M = s.ref.copy()
    ss = []
    for n in range(8000):
        M = mc.sweep(M)
        if n > 500:
            ss.append(np.linalg.norm(M, axis=1).mean())
    d = LongitudinalSpinDynamics(s, T, lam=0.2, dt=1e-3, b=2.0, seed=3)
    _, tab = d.run(s.ref.copy(), 100000, 40)
    assert abs(tab[tab[:, 0] > 10, 2].mean() - np.mean(ss)) < 0.006


def test_micromag_fast_local_fields_and_single_demag():
    from magex.micromag.demag import Demag
    from magex.micromag.mesh import Mesh, geometry_mask
    from magex.micromag.system import Micromagnet
    rng = np.random.default_rng(0)
    for pbc, n, dmi in [((False,) * 3, (12, 10, 3), "interfacial"), ((True, False, False), (12, 10, 1), "bulk")]:
        mesh = Mesh(n, (2e-9, 2.5e-9, 1e-9), pbc)
        s = Micromagnet(mesh, demag=False)
        s.add_region(geometry_mask(mesh, "disk", diameter_nm=20), "a", Ms=8e5, A=1.3e-11, D=2e-3, dmi=dmi, Ku1=1e5,
                     Ku2=2e4, Kc1=3e4, anisU=[0.3, 0, 1])
        s.exchange_axis_scale = np.array([1.0, 0.7, 1.2])
        m = s.normalize(rng.normal(size=n + (3,)))
        s.use_fast = True
        Ff = s.fields(m)
        s.use_fast = False
        Fn = s.fields(m)
        for k in Fn:
            assert np.abs(Ff[k] - Fn[k]).max() <= 1e-12 * max(np.abs(Fn[k]).max(), 1e-30)
    d = Demag(Mesh((64, 64, 1), (2e-9,) * 3))
    M = rng.normal(size=(64, 64, 1, 3)) * 8e5
    H = d.field(M)
    d.single = True
    assert np.abs(d.field(M) - H).max() < 1e-6 * np.abs(H).max()


def test_uppasd_kfile_sign(tmp_path):
    from magex.interop import export_uppasd
    p = tmp_path / "k.json"
    write_model(str(p), lattice="sc", J=[1.0], K=0.2)
    out = export_uppasd(str(p), str(tmp_path / "u"))
    k = open(out + "/kfile").read().split()
    assert float(k[2]) < 0 and abs(float(k[6]) ** 2 - 1) < 1e-9          # easy axis z: K1 < 0 in UppASD
    assert "anisotropy ./kfile" in open(out + "/inpsd.dat.template").read()


def test_sparse_htst_matches_dense(tmp_path):
    from magex.htst import htst_rate_sparse, htst_rate_zero_modes, mmf
    from magex.minimize import minimize
    s = SpinSystem(_uc(tmp_path, lattice="triangular", a=3.0, J=[2.0], D=1.0, K=0.2, moment=3.0), (16, 16, 1),
                   (True, True, False), field=(0, 0, 6.0))
    emin, _ = minimize(s, s.init_spins("skyrmion", radius=6.0), "lbfgs", tol=1e-9)
    # collapsed-skyrmion saddle: shrink the minimum, then follow the softest mode
    from magex.gneb import AtomisticAdapter, GNEB, slerp_path
    ad = AtomisticAdapter(s)
    neb = GNEB(ad, slerp_path(emin, s.init_spins("ferro"), 12), 1.0, True)
    rng = np.random.default_rng(1)
    for k in range(1, len(neb.x) - 1):
        neb.x[k] = normalize(neb.x[k] + 0.02 * rng.normal(size=neb.x[k].shape))
    neb.run(0.0, 3000, 1e-4)
    esp, _ = mmf(s, neb.x[neb.result()["saddle_image"]], tol=1e-8, kick=0.0)
    rd = htst_rate_zero_modes(s, emin, esp, 20.0)
    rs = htst_rate_sparse(s, emin, esp, 20.0)
    assert rd["n_zero"] == rs["n_zero"]
    assert abs(rs["prefactor"] / rd["prefactor"] - 1) < 1e-6


@pytest.mark.parametrize("pot", [dict(type="fpu", k3=-800.0, k4=2000.0), dict(type="morse", D=300.0, a=1.2),
                                 dict(type="lj", eps=40.0)])
def test_anharmonic_lattice_potentials(tmp_path, pot):
    from magex.sld import SpinLattice
    s = SpinSystem(_uc(tmp_path, lattice="sc", J=[1.0], K=0.1), (3, 3, 3))
    rng = np.random.default_rng(0)
    sl = SpinLattice(s, lam=0.5, k_spring=500.0, potential=pot, shells="all")
    sl.u = rng.normal(size=sl.u.shape) * 0.05
    e = normalize(s.ref + 0.4 * rng.normal(size=s.ref.shape))
    F = sl.forces(e)
    for i in (3, 17):
        for a in range(3):
            sl.u[i, a] += 1e-6
            Ep = sl.energy(e)["total"]
            sl.u[i, a] -= 2e-6
            Em = sl.energy(e)["total"]
            sl.u[i, a] += 1e-6
            assert abs(-(Ep - Em) / 2e-6 - F[i, a]) < 1e-5
    out = []
    for dt in (2e-3, 1e-3):
        rng = np.random.default_rng(0)
        sl = SpinLattice(s, lam=0.5, k_spring=500.0, potential=pot, shells="all", dt=dt)
        sl.v = rng.normal(size=sl.v.shape) * 2.0
        e = normalize(s.ref + 0.4 * rng.normal(size=s.ref.shape))
        _, tab = sl.run(e, int(0.6 / dt), every=int(0.02 / dt))
        out.append(np.ptp(tab[:, 1]))
    assert 3.3 < out[0] / out[1] < 4.7


def test_demag_xp_path():
    from magex.micromag.demag import Demag
    from magex.micromag.mesh import Mesh
    d = Demag(Mesh((16, 12, 2), (2e-9,) * 3))
    M = np.random.default_rng(0).normal(size=(16, 12, 2, 3))
    H = d.field(M)
    d.xp = np                           # the code path used with xp = cupy after to_gpu()
    assert np.abs(d.field(M) - H).max() == 0.0

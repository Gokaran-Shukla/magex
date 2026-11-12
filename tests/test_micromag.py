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

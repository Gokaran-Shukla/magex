"""Model spin Hamiltonians written in the magex ``results.json`` format.

Lets every simulation tool (LLG, Monte Carlo, micromagnetic parameter extraction) run on
textbook lattices without a DFT calculation, and provides exactly solvable benchmarks.

    make_model("triangular", J=[1.0], D=0.3, dmi="interfacial", K=0.0, moment=3.0)

Couplings are per ordered pair in the magex convention  E = -sum_{i!=j} e_i.J_ij.e_j
(so a Heisenberg bond with J' in  -J' sum_<ij> e_i.e_j  corresponds to J = J'/2).
DMI vectors: interfacial (C_nv, normal n):  D_ij = D (n x r_ij)/|r_ij|;  bulk (B20):
D_ij = D r_ij/|r_ij|.  Single-ion anisotropy: E_i = -K (e_i.n_K)^2 (K > 0 easy axis).
"""
from __future__ import annotations

import itertools
import json

import numpy as np

LATTICES = {
    # name: (cell rows in units of a, fractional basis, periodic dims)
    "chain": (np.diag([1.0, 10.0, 10.0]), [[0, 0, 0]], 1),
    "square": (np.diag([1.0, 1.0, 10.0]), [[0, 0, 0]], 2),
    "triangular": (np.array([[1, 0, 0], [0.5, np.sqrt(3) / 2, 0], [0, 0, 10.0]]), [[0, 0, 0]], 2),
    "honeycomb": (np.array([[1, 0, 0], [0.5, np.sqrt(3) / 2, 0], [0, 0, 10.0]]) * np.sqrt(3),
                  [[1 / 3, 1 / 3, 0], [2 / 3, 2 / 3, 0]], 2),
    "kagome": (np.array([[2, 0, 0], [1, np.sqrt(3), 0], [0, 0, 10.0]]),
               [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]], 2),
    "sc": (np.eye(3), [[0, 0, 0]], 3),
    "bcc": (np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]]), [[0, 0, 0]], 3),
    "fcc": (np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]), [[0, 0, 0]], 3),
}


def _skew(D):
    """3x3 antisymmetric J^A with (J^A_yz, J^A_zx, J^A_xy) = D."""
    return np.array([[0, D[2], -D[1]], [-D[2], 0, D[0]], [D[1], -D[0], 0]])


def make_model(lattice="sc", a=3.0, J=(1.0,), D=0.0, dmi="interfacial", normal=(0, 0, 1), K=0.0,
               K_axis=(0, 0, 1), moment=2.0, order="ferro", afm_signs=None, Jani=None, mesh=None):
    """Return a results.json-style dict.

    J        couplings per neighbour shell (meV, per ordered pair), J[0] = nearest neighbours
    D        nearest-neighbour DMI magnitude (meV)
    order    "ferro" or "afm" (reference signs; for "afm" give afm_signs per sublattice, or a
             checkerboard is used on bipartite lattices with one site per cell via a doubled cell)
    """
    if lattice not in LATTICES:
        raise ValueError(f"lattice must be one of {list(LATTICES)}")
    cell0, basis, pdim = LATTICES[lattice]
    cell = np.array(cell0, float) * a
    basis = np.array(basis, float)
    signs = np.ones(len(basis))
    if order == "afm":
        if afm_signs is not None:
            signs = np.array(afm_signs, float)
        elif lattice in ("sc", "square", "chain"):
            # doubled (checkerboard / rocksalt) cell
            if lattice == "chain":
                cell[0] *= 2
                basis = np.array([[0, 0, 0], [0.5, 0, 0]])
            elif lattice == "square":
                cell = np.array([[a, a, 0], [a, -a, 0], [0, 0, 10 * a]])
                basis = np.array([[0, 0, 0], [0.5, 0.5, 0]])
            else:
                cell = np.array([[a, a, 0], [a, 0, a], [0, a, a]])
                basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
            signs = np.array([1.0, -1.0])
        else:
            raise ValueError("give afm_signs for this lattice")
    ns = len(basis)
    pos = basis @ cell
    nshell = len(J)
    rng = 4
    cand = []
    rr = range(-rng, rng + 1)
    for i in range(ns):
        for j in range(ns):
            for R in itertools.product(rr, rr if pdim >= 2 else [0], rr if pdim >= 3 else [0]):
                if i == j and R == (0, 0, 0):
                    continue
                r = pos[j] + np.array(R) @ cell - pos[i]
                cand.append((round(float(np.linalg.norm(r)), 6), i, j, R, r))
    dists = sorted({c[0] for c in cand})[:nshell]
    nvec = np.array(normal, float) / np.linalg.norm(normal)
    mesh = mesh or [64 if pdim >= 1 else 1, 64 if pdim >= 2 else 1, 64 if pdim >= 3 else 1]
    pairs = []
    for d, i, j, R, r in cand:
        if d not in dists:
            continue
        s = dists.index(d)
        Jt = J[s] * np.eye(3)
        if Jani is not None and s == 0:
            Jt = Jt + np.asarray(Jani, float)
        if s == 0 and D != 0.0:
            rh = r / np.linalg.norm(r)
            Dv = D * (np.cross(nvec, rh) if dmi == "interfacial" else rh)
            Jt = Jt + _skew(Dv)
        nu = [int(R[k]) % mesh[k] for k in range(3)]
        pairs.append(dict(i=i, j=j, R=list(R), nu=nu, dist=d, vec=r.tolist(), J=Jt.tolist(), resid=0.0, degeneracy=1))
    kax = np.array(K_axis, float) / np.linalg.norm(K_axis)
    Aion = -K * (np.outer(kax, kax) - np.eye(3) / 3)
    ref = np.array([0, 0, 1.0])
    return dict(version="model", sites=list(range(ns)), symbols=["X"] * ns, cell=cell.tolist(), positions=pos.tolist(),
                mesh=mesh, moments=(signs[:, None] * moment * ref[None, :]).tolist(), charges=[0.0] * ns,
                refs=[ref.tolist()], full_tensor=True, A=[Aion.tolist()] * ns, mu=0.0, mag_pos=list(range(ns)),
                lig_pos=[], pairs=pairs, corrections={}, lloyd=None, mae=None, symmetry=None,
                model=dict(lattice=lattice, a=a, J=list(J), D=D, dmi=dmi, K=K, moment=moment, order=order))


def write_model(path, **kw):
    d = make_model(**kw)
    with open(path, "w") as f:
        json.dump(d, f)
    return path

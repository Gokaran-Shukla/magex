"""K(i,j,R) from a k-mesh must equal the Gamma-only calculation of the explicit supercell."""
import itertools
import numpy as np
from magex.contour import Contour
from magex.engine import compute_reference
from magex.model import TBModel
from toymodels import random_cluster


def supercell(m, mesh):
    cells = list(itertools.product(*[range(n) for n in mesh]))
    n = m.norb
    N = n * len(cells)
    H0 = np.zeros((1, N, N), complex)
    X = np.zeros((1, 3, N, N), complex)
    S = np.zeros((1, 3, N, N), complex)
    for c1, C1 in enumerate(cells):
        for ir, R in enumerate(m.R):
            C2 = tuple((np.array(C1) + R) % np.array(mesh))
            c2 = cells.index(C2)
            H0[0, c1 * n:(c1 + 1) * n, c2 * n:(c2 + 1) * n] += m.H0[ir]
            X[0, :, c1 * n:(c1 + 1) * n, c2 * n:(c2 + 1) * n] += m.X[ir]
            S[0, :, c1 * n:(c1 + 1) * n, c2 * n:(c2 + 1) * n] += m.S[ir]
    oa = np.concatenate([m.orb_atom + c * m.natoms for c in range(len(cells))])
    return TBModel(m.cell, np.zeros((m.natoms * len(cells), 3)), m.symbols * len(cells),
                   np.zeros((1, 3), int), H0, X, S, oa, np.tile(m.orb_l, len(cells)),
                   np.tile(m.orb_m, len(cells))), cells


def test_periodic():
    Rl = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [1, 1, 0], [-1, -1, 0], [0, 0, 1], [0, 0, -1]]
    m = random_cluster(nat=2, no=2, seed=11, R=Rl, hop=0.3)
    mesh = (3, 2, 2)
    mu, T = 0.1, 1500.0
    c = Contour("ozaki", mu, T, npole=300)
    r1 = compute_reference(m, [0, 1], mesh, c)
    sc, cells = supercell(m, mesh)
    r2 = compute_reference(sc, list(range(sc.natoms)), (1, 1, 1), c)
    err = 0.0
    for i in range(2):
        for j in range(2):
            for nu in itertools.product(*[range(n) for n in mesh]):
                a = r1["K"][i, j][nu]
                b = r2["K"][i, cells.index(nu) * 2 + j, 0, 0, 0]
                err = max(err, np.max(np.abs(a - b)))
    print("max |K_kmesh - K_supercell| =", err, " T1 diff", np.max(np.abs(r1["T1"] - r2["T1"][:2])))
    assert err < 1e-9


if __name__ == "__main__":
    test_periodic()

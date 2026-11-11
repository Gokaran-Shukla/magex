"""Random tight-binding models for exact validation."""
import numpy as np
from magex.model import TBModel


def random_cluster(nat=3, no=3, soc=0.15, exch=1.2, seed=0, R=None, hop=0.6):
    """Finite cluster (R=[0]) or periodic model if R list given.

    Returns a TBModel whose exchange field is along z (atom 1 antiparallel for AFM flavour).
    """
    rng = np.random.default_rng(seed)
    n = nat * no
    Rl = np.array([[0, 0, 0]] if R is None else R)
    nR = len(Rl)
    H0 = np.zeros((nR, n, n), complex)
    X = np.zeros((nR, 3, n, n), complex)
    S = np.zeros((nR, 3, n, n), complex)
    idx = {tuple(r): i for i, r in enumerate(Rl)}
    for ir, r in enumerate(Rl):
        mr = idx.get(tuple(-np.array(r)))
        if mr is not None and mr < ir:
            continue
        A = rng.normal(size=(n, n)) * hop
        Bx = rng.normal(size=(n, n)) * hop * 0.25      # spin-dependent hopping (time odd)
        if mr == ir:
            A = 0.5 * (A + A.T)
            Bx = 0.5 * (Bx + Bx.T)
        H0[ir] += A
        X[ir, 2] += Bx
        if mr != ir:
            H0[mr] += A.T
            X[mr, 2] += Bx.T
    r0 = idx[(0, 0, 0)]
    orb_atom = np.repeat(np.arange(nat), no)
    for a in range(nat):
        o = np.where(orb_atom == a)[0]
        sgn = -1.0 if a == 1 else 1.0
        D = np.diag(exch * (1 + 0.3 * rng.random(no)))
        X[r0, 2][np.ix_(o, o)] += sgn * D
        for c in range(3):
            M = rng.normal(size=(no, no))
            M = 0.5 * (M - M.T)
            S[r0, c][np.ix_(o, o)] += 1j * soc * M
        H0[r0][np.ix_(o, o)] += np.diag(rng.normal(size=no) * 0.5)
    cell = np.eye(3) * 3.0
    pos = rng.random((nat, 3)) * 3.0
    return TBModel(cell, pos, ["Fe"] * nat, Rl, H0, X, S, orb_atom,
                   np.ones(n, int), np.tile(np.arange(no), nat), 0.0)


def grand_potential(model, kpts, mu, kT, rotations=None):
    """Exact Omega at T on k-points, with optional {atom: 3x3 rotation} of on-site exchange fields."""
    m = model
    if rotations:
        X = m.X.copy()
        r0 = m.r0_index()
        for a, rot in rotations.items():
            o = m.atom_orbitals(a)
            blk = X[r0][:, o][:, :, o]
            X[r0][:, o[:, None], o[None, :]] = np.einsum("ab,bmn->amn", rot, blk)
        m = TBModel(m.cell, m.positions, m.symbols, m.R, m.H0, X, m.S, m.orb_atom, m.orb_l, m.orb_m)
    e = np.linalg.eigvalsh(m.Hk(kpts))
    x = (e - mu) / kT
    return float(np.sum(-kT * np.logaddexp(0, -x)) / len(kpts))

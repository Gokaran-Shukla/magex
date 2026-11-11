"""Weighted-partition engine: equals the on-site engine in 'onsite' mode, matches exact
finite differences of the partitioned rotation, and satisfies the Goldstone sum rule."""
import numpy as np
from magex.contour import Contour
from magex.engine import compute_reference, rodrigues
from magex.engine_partition import compute_reference_partition, partition_weights
from magex.model import TBModel
from toymodels import random_cluster, grand_potential

RL = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]


def test_onsite_mode_equals_fast_engine():
    m = random_cluster(nat=2, no=2, seed=5, R=RL, hop=0.3)
    c = Contour("ozaki", 0.1, 1500, npole=200)
    ll = dict(pairs=[(0, 1, (1, 0, 0))], angles=np.linspace(0, np.pi, 4), axis=[0, 1, 0])
    a = compute_reference(m, [0, 1], (3, 2, 1), c, lloyd=ll)
    b = compute_reference_partition(m, [0, 1], (3, 2, 1), c, mode="onsite", lloyd=ll)
    e1 = np.max(np.abs(a["K"] - b["K"])); e2 = np.max(np.abs(a["T1"] - b["T1"]))
    e3 = np.max(np.abs(a["lloyd"]["E"] - b["lloyd"]["E"]))
    print("onsite-mode vs fast engine:", e1, e2, e3)
    assert max(e1, e2, e3) < 1e-10


def _rotate_partition(m, rots, f):
    """Exact X -> X with X^(i) rotated (cluster, R=[0] only)."""
    X = m.X.copy()
    for a, rot in rots.items():
        P = np.zeros(m.norb); P[m.atom_orbitals(a)] = 1
        XL = m.X[0] * f[None]
        Xi = P[None, :, None] * XL + np.conj(np.swapaxes(XL, -1, -2)) * P[None, None, :]
        X[0] += np.einsum("ab,bmn->amn", rot, Xi) - Xi
    return TBModel(m.cell, m.positions, m.symbols, m.R, m.H0, X, m.S, m.orb_atom, m.orb_l, m.orb_m)


def test_weighted_exact_fd():
    m = random_cluster(nat=3, no=2, seed=8, soc=0.1)
    E = np.linalg.eigvalsh(m.Hk(np.zeros((1, 3))))[0]
    mu = 0.5 * (E[5] + E[6])
    c = Contour("ozaki", mu, 2000, npole=400)
    r = compute_reference_partition(m, [0, 1, 2], (1, 1, 1), c, mode="weighted")
    f, _ = partition_weights(m, "weighted")
    h, err, mx = 1e-3, 0.0, 0.0
    k0 = np.zeros((1, 3))
    for i in range(3):
        for j in range(3):
            for a in range(3):
                for b in range(3):
                    def Om(si, sj):
                        if i == j:
                            v = si * h * np.eye(3)[a] + sj * h * np.eye(3)[b]
                            rots = {i: rodrigues(v, np.linalg.norm(v)) if np.linalg.norm(v) else np.eye(3)}
                        else:
                            rots = {i: rodrigues(np.eye(3)[a], si * h), j: rodrigues(np.eye(3)[b], sj * h)}
                        mm = _rotate_partition(m, rots, f)
                        return grand_potential(mm, k0, mu, c.kT)
                    fd = (Om(1, 1) - Om(1, -1) - Om(-1, 1) + Om(-1, -1)) / (4 * h * h)
                    gf = (0.5 * (r["K"][i, i, 0, 0, 0, a, b] + r["K"][i, i, 0, 0, 0, b, a]) + r["T1"][i, a, b]
                          if i == j else r["K"][i, j, 0, 0, 0, a, b])
                    err, mx = max(err, abs(fd - gf)), max(mx, abs(fd))
    print("weighted partition: max |FD-GF| =", err, " scale", mx)
    assert err < 1e-4 * max(1, mx)


def test_sum_rule():
    m = random_cluster(nat=2, no=2, seed=5, R=RL, hop=0.3, soc=0.0)
    c = Contour("ozaki", 0.1, 1500, npole=200)
    for mode in ("onsite", "weighted"):
        r = compute_reference_partition(m, [0, 1], (3, 2, 1), c, mode=mode)
        sr = r["K"][..., 0, 0].sum(axis=(1, 2, 3, 4)) + r["T1"][:, 0, 0]
        print(mode, "sum rule", sr)
    assert np.max(np.abs(sr)) < 1e-9


if __name__ == "__main__":
    test_onsite_mode_equals_fast_engine(); test_weighted_exact_fd(); test_sum_rule(); test_lloyd_exact()


def test_lloyd_exact():
    """Finite-angle interaction energies (Lloyd) vs exact grand potentials, both partitions."""
    m = random_cluster(nat=3, no=2, seed=4, soc=0.1)
    E = np.linalg.eigvalsh(m.Hk(np.zeros((1, 3))))[0]
    mu = 0.5 * (E[5] + E[6])
    c = Contour("ozaki", mu, 2000, npole=400)
    ang = np.linspace(0, np.pi, 5)
    ll = dict(pairs=[(0, 2, (0, 0, 0))], angles=ang, axis=[0, 1, 0])
    k0 = np.zeros((1, 3))
    for mode in ("onsite", "weighted"):
        r = compute_reference_partition(m, [0, 1, 2], (1, 1, 1), c, mode=mode, lloyd=ll)
        f, _ = partition_weights(m, mode)
        if mode == "onsite":
            f = np.zeros_like(f)
            for a in range(3):
                o = m.atom_orbitals(a); f[np.ix_(o, o)] = 0.5
        ex = []
        for th in ang:
            Ri, Rj = rodrigues([0, 1, 0], -th / 2), rodrigues([0, 1, 0], th / 2)
            O = lambda rots: grand_potential(_rotate_partition(m, rots, f), k0, mu, c.kT)
            ex.append(O({0: Ri, 2: Rj}) - O({0: Ri}) - O({2: Rj}) + O({}))
        err = np.max(np.abs(np.array(ex) - r["lloyd"]["E"][0]))
        print(mode, "Lloyd vs exact:", err, " scale", np.max(np.abs(ex)))
        assert err < 1e-8

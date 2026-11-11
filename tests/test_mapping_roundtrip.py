"""Synthetic round trip: Heisenberg (J tensors, A) -> Hessians -> fitted parameters."""
import itertools
import numpy as np
from types import SimpleNamespace
from magex.heisenberg import build_results

E3 = np.eye(3)


def test_roundtrip():
    rng = np.random.default_rng(7)
    mesh = (3, 2, 1)
    ns = 3
    s = np.array([1, -1, 1.0])
    nus = list(itertools.product(*[range(n) for n in mesh]))
    J = {}
    for i in range(ns):
        for j in range(ns):
            for nu in nus:
                if i == j and nu == (0, 0, 0):
                    continue
                key = (i, j, nu)
                mnu = tuple((-np.array(nu)) % np.array(mesh))
                if (j, i, mnu) in J:
                    J[key] = J[(j, i, mnu)].T
                else:
                    J[key] = rng.normal(size=(3, 3)) * 0.3 + np.eye(3) * rng.normal()
    A = []
    for i in range(ns):
        M = rng.normal(size=(3, 3)); M = 0.5 * (M + M.T); M -= np.trace(M) / 3 * E3
        A.append(M * 0.2)
    refs = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    refdata = []
    for d in refs:
        d = np.array(d, float)
        K = np.zeros((ns, ns) + mesh + (3, 3))
        T1 = np.zeros((ns, 3, 3))
        for (i, j, nu), Jt in J.items():
            for a in range(3):
                for b in range(3):
                    K[i, j][nu][a, b] = -2 * np.cross(E3[a], s[i] * d) @ Jt @ np.cross(E3[b], s[j] * d) / 1000
        for i in range(ns):
            n = s[i] * d
            c = sum(-2 * Jt @ (s[j] * d) for (ii, j, nu), Jt in J.items() if ii == i)
            for a in range(3):
                for b in range(3):
                    ta, tb = np.cross(E3[a], n), np.cross(E3[b], n)
                    Sab = 0.5 * (np.cross(E3[a], np.cross(E3[b], n)) + np.cross(E3[b], np.cross(E3[a], n)))
                    T1[i, a, b] = (2 * ta @ A[i] @ tb + 2 * n @ A[i] @ Sab + c @ Sab) / 1000
        refdata.append(dict(K=K, T1=T1, moments=np.outer(s, d) * 2, charges=np.zeros(ns), mesh=mesh))
    model = SimpleNamespace(frac_positions=rng.random((ns, 3)), cell=np.diag([4., 5., 20.]),
                            positions=np.zeros((ns, 3)), symbols=["Fe"] * ns)
    res = build_results(model, list(range(ns)), refdata, refs, None, True)
    err = max(np.max(np.abs(p.J - J[(p.i, p.j, p.nu)])) for p in res.pairs)
    errA = max(np.max(np.abs(res.A[i] - A[i])) for i in range(ns))
    print("max J error", err, " max A error", errA)
    assert err < 1e-10 and errA < 1e-10


if __name__ == "__main__":
    test_roundtrip()

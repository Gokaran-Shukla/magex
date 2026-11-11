import numpy as np
from magex.contour import Contour
from magex.engine import compute_reference, rodrigues
from toymodels import random_cluster, grand_potential


def test_cluster_hessian():
    m = random_cluster(nat=3, no=3, seed=3)
    E = np.linalg.eigvalsh(m.Hk(np.zeros((1, 3))))[0]
    mu = 0.5 * (E[8] + E[9])            # some filling
    T = 2000.0
    c = Contour("ozaki", mu, T, bandwidth=float(E[-1] - E[0]) + 5, npole=400)
    res = compute_reference(m, [0, 1, 2], (1, 1, 1), c)
    K, T1 = res["K"][:, :, 0, 0, 0], res["T1"]
    kpts = np.zeros((1, 3))
    h = 1e-3
    maxerr, maxval = 0.0, 0.0
    for i in range(3):
        for j in range(3):
            for a in range(3):
                for b in range(3):
                    ax_a = np.eye(3)[a]
                    ax_b = np.eye(3)[b]
                    def Om(si, sj):
                        rots = {}
                        if i == j:
                            v = si * h * ax_a + sj * h * ax_b
                            rots[i] = rodrigues(v, np.linalg.norm(v)) if np.linalg.norm(v) > 0 else np.eye(3)
                        else:
                            rots[i] = rodrigues(ax_a, si * h)
                            rots[j] = rodrigues(ax_b, sj * h)
                        return grand_potential(m, kpts, mu, c.kT, rots)
                    fd = (Om(1, 1) - Om(1, -1) - Om(-1, 1) + Om(-1, -1)) / (4 * h * h)
                    if i == j:
                        # symmetrise the two orderings
                        gf = 0.5 * (K[i, i, a, b] + K[i, i, b, a]) + T1[i, a, b]
                    else:
                        gf = K[i, j, a, b]
                    maxerr = max(maxerr, abs(fd - gf))
                    maxval = max(maxval, abs(fd))
    print("max |FD - GF| =", maxerr, " max |Hessian| =", maxval)
    assert maxerr < 1e-4 * max(1, maxval)


if __name__ == "__main__":
    test_cluster_hessian()

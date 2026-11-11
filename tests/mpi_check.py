"""MPI consistency check used by CI: run the same toy model serially and on several ranks."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from toymodels import random_cluster  # noqa: E402

from magex import parallel as par  # noqa: E402
from magex.contour import Contour  # noqa: E402
from magex.engine_partition import compute_reference_partition  # noqa: E402

RL = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
OUT = os.path.join(os.path.dirname(__file__), "_mpi_{}.npy")


def run():
    m = random_cluster(nat=2, no=2, seed=5, R=RL, hop=0.3)
    c = Contour("ozaki", 0.1, 1500, npole=120)
    ll = dict(pairs=[(0, 1, (1, 0, 0))], angles=np.linspace(0, np.pi, 5), axis=[0, 1, 0])
    r = compute_reference_partition(m, [0, 1], (4, 3, 1), c, lloyd=ll)
    return np.concatenate([r["K"].ravel(), r["T1"].ravel(), r["lloyd"]["E"].ravel()])


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode in ("serial", "mpi"):
        v = run()
        if par.is_root():
            np.save(OUT.format(mode), v)
            print(mode, "ranks:", par.SIZE)
    else:
        a, b = np.load(OUT.format("serial")), np.load(OUT.format("mpi"))
        err = np.max(np.abs(a - b))
        print("max |serial - mpi| =", err)
        assert err < 1e-12

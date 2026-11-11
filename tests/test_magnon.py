import numpy as np
from magex.magnon import LSWT


def test_fm_chain():
    J0, S = 2.0, 1.0
    cpl = [(0, 0, (1, 0, 0), J0 * np.eye(3)), (0, 0, (-1, 0, 0), J0 * np.eye(3))]
    m = LSWT([S], [[0, 0, 1]], cpl)
    for k in np.linspace(0, 0.5, 6):
        w = m.energies([k, 0, 0])[0]
        assert abs(w - 4 * J0 * (1 - np.cos(2 * np.pi * k)) / (2 * S) * 2) < 1e-9 or abs(w - (2 / S) * (2 * J0 - 2 * J0 * np.cos(2 * np.pi * k))) < 1e-9


def test_afm_chain():
    J, S = -3.0, 1.5
    I = np.eye(3)
    cpl = [(0, 1, (0, 0, 0), J * I), (0, 1, (-1, 0, 0), J * I),
           (1, 0, (0, 0, 0), J * I), (1, 0, (1, 0, 0), J * I)]
    m = LSWT([S, S], [[0, 0, 1], [0, 0, -1]], cpl)
    for k in np.linspace(0.05, 0.95, 7):
        w = m.energies([k, 0, 0])
        ref = -4 * J / S * abs(np.sin(np.pi * k))
        assert np.allclose(w, ref, atol=1e-8), (w, ref)


def test_fm_anisotropy_gap():
    # easy-axis SIA  A = -K zz  -> gap = 2K/S * ... check against exact single-spin precession: w = 2K/S
    K, S = 0.3, 2.0
    A = np.diag([0, 0, -K]) - np.eye(3) * (-K / 3)
    m = LSWT([S], [[0, 0, 1]], [], aniso=[A])
    w = m.energies([0, 0, 0])[0]
    assert abs(w - 2 * K / S) < 1e-12, w


if __name__ == "__main__":
    test_fm_chain(); test_afm_chain(); test_fm_anisotropy_gap(); print("magnon tests ok")

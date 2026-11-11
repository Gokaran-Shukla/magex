"""Energy integration of  -(1/pi) Im \\int f(e) F(e+i0) de  for F analytic in the UHP.

Two schemes:

* ``ozaki``  (default)  Fermi-Dirac at temperature T via the continued-fraction
  pole expansion of Ozaki, PRB 75, 035123 (2007).  No lower energy bound is
  needed, convergence is exponential in the number of poles, and for metals the
  small Matsubara-like offset (~ pi kT) regularises the k-sum near E_F.
      -(1/pi) Im \\int f F de  =  2 kT  sum_p  R_p  Re F(mu + i z_p kT)
  valid when F(z) decays at least as 1/z^2 (all integrands used here are traces
  of Pauli-traceless perturbations, so this holds).

* ``semicircle``  T = 0 contour from Emin to E_F through the upper half plane
  with Gauss-Legendre nodes in the angle (TB2J-like).

Both return complex nodes ``z`` and complex weights ``w`` such that

      -(1/pi) Im \\int f(e) F(e+i0) de  ~=  sum_n  Re[ w_n F(z_n) ]   (ozaki)
                                       or  -(1/pi) Im sum_n w_n F(z_n)  (semicircle)

To give callers one formula we return weights already converted so that

      result = sum_n Re( w_n * F(z_n) )      in both cases.
"""
from __future__ import annotations

import numpy as np

KB_EV = 8.617333262e-5  # eV/K


def ozaki_poles(npole: int):
    """Return (z_p, R_p): positive poles and residues of the Ozaki CF expansion.

    f(x) = 1/2 - sum_p R_p [ 1/(x - i z_p) + 1/(x + i z_p) ],  x = (e-mu)/kT
    """
    n = 2 * npole
    b = np.array([1.0 / (2.0 * np.sqrt((2 * k + 1) * (2 * k + 3))) for k in range(n - 1)])
    B = np.diag(b, 1) + np.diag(b, -1)
    lam, vec = np.linalg.eigh(B)
    sel = lam > 0
    lam = lam[sel]
    vec = vec[:, sel]
    z = 1.0 / lam
    R = np.abs(vec[0, :]) ** 2 / (4.0 * lam**2)
    order = np.argsort(z)
    return z[order], R[order]


def fermi_ozaki(x, npole):
    z, R = ozaki_poles(npole)
    x = np.asarray(x, float)[..., None]
    return 0.5 - np.sum(R * (1.0 / (x - 1j * z) + 1.0 / (x + 1j * z)), axis=-1).real


def ozaki_npoles(bandwidth_ev: float, kT: float, tol: float = 1e-10) -> int:
    """Smallest pole count reproducing the Fermi function to tol over the band range."""
    xmax = max(bandwidth_ev / kT, 10.0)
    xs = np.linspace(-xmax, xmax, 4001)
    exact = 0.5 * (1 - np.tanh(xs / 2))
    for n in (40, 60, 80, 100, 140, 180, 240, 320, 400, 500, 640, 800, 1000, 1300, 1600, 2000):
        if np.max(np.abs(fermi_ozaki(xs, n) - exact)) < tol:
            return n
    return 2000


class Contour:
    def __init__(self, kind="ozaki", efermi=0.0, temperature=300.0, npole=None,
                 emin=None, nz=100, bandwidth=None):
        self.kind = kind
        self.efermi = efermi
        if kind == "ozaki":
            kT = KB_EV * temperature
            if npole is None:
                npole = ozaki_npoles(bandwidth if bandwidth else 40.0, kT)
            zp, Rp = ozaki_poles(npole)
            self.z = efermi + 1j * zp * kT
            # -(1/pi) Im int f F = 2kT sum R Re F  ->  w = 2 kT R  (real)
            self.w = (2.0 * kT * Rp).astype(complex)
            self.kT = kT
        elif kind == "semicircle":
            if emin is None:
                raise ValueError("semicircle contour needs emin (below lowest band)")
            x, wx = np.polynomial.legendre.leggauss(nz)
            # theta from pi -> 0 ; map x in [-1,1] -> theta = pi (1-x)/2
            th = np.pi * (1 - x) / 2
            c = 0.5 * (emin + efermi)
            r = 0.5 * (efermi - emin)
            self.z = c + r * np.exp(1j * th)
            dz = 1j * r * np.exp(1j * th) * (-np.pi / 2) * wx  # dz = (dz/dth)(dth/dx) dx
            # result = -(1/pi) Im sum dz F = Re( (i/pi) dz F )
            self.w = (1j / np.pi) * dz
            self.kT = 0.0
        else:
            raise ValueError(kind)

    def __len__(self):
        return len(self.z)

    def integrate(self, Fvals):
        """Fvals: array (nz, ...) of F(z_n). Returns -(1/pi) Im int f F de."""
        return np.tensordot(self.w, Fvals, axes=(0, 0)).real

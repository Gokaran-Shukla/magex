"""SOT-driven antiferromagnetic domain-wall velocity vs current (two-sublattice micromagnetics).

    python afm_dw_velocity.py            -> afm_dw_out/afm_dw_velocity.png/.pdf

Damping-like SOT with sigma _|_ the wall plane moves an AFM Neel wall at v = (pi/2) gamma B_DL Delta / alpha
(Delta = sqrt(A/K); Gomonay et al., PRL 117, 017202 (2016)) until v approaches the magnon velocity
v_m = 2 gamma sqrt(A J_AF)/Ms, where the wall Lorentz-contracts and
the velocity saturates (no Walker breakdown).  Compare: a ferromagnet with the same parameters
breaks down at the Walker field.
"""
import os

import numpy as np

from magex.micromag import Mesh
from magex.micromag.afm import TwoSublatticeMagnet
from magex.micromag.constants import E_CHARGE, GAMMA_LL, HBAR
from magex.micromag.solver import RK45, minimize
from magex.plotstyle import OKABE_ITO, mpl, savefig

Ms, A, K, J_AF, alpha, theta, t = 4e5, 1e-11, 1e5, 4e7, 0.01, 0.1, 1e-9
mesh = Mesh((800, 1, 1), (1e-9, 1e-9, 1e-9))
os.makedirs("afm_dw_out", exist_ok=True)


def wall(s, m):
    nz = 0.5 * (m[0][:, 0, 0, 2] - m[1][:, 0, 0, 2])
    X = s.mesh.coords()[0][:, 0, 0]
    i = np.where(np.diff(np.sign(nz)))[0][0]
    return (X[i] - nz[i] * (X[i + 1] - X[i]) / (nz[i + 1] - nz[i])) * 1e9


def width(s, m):
    nz = 0.5 * (m[0][:, 0, 0, 2] - m[1][:, 0, 0, 2])
    X = s.mesh.coords()[0][:, 0, 0] * 1e9
    i = np.argmin(abs(nz))
    return 1.0 / (np.abs(np.gradient(nz, X))[i])        # Delta from the slope at the centre (tanh profile)


Js = np.array([0.25, 0.5, 1, 2, 4, 8, 16]) * 1e11
v, d = [], []
for J in Js:
    s = TwoSublatticeMagnet(mesh, demag=False)
    s.add_region(np.ones(mesh.n, bool), Ms=Ms, A=A, Ku1=K, alpha=alpha, J_AF=J_AF)
    m = s.init("domain_wall", width_nm=10, x0_nm=-300)
    m, _, _ = minimize(s, m, 1e-4)
    s.sot = dict(J=J, theta=theta, sigma=[0, 1, 0], thickness=t)
    xs, ts = [], []
    m, _ = RK45(s).run(m, 30e-12, 2e-12, lambda tt, mm: (ts.append(tt), xs.append(wall(s, mm))))
    sel = np.array(ts) > 10e-12
    v.append(np.polyfit(np.array(ts)[sel], np.array(xs)[sel] * 1e-9, 1)[0])
    d.append(width(s, m))
    print(f"J = {J:.2e} A/m^2   v = {v[-1]:8.1f} m/s   wall width {d[-1]:.2f} nm")
v = np.array(v)
BDL = HBAR * theta * Js / (2 * E_CHARGE * Ms * t)
Delta = np.sqrt(A / K)
v_lin = np.pi / 2 * GAMMA_LL * BDL * Delta / alpha
v_m = 2 * GAMMA_LL * np.sqrt(A * J_AF) / Ms          # magnon velocity of the two-sublattice model (A_AB = 0)
plt = mpl()
fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
axs[0].loglog(Js, v, "o-", color=OKABE_ITO[0], label="two-sublattice LLG")
axs[0].loglog(Js, v_lin, "--", color=OKABE_ITO[1], label=r"$(\pi/2)\gamma B_{DL}\Delta/\alpha$")
axs[0].axhline(v_m, color="0.4", ls=":", label=f"magnon velocity {v_m / 1e3:.1f} km/s")
axs[0].set_xlabel(r"$J$ (A m$^{-2}$)")
axs[0].set_ylabel(r"wall velocity (m s$^{-1}$)")
axs[0].legend(fontsize=7)
axs[1].semilogx(v / v_m, np.array(d) / (Delta * 1e9), "s-", color=OKABE_ITO[2], label="simulation")
vv = np.linspace(0, 0.999, 200)
axs[1].semilogx(vv, np.sqrt(1 - vv**2), "--", color="k", lw=1, label=r"$\sqrt{1 - v^2/v_m^2}$")
axs[1].set_xlabel(r"$v / v_m$")
axs[1].set_ylabel(r"wall width $\Delta(v)/\Delta_0$")
axs[1].legend(fontsize=7)
for k, ax in enumerate(axs):
    ax.set_title(f"({'ab'[k]})", loc="left")
savefig(fig, "afm_dw_out/afm_dw_velocity.png")
np.savetxt("afm_dw_out/afm_dw_velocity.txt", np.column_stack([Js, v, d]), header="J_A/m2 v_m/s width_nm")

"""Ultrafast laser-induced demagnetisation of Ni: LLB + two-temperature model.

    python ultrafast_demag.py       -> ultrafast_out/ultrafast_demag.png/.pdf

A 50 fs pump heats the electrons (two-temperature model); the spin system follows Te through the
Landau-Lifshitz-Bloch equation with mean-field m_e(T) and chi_par(T).  Increasing fluence gives the
sub-picosecond quenching and a recovery that slows down as Te approaches Tc (critical slowing
down).  The LLB here is deterministic: if Te stays far above Tc, |m| reaches 0 and can only recover
through fluctuations, which need the stochastic LLB (not implemented).
"""
import os

import numpy as np

from magex.micromag import Mesh, Micromagnet
from magex.micromag.llb import LLB, TwoTemperature
from magex.plotstyle import OKABE_ITO, mpl, savefig

os.makedirs("ultrafast_out", exist_ok=True)
Tc = 630.0                      # Ni
mesh = Mesh((1, 1, 1), (2e-9,) * 3)
res = []
for P0 in (5e20, 1e21, 1.5e21, 2e21):
    s = Micromagnet(mesh, demag=False)
    s.add_region(np.ones(mesh.n, bool), Ms=4.9e5, A=9e-12, Ku1=1e4)
    llb = LLB(s, Tc=Tc, lam=0.05, mu_s=0.6)
    ttm = TwoTemperature(T0=300.0, gamma_e=1065.0, C_l=4.0e6, G=8e17, P0=P0, t0=0.3e-12, tau=0.05e-12)
    llb.set_temperature(300.0)
    m = np.zeros(mesh.n + (3,))
    m[..., 2] = llb.me
    m, tab = llb.run(m, 4e-12, 2e-16, T_of_t=ttm.advance, record_every=10e-15)
    res.append((P0, tab))
    print(f"P0 = {P0:.1e} W/m^3: Te max {tab[:, 1].max():.0f} K, min |m|/m0 {tab[:, 5].min() / tab[0, 5]:.3f}")
plt = mpl()
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2), constrained_layout=True)
for k, (P0, tab) in enumerate(res):
    t = (tab[:, 0] - 0.3e-12) * 1e12
    axs[0].plot(t, tab[:, 5] / tab[0, 5], color=OKABE_ITO[k], label=f"$P_0$ = {P0:.1e} W m$^{{-3}}$")
    axs[1].plot(t, tab[:, 1], color=OKABE_ITO[k])
axs[0].set_ylabel(r"$|\mathbf{m}|(t)/|\mathbf{m}|(0)$")
axs[1].set_ylabel(r"electron temperature $T_e$ (K)")
axs[1].axhline(Tc, color="0.5", ls=":", lw=1)
axs[1].text(3.4, Tc + 30, "$T_C$", color="0.4", ha="right")
for k, ax in enumerate(axs):
    ax.set_xlabel("delay (ps)")
    ax.set_title(f"({'ab'[k]})", loc="left")
axs[0].legend(fontsize=7)
savefig(fig, "ultrafast_out/ultrafast_demag.png")

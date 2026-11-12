"""Gilbert damping from first principles: torque-torque correlation (Kambersky) on the Wannier model.

Linear response of the band energy to a slowly rotating magnetisation gives the Gilbert term
of the LLG equation with (per magnetic unit cell, T = 0)

    alpha_{mu nu} = g / (pi M)  (1/N_k) sum_k  Tr[ T_mu(k) A(k) T_nu(k) A(k) ]        (1)

    A(k) = eta [ (E_F - H(k))^2 + eta^2 ]^{-1}  = -Im G^R(k, E_F)  (Lorentzian broadening eta)

where M is the spin moment per cell (mu_B), g ~ 2, and T_mu = dH/dtheta_mu is the torque
operator for a rotation about the axis e_mu perpendicular to the magnetisation n.  Two exactly
equivalent (for eta -> 0) choices are implemented:

  "exchange":  T_mu = (e_mu x X) . sigma        rotate only the time-odd exchange field X
                                                 (the magex reference rotation; works for spinor
                                                 and collinear+atomic-SOC input)
  "soc":       T_mu = (i/2) [sigma_mu, H_so]      SOC torque of Kambersky / Gilmore-Stiles

Their difference at finite eta measures the broadening artefact (the exchange form carries a
spurious interband term O(eta) that the SOC form does not have).  Intraband (n = m, "breathing
Fermi surface", ~1/eta) and interband (n != m, "bubbling", ~eta) parts are reported separately,
which gives the characteristic minimum of alpha(eta) of transition metals (Gilmore et al., PRL
99, 027204 (2007)).  eta plays the role of hbar/(2 tau) for electron scattering; it can be
related to the resistivity or to temperature through eta(T).

Finite electronic temperature: the Fermi-surface delta is replaced by -df/dE on a
Gauss-Legendre grid (``kT``), which also stabilises the k-sum.

Derivation of the prefactor: the dissipated power is P = Lambda_{mu nu} dtheta_mu dtheta_nu /dt^2
with Lambda = pi hbar Tr[T delta(E_F - H) T delta(E_F - H)] (Kubo, omega -> 0), and the Gilbert
form P = (alpha M mu_B / gamma) |dm/dt|^2 with gamma = g mu_B / hbar gives alpha = g Lambda/(hbar M);
delta = A/pi then yields (1).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from . import parallel as par
from .model import PAULI, rotation_to

G_FACTOR = 2.0


def _perp_axes(n):
    n = np.asarray(n, float) / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


def _spin_field_HR(F, norb):
    """Spinor H(R) of a pure spin field F (nR,3,n,n): F . sigma."""
    n = norb
    out = np.zeros((F.shape[0], 2 * n, 2 * n), complex)
    out[:, :n, :n] = F[:, 2]
    out[:, n:, n:] = -F[:, 2]
    out[:, :n, n:] = F[:, 0] - 1j * F[:, 1]
    out[:, n:, :n] = F[:, 0] + 1j * F[:, 1]
    return out


def torque_operators(model, n, form="exchange"):
    """Real-space torque operators T_1, T_2 (each (nR, 2n, 2n)) for rotations about e1, e2 _|_ n."""
    e1, e2 = _perp_axes(n)
    out = []
    for e in (e1, e2):
        if form == "exchange":
            # (e x X)_a = eps_abc e_b X_c
            F = np.cross(np.broadcast_to(e, (1, 1, 1, 3)), np.moveaxis(model.X, 1, -1)).transpose(0, 3, 1, 2)
            out.append(_spin_field_HR(np.ascontiguousarray(F), model.norb))
        elif form == "soc":
            Hso = _spin_field_HR(model.S, model.norb)
            sig = sum(e[c] * np.kron(PAULI[c], np.eye(model.norb)) for c in range(3))
            out.append(0.5j * (np.einsum("ab,rbc->rac", sig, Hso) - np.einsum("rab,bc->rac", Hso, sig)))
        else:
            raise ValueError("form must be 'exchange' or 'soc'")
    return out, (e1, e2)


def kgrid(mesh, shift=(0, 0, 0)):
    g = [(np.arange(m) + s) / m for m, s in zip(mesh, shift)]
    return np.array(np.meshgrid(*g, indexing="ij")).reshape(3, -1).T


def _fermi_weights(kT, npts=16):
    """Nodes x (eV, relative to E_F) and weights of -df/dE (sum to 1)."""
    if kT <= 0:
        return np.zeros(1), np.ones(1)
    t, w = np.polynomial.legendre.leggauss(npts)
    x = 12 * kT * t                                    # |x| < 12 kT
    f = 1 / (4 * kT * np.cosh(x / (2 * kT)) ** 2)
    ww = w * 12 * kT * f
    return x, ww / ww.sum()


def gilbert_damping(model, efermi, n=(0, 0, 1), mesh=(40, 40, 40), etas=(0.005, 0.01, 0.02, 0.05, 0.1),
                    forms=("exchange", "soc"), kT=0.0, chunk=2000, shift=(0.5, 0.5, 0.5), moment=None,
                    log=par.log):
    """Gilbert damping tensor vs broadening.

    model    TBModel whose magnetisation is along z (magex convention); it is rotated to n
    efermi   Fermi energy (eV)
    returns  dict with alpha (form -> array (neta, 2, 2)), intra/inter parts, spin moment, DOS
    """
    t0 = time.time()
    n = np.asarray(n, float) / np.linalg.norm(n)
    m = model.with_direction(n)
    HR = m.spinor_HR()
    forms = [f for f in forms if f != "soc" or m.has_soc()]
    if not m.has_soc():
        log("WARNING: the model has no spin-orbit coupling; damping vanishes (only the O(eta) "
            "artefact of the exchange-torque form remains). Add soc = 'atomic'.")
    TR = {f: torque_operators(m, n, f)[0] for f in forms}
    e1, e2 = _perp_axes(n)
    etas = np.atleast_1d(np.asarray(etas, float))
    xs, ws = _fermi_weights(kT)
    kpts = kgrid(mesh, shift)
    nk = len(kpts)
    nb = HR.shape[1]
    sig_n = sum(n[c] * np.kron(PAULI[c], np.eye(m.norb)) for c in range(3))
    acc = {f: np.zeros((len(etas), 2, 2, 2)) for f in forms}      # [eta, intra/inter, mu, nu]
    dos = np.zeros(len(etas))
    moment_acc = 0.0
    idx = par.split(int(np.ceil(nk / chunk)))
    for ib in idx:
        kb = kpts[ib * chunk:(ib + 1) * chunk]
        ph = np.exp(2j * np.pi * kb @ m.R.T)
        Hk = np.tensordot(ph, HR, axes=(1, 0))
        Hk = 0.5 * (Hk + np.conj(np.swapaxes(Hk, 1, 2)))
        eps, V = np.linalg.eigh(Hk)
        Vd = np.conj(np.swapaxes(V, 1, 2))
        # spin moment: occupied <sigma_n> (T = 0 step, or Fermi smearing)
        sn = np.einsum("kin,ij,kjn->kn", np.conj(V), sig_n, V).real
        occ = 1 / (1 + np.exp(np.clip((eps - efermi) / max(kT, 1e-4), -60, 60)))
        moment_acc += float(np.sum(sn * occ))
        Tk = {}
        for f in forms:
            Tk[f] = [Vd @ np.tensordot(ph, T, axes=(1, 0)) @ V for T in TR[f]]
        for ie, eta in enumerate(etas):
            for x, w in zip(xs, ws):
                A = eta / ((efermi + x - eps) ** 2 + eta**2)             # (nk, nb)
                dos[ie] += w * float(np.sum(A)) / np.pi
                for f in forms:
                    T1, T2 = Tk[f]
                    for a, Ta in enumerate((T1, T2)):
                        for b, Tb in enumerate((T1, T2)):
                            # sum_nm Ta_nm A_m Tb_mn A_n
                            M = np.einsum("knm,km,kmn,kn->k", Ta, A, Tb, A, optimize=True).real
                            diag = np.einsum("knn,kn,knn,kn->k", Ta, A, Tb, A).real
                            acc[f][ie, 0, a, b] += w * diag.sum()
                            acc[f][ie, 1, a, b] += w * (M - diag).sum()
    for f in forms:
        acc[f] = par.allreduce_sum(acc[f])
    dos = par.allreduce_sum(dos) / nk
    moment_acc = float(par.allreduce_sum(np.array(moment_acc))) / nk
    M = abs(moment_acc) if moment is None else float(moment)
    pref = G_FACTOR / (np.pi * M * nk)
    res = dict(n=n.tolist(), e1=e1.tolist(), e2=e2.tolist(), eta_eV=etas.tolist(), mesh=list(mesh), kT_eV=kT,
               efermi=efermi, moment_muB=M, moment_from_bands=moment_acc, dos_per_eV=dos.tolist(),
               seconds=time.time() - t0, alpha={}, alpha_intra={}, alpha_inter={})
    for f in forms:
        a = acc[f] * pref
        res["alpha"][f] = (a[:, 0] + a[:, 1]).tolist()
        res["alpha_intra"][f] = a[:, 0].tolist()
        res["alpha_inter"][f] = a[:, 1].tolist()
    return res


def scalar(res, form):
    """Isotropic Gilbert damping (alpha_11 + alpha_22)/2 vs eta."""
    a = np.array(res["alpha"][form])
    return 0.5 * (a[:, 0, 0] + a[:, 1, 1])


def alpha_from_file(path, eta_meV, form="soc"):
    """Isotropic alpha at broadening eta (meV) from a damping_*.json file (log-log interpolation)."""
    with open(path) as f:
        res = json.load(f)
    eta = np.array(res["eta_eV"]) * 1e3
    a = scalar(res, form if form in res["alpha"] else list(res["alpha"])[0])
    return float(np.exp(np.interp(np.log(eta_meV), np.log(eta), np.log(a))))


def resolve_alpha(value, base="."):
    """alpha given as a number or as {from = "damping_n001.json", eta_meV = 50, form = "soc"}."""
    if isinstance(value, dict):
        p = value["from"]
        p = p if os.path.isabs(p) else os.path.join(base, p)
        return alpha_from_file(p, float(value.get("eta_meV", 50.0)), value.get("form", "soc"))
    return float(value)


def report(res, log=par.log):
    log(f"Gilbert damping, magnetisation n = {np.round(res['n'], 4).tolist()}, k-mesh {res['mesh']}, "
        f"spin moment {res['moment_muB']:.3f} mu_B (bands), {res['seconds']:.1f} s")
    forms = list(res["alpha"])
    hdr = "  eta (meV)  " + "  ".join(f"{f + ' total':>14s}{f + ' intra':>14s}{f + ' inter':>14s}" for f in forms)
    log(hdr)
    for i, eta in enumerate(res["eta_eV"]):
        row = f"  {eta * 1e3:9.2f}  "
        for f in forms:
            tot = scalar(res, f)[i]
            ia = np.array(res["alpha_intra"][f])[i]
            ie = np.array(res["alpha_inter"][f])[i]
            row += f"{tot:14.5e}{0.5 * (ia[0, 0] + ia[1, 1]):14.5e}{0.5 * (ie[0, 0] + ie[1, 1]):14.5e}"
        log(row)


def plot(res, path):
    from .plotstyle import OKABE_ITO, mpl, savefig
    plt = mpl()
    fig, ax = plt.subplots(figsize=(4.4, 3.4), constrained_layout=True)
    eta = np.array(res["eta_eV"]) * 1e3
    for k, f in enumerate(res["alpha"]):
        ia = np.array(res["alpha_intra"][f])
        ie = np.array(res["alpha_inter"][f])
        ax.loglog(eta, scalar(res, f), "-o", color=OKABE_ITO[k], label=f"total ({f} torque)")
        ax.loglog(eta, 0.5 * (ia[:, 0, 0] + ia[:, 1, 1]), "--", color=OKABE_ITO[k], lw=1, label="intraband")
        inter = 0.5 * (ie[:, 0, 0] + ie[:, 1, 1])
        ax.loglog(eta, np.where(inter > 0, inter, np.nan), ":", color=OKABE_ITO[k], lw=1.2, label="interband")
    ax.set_xlabel(r"broadening $\eta$ (meV)")
    ax.set_ylabel(r"Gilbert damping $\alpha$")
    ax.legend(fontsize=7)
    savefig(fig, path)
    plt.close(fig)


# ----------------------------------------------------------------------------- driver
def run_damping(cfg):
    """[damping] section of a magex input (uses [input], [magnetic], soc settings of [calc])."""
    from .model import load_wannier90
    from .orbitals import L_OF, XI_DEFAULT
    d = dict(dict(kmesh=[40, 40, 40], eta_meV=[2, 5, 10, 20, 50, 100, 200], forms=["soc"],
                  directions=[[0, 0, 1]], kT_meV=0.0, chunk=2000, moment=None, soc=None), **cfg.get("damping", {}))
    c = cfg.get("calc", {})
    outdir = c.get("outdir", "magex_results")
    if par.is_root():
        os.makedirs(outdir, exist_ok=True)
    model = load_wannier90(cfg["input"])
    soc = d["soc"] or c.get("soc", "auto")
    collinear = model.info["source"] == "wannier90-collinear"
    if soc == "auto":
        soc = "atomic" if collinear else ("input" if model.has_soc() else "atomic")
    if soc == "replace":
        model.S[:] = 0
        model.H0 = model.H0.real.astype(complex)
        model.X = model.X.real.astype(complex)
        soc = "atomic"
    if soc == "atomic":
        xi = None
        if cfg.get("soc_xi"):
            xi = dict(XI_DEFAULT)
            for k, v in cfg["soc_xi"].items():
                el, sh = k.split(":")
                xi[(el, L_OF[sh])] = float(v)
        model = model.add_atomic_soc(xi)
        par.log("atomic SOC added:", {f"{k[0]}:{'spdf'[k[1]]}": v for k, v in model.info["soc_xi"].items()})
    if not collinear:
        from .driver import native_direction
        mag = [a for a, s in enumerate(model.symbols) if s in cfg["magnetic"]["elements"]]
        model = model.rotated(rotation_to(native_direction(model, mag)).T)
    ef = float(cfg["input"].get("efermi", 0.0))
    out = []
    for nd in d["directions"]:
        res = gilbert_damping(model, ef, nd, tuple(d["kmesh"]), np.array(d["eta_meV"], float) * 1e-3,
                              tuple(d["forms"]), float(d["kT_meV"]) * 1e-3, int(d["chunk"]), moment=d["moment"])
        if par.is_root():
            report(res)
            tag = "".join(f"{int(round(x))}" if abs(x - round(x)) < 1e-9 else f"{x:.2f}" for x in nd)
            with open(os.path.join(outdir, f"damping_n{tag}.json"), "w") as f:
                json.dump(res, f, indent=1)
            try:
                plot(res, os.path.join(outdir, f"damping_n{tag}.png"))
            except Exception as e:  # pragma: no cover
                par.log("plot skipped:", e)
        out.append(res)
    return out

"""Micromagnetic system: per-cell material parameters, effective field and energies.

Continuum energy density (SI)

  w = A |grad m|^2
    + D_int [ m_z div m - (m.grad) m_z ]                  (interfacial, C_nv, normal z)
    + D_bulk m.(curl m)                                    (bulk, B20)
    - Ku1 (m.u)^2 - Ku2 (m.u)^4                            (uniaxial)
    + Kc1 (a1^2 a2^2 + a2^2 a3^2 + a3^2 a1^2)              (cubic, a_k = m.c_k)
    - Ms m.B_ext - (mu0/2) Ms m.H_d                         (Zeeman, demag)

Effective field B = -(1/Ms) dw/dm (tesla).  Free surfaces carry the chiral boundary
condition  2A d_n m = D_int [ (m.n) z - m_z n ] + D_bulk (n x m)  (derived from the surface
term of the variation), implemented with ghost cells so that skyrmions and chiral edge
canting at film edges are correct.  Exchange between cells with different A uses the
harmonic mean 2 A_i A_j / (A_i + A_j) (OOMMF/mumax convention), scaled by an optional
inter-region factor.

Spin-orbit torques enter through B_SOT = B_DL (m x sigma) + B_FL sigma with
B_DL = hbar theta_SH J / (2 e Ms t), giving the damping-like torque -gamma B_DL m x (m x sigma).
"""
from __future__ import annotations

import numpy as np

from .constants import E_CHARGE, HBAR, MU0
from .mesh import Mesh

PARAMS = ("Ms", "A", "D_int", "D_bulk", "Ku1", "Ku2", "Kc1", "alpha")


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class Micromagnet:
    def __init__(self, mesh: Mesh, demag=True, workers=-1, log=None):
        self.mesh = mesh
        sh = mesh.n
        for p in PARAMS:
            setattr(self, p, np.zeros(sh))
        self.alpha[:] = 0.02
        self.anisU = np.zeros(sh + (3,))
        self.anisU[..., 2] = 1.0
        self.anisC1 = np.zeros(sh + (3,))
        self.anisC1[..., 0] = 1.0
        self.anisC2 = np.zeros(sh + (3,))
        self.anisC2[..., 1] = 1.0
        self.region = np.full(sh, -1, int)
        self.region_names = []
        self.B_ext = np.zeros(3)
        self.B_ext_cell = None                  # optional (nx,ny,nz,3) static non-uniform field
        self.B_func = None                      # optional callable t -> (3,) added to B_ext
        self.B_local = []                       # list of (mask (nx,ny,nz), callable t -> (3,)) local fields
        self.sot = None                         # dict(J, theta, sigma, fl_ratio, thickness)
        self.stt = None                         # dict(type="zhang-li", J=[..], P, beta) | dict(type="slonczewski", ...)
        self.exchange_scale = {}                # {(r1, r2): factor} inter-region exchange scaling
        self.exchange_axis_scale = np.ones(3)   # anisotropic stiffness A_x, A_y, A_z = A * scale (altermagnets)
        self.grain = None                       # (nx,ny,nz) grain ids; links across grains scaled by gb_scale
        self.gb_scale = 1.0
        self.rkky = []                          # [(k1, k2, J1, J2)] interlayer coupling (J/m^2) across a spacer
        self.oersted = None                     # static Oersted field (nx,ny,nz,3), T
        self.use_demag = bool(demag)
        self.workers = workers
        self._demag = None
        self.log = log
        self.t = 0.0
        self._links = {}
        self._version = 0

    # ------------------------------------------------------------------ set-up
    def add_region(self, mask, name="region", Ms=0.0, A=0.0, D=0.0, dmi="interfacial", Ku1=0.0, Ku2=0.0,
                   anisU=(0, 0, 1), Kc1=0.0, anisC1=(1, 0, 0), anisC2=(0, 1, 0), alpha=0.02, **_):
        """Assign a material to the cells in mask (later regions override earlier ones)."""
        mask = np.asarray(mask, bool)
        rid = len(self.region_names)
        self.region_names.append(name)
        self.region[mask] = rid
        self.Ms[mask] = Ms
        self.A[mask] = A
        self.D_int[mask] = D if dmi == "interfacial" else 0.0
        self.D_bulk[mask] = D if dmi == "bulk" else 0.0
        self.Ku1[mask] = Ku1
        self.Ku2[mask] = Ku2
        self.Kc1[mask] = Kc1
        self.alpha[mask] = alpha
        self.anisU[mask] = _unit(anisU)
        c1 = _unit(anisC1)
        c2 = _unit(np.asarray(anisC2, float) - np.dot(anisC2, c1) * c1)
        self.anisC1[mask] = c1
        self.anisC2[mask] = c2
        self.invalidate()
        return rid

    @property
    def magnetic(self):
        return self.Ms > 0

    @property
    def demag(self):
        if self._demag is None and self.use_demag:
            from .demag import Demag
            self._demag = Demag(self.mesh, self.workers, log=self.log)
        return self._demag

    def set_field(self, B):
        self.B_ext = np.asarray(B, float).reshape(3)

    def normalize(self, m):
        n = np.linalg.norm(m, axis=-1, keepdims=True)
        out = np.divide(m, n, out=np.zeros_like(m), where=n > 0)
        out[~self.magnetic] = 0.0
        return out

    # ------------------------------------------------------------------ neighbours with BCs
    def _link(self, axis, step):
        """Static link data for one direction (cached): exists, Alink, missing, ci, cb."""
        key = (axis, step, id(self.Ms), self._version)
        c = self._links.get(key)
        if c is not None:
            return c
        mesh = self.mesh
        Ms_nb = np.roll(self.Ms, -step, axis=axis)
        A_nb = np.roll(self.A, -step, axis=axis)
        reg_nb = np.roll(self.region, -step, axis=axis)
        exists = Ms_nb > 0
        if not mesh.pbc[axis]:
            edge = [slice(None)] * 3
            edge[axis] = -1 if step > 0 else 0
            exists[tuple(edge)] = False
        A = self.A
        with np.errstate(invalid="ignore", divide="ignore"):
            Alink = np.where(exists & (A + A_nb > 0), 2 * A * A_nb / (A + A_nb), 0.0)
            ci = np.where(A > 0, self.D_int / (2 * A), 0.0)[..., None]
            cb = np.where(A > 0, self.D_bulk / (2 * A), 0.0)[..., None]
        for (r1, r2), sc in self.exchange_scale.items():
            sel = ((self.region == r1) & (reg_nb == r2)) | ((self.region == r2) & (reg_nb == r1))
            Alink = np.where(sel, Alink * sc, Alink)
        if self.grain is not None and self.gb_scale != 1.0:
            gb = self.grain != np.roll(self.grain, -step, axis=axis)
            Alink = np.where(gb & exists, Alink * self.gb_scale, Alink)
        missing = ~exists & self.magnetic
        Alink = np.where(missing, A, Alink)          # ghost link carries the cell's own A
        chiral = bool(np.any(ci[missing]) or np.any(cb[missing]))
        c = (exists, Alink, missing, ci, cb, bool(np.any(missing)), chiral)
        self._links = {k: v for k, v in self._links.items() if k[3] == self._version}
        self._links[key] = c
        return c

    def invalidate(self):
        """Call after changing material arrays by hand (Ms, A, D, regions)."""
        self._version += 1
        self._links = {}

    def _neighbour(self, m, axis, step):
        """Neighbour values of m along axis (step=+1/-1) and the exchange constant of the link.

        Missing neighbours (open boundary or non-magnetic cell) are replaced by ghost values
        m + step*d*(d_n m) obeying the chiral boundary condition.
        """
        mesh = self.mesh
        if mesh.n[axis] == 1 and not mesh.pbc[axis]:
            return m, None                                 # no gradients along a single-cell axis
        exists, Alink, missing, ci, cb, anymiss, chiral = self._link(axis, step)
        nb = np.roll(m, -step, axis=axis)
        if anymiss:
            if chiral:
                n = np.zeros(3)
                n[axis] = step
                mn = m[..., axis:axis + 1] * step
                dmdn = ci * (mn * np.array([0, 0, 1.0]) - m[..., 2:3] * n) + cb * np.cross(n, m)
                ghost = m + mesh.d[axis] * dmdn
            else:
                ghost = m
            nb = np.where(missing[..., None], ghost, nb)
        return nb, Alink

    def _stencil(self, m):
        out = []
        for axis in range(3):
            p, Ap = self._neighbour(m, axis, +1)
            q, Aq = self._neighbour(m, axis, -1)
            out.append((p, Ap, q, Aq))
        return out

    # ------------------------------------------------------------------ fields (tesla)
    def fields(self, m, t=None, terms=None):
        """Dict of the effective-field contributions (T), each (nx,ny,nz,3)."""
        Ms = self.Ms
        inv = np.divide(1.0, Ms, out=np.zeros_like(Ms), where=Ms > 0)[..., None]
        st = self._stencil(m)
        F = {}
        lap = np.zeros_like(m)
        grad = []
        for axis, (p, Ap, q, Aq) in enumerate(st):
            if Ap is None:
                grad.append(np.zeros_like(m))
                continue
            d2 = self.mesh.d[axis] ** 2
            lap += self.exchange_axis_scale[axis] * (Ap[..., None] * (p - m) + Aq[..., None] * (q - m)) / d2
            grad.append((p - q) / (2 * self.mesh.d[axis]))
        F["exchange"] = 2 * inv * lap
        if np.any(self.D_int):
            gx, gy = grad[0], grad[1]
            Bd = np.stack([gx[..., 2], gy[..., 2], -gx[..., 0] - gy[..., 1]], -1)
            F["dmi"] = 2 * (self.D_int * inv[..., 0])[..., None] * Bd
        if np.any(self.D_bulk):
            gx, gy, gz = grad
            curl = np.stack([gy[..., 2] - gz[..., 1], gz[..., 0] - gx[..., 2], gx[..., 1] - gy[..., 0]], -1)
            Bb = -2 * (self.D_bulk * inv[..., 0])[..., None] * curl
            F["dmi"] = F.get("dmi", 0) + Bb
        if np.any(self.Ku1) or np.any(self.Ku2):
            mu = np.einsum("...a,...a->...", m, self.anisU)
            F["anisotropy"] = ((2 * self.Ku1 * mu + 4 * self.Ku2 * mu**3) * inv[..., 0])[..., None] * self.anisU
        if np.any(self.Kc1):
            c1, c2 = self.anisC1, self.anisC2
            c3 = np.cross(c1, c2)
            a1, a2, a3 = (np.einsum("...a,...a->...", m, c) for c in (c1, c2, c3))
            pre = (-2 * self.Kc1 * inv[..., 0])[..., None]
            F["cubic"] = pre * ((a1 * (a2**2 + a3**2))[..., None] * c1 + (a2 * (a1**2 + a3**2))[..., None] * c2
                                + (a3 * (a1**2 + a2**2))[..., None] * c3)
        if self.use_demag:
            F["demag"] = MU0 * self.demag.field(Ms[..., None] * m)
        Bz = self.B_ext.copy()
        if self.B_func is not None:
            Bz = Bz + np.asarray(self.B_func(self.t if t is None else t), float)
        Bzee = np.broadcast_to(Bz, m.shape).copy()
        if self.B_ext_cell is not None:
            Bzee = Bzee + self.B_ext_cell
        for mask, fn in self.B_local:
            Bzee = Bzee + mask[..., None] * np.asarray(fn(self.t if t is None else t), float)
        if self.oersted is not None:
            Bzee = Bzee + self.oersted
        F["zeeman"] = Bzee
        if self.rkky:
            Br = np.zeros_like(m)
            dz = self.mesh.d[2]
            for (k1, k2, J1, J2) in self.rkky:
                m1, m2 = m[:, :, k1], m[:, :, k2]
                d12 = np.einsum("...a,...a->...", m1, m2)[..., None]
                for ka, mo, Msa in ((k1, m2, Ms[:, :, k1]), (k2, m1, Ms[:, :, k2])):
                    inv = np.divide(1.0, Msa, out=np.zeros_like(Msa), where=Msa > 0)[..., None]
                    Br[:, :, ka] += (J1 + 2 * J2 * d12) * mo * inv / dz
            F["rkky"] = Br
        mag = self.magnetic[..., None]
        for k in F:
            F[k] = np.where(mag, F[k], 0.0)
        return F

    def sot_field(self, m):
        if not self.sot:
            return 0.0
        s = self.sot
        sigma = _unit(s.get("sigma", [0, 1, 0]))
        t_fm = s.get("thickness", self.mesh.n[2] * self.mesh.d[2])
        Ms = self.Ms
        inv = np.divide(1.0, Ms, out=np.zeros_like(Ms), where=Ms > 0)
        bdl = HBAR * s.get("theta", 0.1) * s.get("J", 0.0) / (2 * E_CHARGE * t_fm) * inv
        if callable(s.get("J_func")):
            bdl = bdl * s["J_func"](self.t)
        if "pulse_ns" in s:                       # [t0, width] in ns: current on only inside the window
            t0, w = s["pulse_ns"]
            if not (t0 * 1e-9 <= self.t < (t0 + w) * 1e-9):
                return np.zeros(m.shape)
        B = bdl[..., None] * np.cross(m, sigma) + (s.get("fl_ratio", 0.0) * bdl)[..., None] * sigma
        return np.where(self.magnetic[..., None], B, 0.0)

    def effective_field(self, m, t=None):
        F = self.fields(m, t)
        B = sum(F.values())
        return B

    # ------------------------------------------------------------------ energies (J)
    _FACTOR = {"exchange": 0.5, "dmi": 0.5, "anisotropy": None, "cubic": 0.25, "demag": 0.5, "zeeman": 1.0,
               "rkky": None}

    def energies(self, m, F=None):
        F = F or self.fields(m)
        V = self.mesh.V
        E = {}
        for k, B in F.items():
            dens = np.einsum("...a,...a->...", m, B) * self.Ms
            if k == "anisotropy":
                mu = np.einsum("...a,...a->...", m, self.anisU)
                e = -(self.Ku1 * mu**2 + self.Ku2 * mu**4)
                E[k] = float(np.sum(e[self.magnetic]) * V)
            elif k == "rkky":
                dA = self.mesh.d[0] * self.mesh.d[1]
                Er = 0.0
                for (k1, k2, J1, J2) in self.rkky:
                    both = self.magnetic[:, :, k1] & self.magnetic[:, :, k2]
                    d12 = np.einsum("...a,...a->...", m[:, :, k1], m[:, :, k2])[both]
                    Er += float(np.sum(-J1 * d12 - J2 * d12**2) * dA)
                E[k] = Er
            else:
                E[k] = float(-self._FACTOR[k] * np.sum(dens) * V)
        E["total"] = sum(E.values())
        return E

    def average(self, m):
        mag = self.magnetic
        w = self.Ms[mag]
        return (m[mag] * w[:, None]).sum(0) / w.sum()

    def torque(self, m, B=None):
        B = self.effective_field(m) if B is None else B
        return np.cross(m, B)

    def summary(self):
        mag = self.magnetic
        lines = [repr(self.mesh), f"  magnetic cells {int(mag.sum())} / {mag.size}, regions: {self.region_names}"]
        if mag.any():
            for p in ("Ms", "A", "D_int", "D_bulk", "Ku1", "Kc1", "alpha"):
                v = getattr(self, p)[mag]
                if np.any(v):
                    lines.append(f"  {p:7s} {v.min():.4g} .. {v.max():.4g}")
            A, Ms = self.A[mag].max(), self.Ms[mag].max()
            lex = np.sqrt(2 * A / (MU0 * Ms**2)) if Ms > 0 else np.inf
            lines.append(f"  exchange length l_ex = {lex * 1e9:.2f} nm  (cell {max(self.mesh.d[:2]) * 1e9:g} nm)")
            K = self.Ku1[mag].max()
            if K > 0:
                Keff = K - 0.5 * MU0 * Ms**2 if self.use_demag else K
                lines.append(f"  wall width pi sqrt(A/K) = {np.pi * np.sqrt(A / K) * 1e9:.2f} nm, "
                             f"K_eff = Ku1 - mu0 Ms^2/2 = {Keff:.3g} J/m^3")
                D = max(abs(self.D_int[mag]).max(), abs(self.D_bulk[mag]).max())
                if D > 0 and Keff > 0:
                    Dc = 4 * np.sqrt(A * Keff) / np.pi
                    lines.append(f"  |D| / D_c = {D / Dc:.3f}  (D_c = 4 sqrt(A K_eff)/pi = {Dc * 1e3:.3f} mJ/m^2;"
                                 " > 1 -> spin spirals)")
            if max(self.mesh.d[:2]) > lex:
                lines.append("  WARNING: cell larger than the exchange length")
        return "\n".join(lines)

    # ------------------------------------------------------------------ initial states
    def init(self, kind="uniform", direction=(0, 0, 1), seed=1, **p):
        X, Y, Z = self.mesh.coords()
        nm = 1e-9
        m = np.zeros(self.mesh.n + (3,))
        if kind == "uniform":
            m[...] = _unit(direction)
        elif kind == "random":
            m = np.random.default_rng(seed).normal(size=m.shape)
        elif kind in ("skyrmion", "skyrmions"):
            centers = p.get("centers_nm", [[0, 0]])
            R = p.get("radius_nm", 10) * nm
            w = p.get("wall_nm", R / nm / 2) * nm
            core = p.get("core", -1)
            helicity = np.deg2rad(p.get("helicity_deg", 0.0))       # 0 Neel, 90 Bloch
            m[...] = [0, 0, -core]
            for c in centers:
                x, y = X - c[0] * nm, Y - c[1] * nm
                r = np.sqrt(x**2 + y**2)
                th = 2 * np.arctan2(np.sinh(R / w), np.sinh(r / w))  # pi at centre, 0 far away
                phi = np.arctan2(y, x) + helicity
                sel = r < 3 * R + 4 * w
                mz = -core * np.cos(th)
                m[sel] = np.stack([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), mz], -1)[sel]
        elif kind == "vortex":
            r = np.sqrt(X**2 + Y**2)
            rc = p.get("core_nm", 10) * nm
            chir = p.get("circulation", 1)
            pol = p.get("polarity", 1)
            mz = pol * np.exp(-(r / rc) ** 2)
            pf = np.sqrt(1 - mz**2)
            m = np.stack([-chir * pf * Y / np.maximum(r, 1e-30), chir * pf * X / np.maximum(r, 1e-30), mz], -1)
        elif kind == "domain_wall":
            x0 = p.get("x0_nm", 0) * nm
            w = p.get("width_nm", 5) * nm
            th = 2 * np.arctan(np.exp((X - x0) / w))
            phi = np.deg2rad(p.get("phi_deg", 0))          # 0: Neel (in xz), 90: Bloch
            axis = p.get("axis", "z")
            if axis == "z":
                m = np.stack([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)], -1)
            else:
                m = np.stack([np.cos(th), np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi)], -1)
        elif kind == "helix":
            lam = p.get("period_nm", 50) * nm
            k = 2 * np.pi * np.asarray(p.get("q_dir", [1, 0, 0]), float) / lam
            ph = k[0] * X + k[1] * Y + k[2] * Z
            m = np.stack([np.sin(ph), np.zeros_like(ph), np.cos(ph)], -1)
        elif kind == "file":
            from .io import read_ovf
            m = read_ovf(p["path"])[0]
        else:
            raise ValueError(f"unknown initial state '{kind}'")
        return self.normalize(m)

"""Fused multithreaded kernel for the local micromagnetic fields (exchange, interfacial/bulk DMI with the
chiral ghost-cell boundary condition, uniaxial and cubic anisotropy).  Bit-for-bit the same stencil as
``Micromagnet.fields`` (tests/test_fast.py), without the array temporaries of the NumPy version.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
    HAVE = True
except Exception:  # pragma: no cover
    HAVE = False

if HAVE:
    @njit(parallel=True, cache=True)
    def _local(m, Ms, Alink, exists, use_axis, d, ascale, ci, cb, Dint, Dbulk, Ku1, Ku2, U, Kc1, C1, C2,
               Bex, Bdmi, Ban, Bcub, do_dmi, do_ani, do_cub):
        nx, ny, nz = Ms.shape
        for ixp in prange(nx):
            ix = np.int64(ixp)          # parfor index is unsigned: keep integer arithmetic signed
            for iy in range(ny):
                for iz in range(nz):
                    if Ms[ix, iy, iz] <= 0:
                        for c in range(3):
                            Bex[ix, iy, iz, c] = 0.0
                            Bdmi[ix, iy, iz, c] = 0.0
                            Ban[ix, iy, iz, c] = 0.0
                            Bcub[ix, iy, iz, c] = 0.0
                        continue
                    inv = 1.0 / Ms[ix, iy, iz]
                    m0 = m[ix, iy, iz, 0]
                    m1 = m[ix, iy, iz, 1]
                    m2 = m[ix, iy, iz, 2]
                    lap0 = lap1 = lap2 = 0.0
                    g = np.zeros((3, 3))           # g[axis, comp] = (p - q)/(2d)
                    for ax in range(3):
                        if not use_axis[ax]:
                            continue
                        pv = np.empty(3)
                        qv = np.empty(3)
                        for sgn in range(2):
                            step = 1 - 2 * sgn
                            fs = 1.0 - 2.0 * sgn
                            k = 2 * ax + sgn
                            j0, j1, j2 = ix, iy, iz
                            if ax == 0:
                                j0 = (ix + step) % nx
                            elif ax == 1:
                                j1 = (iy + step) % ny
                            else:
                                j2 = (iz + step) % nz
                            if exists[k, ix, iy, iz]:
                                v0, v1, v2 = m[j0, j1, j2, 0], m[j0, j1, j2, 1], m[j0, j1, j2, 2]
                            else:
                                # ghost: m + d * dm/dn,  dm/dn = ci [(m.n) z - m_z n] + cb (n x m)
                                c_i = ci[ix, iy, iz]
                                c_b = cb[ix, iy, iz]
                                n0 = fs if ax == 0 else 0.0
                                n1 = fs if ax == 1 else 0.0
                                n2 = fs if ax == 2 else 0.0
                                mn = (m0 * n0 + m1 * n1 + m2 * n2)
                                dd0 = c_i * (-m2 * n0) + c_b * (n1 * m2 - n2 * m1)
                                dd1 = c_i * (-m2 * n1) + c_b * (n2 * m0 - n0 * m2)
                                dd2 = c_i * (mn - m2 * n2) + c_b * (n0 * m1 - n1 * m0)
                                v0, v1, v2 = m0 + d[ax] * dd0, m1 + d[ax] * dd1, m2 + d[ax] * dd2
                            Al = Alink[k, ix, iy, iz]
                            s_ = ascale[ax] / (d[ax] * d[ax])
                            lap0 += s_ * Al * (v0 - m0)
                            lap1 += s_ * Al * (v1 - m1)
                            lap2 += s_ * Al * (v2 - m2)
                            if sgn == 0:
                                pv[0], pv[1], pv[2] = v0, v1, v2
                            else:
                                qv[0], qv[1], qv[2] = v0, v1, v2
                        for c in range(3):
                            g[ax, c] = (pv[c] - qv[c]) / (2 * d[ax])
                    Bex[ix, iy, iz, 0] = 2 * inv * lap0
                    Bex[ix, iy, iz, 1] = 2 * inv * lap1
                    Bex[ix, iy, iz, 2] = 2 * inv * lap2
                    if do_dmi:
                        di = 2 * Dint[ix, iy, iz] * inv
                        db = -2 * Dbulk[ix, iy, iz] * inv
                        Bdmi[ix, iy, iz, 0] = di * g[0, 2] + db * (g[1, 2] - g[2, 1])
                        Bdmi[ix, iy, iz, 1] = di * g[1, 2] + db * (g[2, 0] - g[0, 2])
                        Bdmi[ix, iy, iz, 2] = di * (-g[0, 0] - g[1, 1]) + db * (g[0, 1] - g[1, 0])
                    if do_ani:
                        u0, u1, u2 = U[ix, iy, iz, 0], U[ix, iy, iz, 1], U[ix, iy, iz, 2]
                        mu = m0 * u0 + m1 * u1 + m2 * u2
                        f = (2 * Ku1[ix, iy, iz] * mu + 4 * Ku2[ix, iy, iz] * mu**3) * inv
                        Ban[ix, iy, iz, 0] = f * u0
                        Ban[ix, iy, iz, 1] = f * u1
                        Ban[ix, iy, iz, 2] = f * u2
                    if do_cub:
                        c10, c11, c12 = C1[ix, iy, iz, 0], C1[ix, iy, iz, 1], C1[ix, iy, iz, 2]
                        c20, c21, c22 = C2[ix, iy, iz, 0], C2[ix, iy, iz, 1], C2[ix, iy, iz, 2]
                        c30 = c11 * c22 - c12 * c21
                        c31 = c12 * c20 - c10 * c22
                        c32 = c10 * c21 - c11 * c20
                        a1 = m0 * c10 + m1 * c11 + m2 * c12
                        a2 = m0 * c20 + m1 * c21 + m2 * c22
                        a3 = m0 * c30 + m1 * c31 + m2 * c32
                        pre = -2 * Kc1[ix, iy, iz] * inv
                        f1 = pre * a1 * (a2 * a2 + a3 * a3)
                        f2 = pre * a2 * (a1 * a1 + a3 * a3)
                        f3 = pre * a3 * (a1 * a1 + a2 * a2)
                        Bcub[ix, iy, iz, 0] = f1 * c10 + f2 * c20 + f3 * c30
                        Bcub[ix, iy, iz, 1] = f1 * c11 + f2 * c21 + f3 * c31
                        Bcub[ix, iy, iz, 2] = f1 * c12 + f2 * c22 + f3 * c32


def local_fields(sysm, m):
    """dict(exchange, dmi?, anisotropy?, cubic?) from the fused kernel (NumPy arrays)."""
    mesh = sysm.mesh
    n = mesh.n
    Alink = np.zeros((6,) + n)
    exists = np.zeros((6,) + n, np.bool_)
    use = np.zeros(3, np.bool_)
    ci = cb = None
    for ax in range(3):
        if mesh.n[ax] == 1 and not mesh.pbc[ax]:
            continue
        use[ax] = True
        for sgn, step in enumerate((1, -1)):
            ex, Al, missing, ci_, cb_, anymiss, chiral = sysm._link(ax, step)
            Alink[2 * ax + sgn] = Al
            exists[2 * ax + sgn] = ex
            ci, cb = ci_[..., 0], cb_[..., 0]
    if ci is None:
        ci = np.zeros(n)
        cb = np.zeros(n)
    shp = n + (3,)
    Bex, Bdmi, Ban, Bcub = (np.empty(shp) for _ in range(4))
    do_dmi = bool(np.any(sysm.D_int) or np.any(sysm.D_bulk))
    do_ani = bool(np.any(sysm.Ku1) or np.any(sysm.Ku2))
    do_cub = bool(np.any(sysm.Kc1))
    _local(np.ascontiguousarray(m, float), sysm.Ms, Alink, exists, use, np.asarray(mesh.d, float),
           np.asarray(sysm.exchange_axis_scale, float), np.ascontiguousarray(ci), np.ascontiguousarray(cb),
           sysm.D_int, sysm.D_bulk, sysm.Ku1, sysm.Ku2, np.ascontiguousarray(sysm.anisU), sysm.Kc1,
           np.ascontiguousarray(sysm.anisC1), np.ascontiguousarray(sysm.anisC2), Bex, Bdmi, Ban, Bcub,
           do_dmi, do_ani, do_cub)
    F = {"exchange": Bex}
    if do_dmi:
        F["dmi"] = Bdmi
    if do_ani:
        F["anisotropy"] = Ban
    if do_cub:
        F["cubic"] = Bcub
    return F

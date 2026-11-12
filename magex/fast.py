"""Multithreaded (numba, prange) kernels for the atomistic spin model.

The 3N x 3N sparse coupling matrix is converted once into a per-site neighbour list with 3x3 blocks,
so the effective field, a whole Heun / Depondt step and a whole Monte Carlo colour sweep are single
fused loops over sites (no temporaries, no Python overhead, all cores).  Results are identical to the
NumPy path to round-off (tests/test_fast.py).

Covered in the kernels: exchange tensors (isotropic + DMI + symmetric anisotropic), single-ion tensors
(incl. folded Neel surface terms), Zeeman, on-site cubic and higher-order uniaxial terms, pinned spins,
site temperatures and damping.  Anything else (dipolar, biquadratic, four-spin, quadruplets, torques,
user fields) is added as an extra field computed by the NumPy code, so every model still runs.
"""
from __future__ import annotations

import numpy as np

try:
    import numba
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

MU_B = 5.7883818060e-2
K_B = 8.617333262e-2
GAMMA = 0.1760859630

if HAVE_NUMBA:
    @njit(cache=True)
    def _seed(s):
        np.random.seed(s)

    @njit(parallel=True, cache=True)
    def _seed_threads(seeds):
        # one draw per thread index seeds the per-thread generators deterministically
        for k in prange(len(seeds)):
            np.random.seed(seeds[k])

    @njit(inline="always")
    def _onsite_grad(i, x, y, z, A, Kc, C, K2, K3, u, has_c, has_u):
        """dE_onsite/de (meV) for site i at direction (x, y, z)."""
        gx = 2.0 * (A[i, 0, 0] * x + A[i, 0, 1] * y + A[i, 0, 2] * z)
        gy = 2.0 * (A[i, 1, 0] * x + A[i, 1, 1] * y + A[i, 1, 2] * z)
        gz = 2.0 * (A[i, 2, 0] * x + A[i, 2, 1] * y + A[i, 2, 2] * z)
        if has_c:
            a1 = C[0, 0] * x + C[0, 1] * y + C[0, 2] * z
            a2 = C[1, 0] * x + C[1, 1] * y + C[1, 2] * z
            a3 = C[2, 0] * x + C[2, 1] * y + C[2, 2] * z
            q1, q2, q3 = a1 * a1, a2 * a2, a3 * a3
            d1 = 2 * a1 * (q2 + q3) * Kc[i]
            d2 = 2 * a2 * (q1 + q3) * Kc[i]
            d3 = 2 * a3 * (q1 + q2) * Kc[i]
            gx += d1 * C[0, 0] + d2 * C[1, 0] + d3 * C[2, 0]
            gy += d1 * C[0, 1] + d2 * C[1, 1] + d3 * C[2, 1]
            gz += d1 * C[0, 2] + d2 * C[1, 2] + d3 * C[2, 2]
        if has_u:
            c = x * u[0] + y * u[1] + z * u[2]
            f = -4.0 * K2[i] * c**3 - 6.0 * K3[i] * c**5
            gx += f * u[0]
            gy += f * u[1]
            gz += f * u[2]
        return gx, gy, gz

    @njit(inline="always")
    def _onsite_E(i, x, y, z, A, Kc, C, K2, K3, u, has_c, has_u):
        E = 0.0
        v = (x, y, z)
        for a in range(3):
            for b in range(3):
                E += v[a] * A[i, a, b] * v[b]
        if has_c:
            a1 = C[0, 0] * x + C[0, 1] * y + C[0, 2] * z
            a2 = C[1, 0] * x + C[1, 1] * y + C[1, 2] * z
            a3 = C[2, 0] * x + C[2, 1] * y + C[2, 2] * z
            q1, q2, q3 = a1 * a1, a2 * a2, a3 * a3
            E += Kc[i] * (q1 * q2 + q2 * q3 + q3 * q1)
        if has_u:
            c = x * u[0] + y * u[1] + z * u[2]
            E += -K2[i] * c**4 - K3[i] * c**6
        return E

    @njit(inline="always")
    def _hfield(i, e, ptr, nbr, Jb):
        """sum_j J_ij e_j; Jb = (typ, table): neighbour k uses table[typ[k]] -- a 3x3 block (9 numbers),
        isotropic + DMI (4 numbers: J, Dx, Dy, Dz with J e + D x e... see neighbour_blocks) or isotropic (1)."""
        typ, T = Jb
        hx = 0.0
        hy = 0.0
        hz = 0.0
        w = T.shape[1]
        if w == 1:
            for k in range(ptr[i], ptr[i + 1]):
                j = nbr[k]
                Jv = T[typ[k], 0]
                hx += Jv * e[j, 0]
                hy += Jv * e[j, 1]
                hz += Jv * e[j, 2]
        elif w == 4:
            for k in range(ptr[i], ptr[i + 1]):
                j = nbr[k]
                t = typ[k]
                Jv, Dx, Dy, Dz = T[t, 0], T[t, 1], T[t, 2], T[t, 3]
                ex, ey, ez = e[j, 0], e[j, 1], e[j, 2]
                # block J I + antisymmetric part with e_i.J^A.e_j = D.(e_i x e_j): (J^A e_j) = e_j x D
                hx += Jv * ex + (ey * Dz - ez * Dy)
                hy += Jv * ey + (ez * Dx - ex * Dz)
                hz += Jv * ez + (ex * Dy - ey * Dx)
        else:
            for k in range(ptr[i], ptr[i + 1]):
                j = nbr[k]
                t = typ[k]
                ex, ey, ez = e[j, 0], e[j, 1], e[j, 2]
                hx += T[t, 0] * ex + T[t, 1] * ey + T[t, 2] * ez
                hy += T[t, 3] * ex + T[t, 4] * ey + T[t, 5] * ez
                hz += T[t, 6] * ex + T[t, 7] * ey + T[t, 8] * ez
        return hx, hy, hz

    # ------------------------------------------------------------------ fast parallel Gaussian RNG
    @njit(inline="always")
    def _next(st, c):
        x = st[c]
        x ^= x >> np.uint64(12)
        x ^= x << np.uint64(25)
        x ^= x >> np.uint64(27)
        st[c] = x
        return x * np.uint64(2685821657736338717)

    @njit(inline="always")
    def _unif(st, c):
        return (_next(st, c) >> np.uint64(11)) * (1.0 / 9007199254740992.0)

    # Marsaglia-Tsang ziggurat (128 layers) for the normal distribution
    def _zig_tables():
        dn = 3.442619855899
        tn = dn
        vn = 9.91256303526217e-3
        m1 = 2147483648.0
        kn = np.zeros(128)
        wn = np.zeros(128)
        fn = np.zeros(128)
        q = vn / np.exp(-0.5 * dn * dn)
        kn[0] = (dn / q) * m1
        kn[1] = 0.0
        wn[0] = q / m1
        wn[127] = dn / m1
        fn[0] = 1.0
        fn[127] = np.exp(-0.5 * dn * dn)
        for i in range(126, 0, -1):
            dn = np.sqrt(-2.0 * np.log(vn / dn + np.exp(-0.5 * dn * dn)))
            kn[i + 1] = (dn / tn) * m1
            tn = dn
            fn[i] = np.exp(-0.5 * dn * dn)
            wn[i] = dn / m1
        return kn, wn, fn

    ZKN, ZWN, ZFN = _zig_tables()

    @njit(inline="always")
    def _hz(st, c):
        return np.int64(np.int32(np.uint32(_next(st, c) >> np.uint64(32))))

    @njit(inline="always")
    def _znorm(st, c, kn, wn, fn):
        hz = _hz(st, c)
        iz = hz & 127
        if abs(hz) < kn[iz]:
            return hz * wn[iz]
        r = 3.442619855899
        while True:
            x = hz * wn[iz]
            if iz == 0:
                while True:
                    x = -np.log(max(_unif(st, c), 1e-300)) * 0.2904764
                    y = -np.log(max(_unif(st, c), 1e-300))
                    if y + y >= x * x:
                        break
                return r + x if hz > 0 else -r - x
            if fn[iz] + _unif(st, c) * (fn[iz - 1] - fn[iz]) < np.exp(-0.5 * x * x):
                return x
            hz = _hz(st, c)
            iz = hz & 127
            if abs(hz) < kn[iz]:
                return hz * wn[iz]

    @njit(parallel=True, cache=True)
    def _gauss3(out, st, scale, pinned):
        """out[i, :] = scale[i] * N(0, 1)^3 (ziggurat on xorshift64* streams over fixed chunks: results do
        not depend on the number of threads)."""
        n = out.shape[0]
        nc = st.size
        chunk = (n + nc - 1) // nc
        for c in prange(nc):
            for i in range(c * chunk, min(n, (c + 1) * chunk)):
                for comp in range(3):
                    g = _znorm(st, c, ZKN, ZWN, ZFN)
                    out[i, comp] = 0.0 if pinned[i] else scale[i] * g

    @njit(parallel=True, cache=True)
    def _field(e, out, ptr, nbr, Jb, A, mu, B, Kc, C, K2, K3, u, has_c, has_u, pinned, extra, has_extra):
        N = e.shape[0]
        for i in prange(N):
            if pinned[i]:
                out[i, 0] = 0.0
                out[i, 1] = 0.0
                out[i, 2] = 0.0
                continue
            hx, hy, hz = _hfield(i, e, ptr, nbr, Jb)
            if A.shape[0] > 0:
                gx, gy, gz = _onsite_grad(i, e[i, 0], e[i, 1], e[i, 2], A, Kc, C, K2, K3, u, has_c, has_u)
            else:
                gx = gy = gz = 0.0
            inv = 1.0 / (mu[i] * MU_B)
            out[i, 0] = (2 * hx - gx) * inv + B[0]
            out[i, 1] = (2 * hy - gy) * inv + B[1]
            out[i, 2] = (2 * hz - gz) * inv + B[2]
            if has_extra:
                out[i, 0] += extra[i, 0]
                out[i, 1] += extra[i, 1]
                out[i, 2] += extra[i, 2]

    @njit(inline="always")
    def _rhs(ex, ey, ez, bx, by, bz, a, prec):
        cx = ey * bz - ez * by
        cy = ez * bx - ex * bz
        cz = ex * by - ey * bx
        dx = ey * cz - ez * cy
        dy = ez * cx - ex * cz
        dz = ex * cy - ey * cx
        pre = -GAMMA / (1.0 + a * a)
        if prec:
            return pre * (cx + a * dx), pre * (cy + a * dy), pre * (cz + a * dz)
        return pre * a * dx, pre * a * dy, pre * a * dz

    @njit(parallel=True, cache=True)
    def _noise(b, sig, pinned):
        for i in prange(b.shape[0]):
            if pinned[i]:
                b[i, 0] = 0.0
                b[i, 1] = 0.0
                b[i, 2] = 0.0
            else:
                b[i, 0] = sig[i] * np.random.standard_normal()
                b[i, 1] = sig[i] * np.random.standard_normal()
                b[i, 2] = sig[i] * np.random.standard_normal()

    @njit(parallel=True, cache=True)
    def _heun_pred(e, Bf, b, alpha, dt, prec, ep, k1):
        for i in prange(e.shape[0]):
            fx, fy, fz = _rhs(e[i, 0], e[i, 1], e[i, 2], Bf[i, 0] + b[i, 0], Bf[i, 1] + b[i, 1], Bf[i, 2] + b[i, 2],
                              alpha[i], prec)
            k1[i, 0], k1[i, 1], k1[i, 2] = fx, fy, fz
            x, y, z = e[i, 0] + dt * fx, e[i, 1] + dt * fy, e[i, 2] + dt * fz
            n = 1.0 / np.sqrt(x * x + y * y + z * z)
            ep[i, 0], ep[i, 1], ep[i, 2] = x * n, y * n, z * n

    @njit(parallel=True, cache=True)
    def _heun_corr(e, ep, Bf, b, alpha, dt, prec, k1):
        for i in prange(e.shape[0]):
            fx, fy, fz = _rhs(ep[i, 0], ep[i, 1], ep[i, 2], Bf[i, 0] + b[i, 0], Bf[i, 1] + b[i, 1],
                              Bf[i, 2] + b[i, 2], alpha[i], prec)
            x = e[i, 0] + 0.5 * dt * (k1[i, 0] + fx)
            y = e[i, 1] + 0.5 * dt * (k1[i, 1] + fy)
            z = e[i, 2] + 0.5 * dt * (k1[i, 2] + fz)
            n = 1.0 / np.sqrt(x * x + y * y + z * z)
            e[i, 0], e[i, 1], e[i, 2] = x * n, y * n, z * n

    @njit(inline="always")
    def _rot(x, y, z, wx, wy, wz):
        th = np.sqrt(wx * wx + wy * wy + wz * wz)
        if th < 1e-14:
            return x, y, z
        kx, ky, kz = wx / th, wy / th, wz / th
        c, s = np.cos(th), np.sin(th)
        kd = kx * x + ky * y + kz * z
        cx = ky * z - kz * y
        cy = kz * x - kx * z
        cz = kx * y - ky * x
        return (x * c + cx * s + kx * kd * (1 - c), y * c + cy * s + ky * kd * (1 - c), z * c + cz * s + kz * kd * (1 - c))

    @njit(inline="always")
    def _omega(ex, ey, ez, bx, by, bz, a, prec):
        # de/dt = w x e with w = e x f for the tangent rhs f
        fx, fy, fz = _rhs(ex, ey, ez, bx, by, bz, a, prec)
        return ey * fz - ez * fy, ez * fx - ex * fz, ex * fy - ey * fx

    @njit(parallel=True, cache=True)
    def _dep_pred(e, Bf, b, alpha, dt, prec, ep, w1):
        for i in prange(e.shape[0]):
            wx, wy, wz = _omega(e[i, 0], e[i, 1], e[i, 2], Bf[i, 0] + b[i, 0], Bf[i, 1] + b[i, 1],
                                Bf[i, 2] + b[i, 2], alpha[i], prec)
            w1[i, 0], w1[i, 1], w1[i, 2] = wx, wy, wz
            ep[i, 0], ep[i, 1], ep[i, 2] = _rot(e[i, 0], e[i, 1], e[i, 2], dt * wx, dt * wy, dt * wz)

    @njit(parallel=True, cache=True)
    def _dep_corr(e, ep, Bf, b, alpha, dt, prec, w1):
        for i in prange(e.shape[0]):
            wx, wy, wz = _omega(ep[i, 0], ep[i, 1], ep[i, 2], Bf[i, 0] + b[i, 0], Bf[i, 1] + b[i, 1],
                                Bf[i, 2] + b[i, 2], alpha[i], prec)
            x, y, z = _rot(e[i, 0], e[i, 1], e[i, 2], 0.5 * dt * (w1[i, 0] + wx), 0.5 * dt * (w1[i, 1] + wy),
                           0.5 * dt * (w1[i, 2] + wz))
            n = 1.0 / np.sqrt(x * x + y * y + z * z)
            e[i, 0], e[i, 1], e[i, 2] = x * n, y * n, z * n

    # ------------------------------------------------------------------ fused Heun / Depondt (no extra fields)
    @njit(inline="always")
    def _bloc(i, e, ptr, nbr, Jb, A, mu, B, Kc, C, K2, K3, u, has_c, has_u):
        hx, hy, hz = _hfield(i, e, ptr, nbr, Jb)
        if A.shape[0] > 0:
            gx, gy, gz = _onsite_grad(i, e[i, 0], e[i, 1], e[i, 2], A, Kc, C, K2, K3, u, has_c, has_u)
        else:
            gx = gy = gz = 0.0
        inv = 1.0 / (mu[i] * MU_B)
        return (2 * hx - gx) * inv + B[0], (2 * hy - gy) * inv + B[1], (2 * hz - gz) * inv + B[2]

    @njit(inline="always")
    def _g(st, c, spare):
        # spare = [value, flag]
        if spare[1] > 0.5:
            spare[1] = 0.0
            return spare[0]
        while True:
            u_ = 2.0 * _unif(st, c) - 1.0
            v_ = 2.0 * _unif(st, c) - 1.0
            s2 = u_ * u_ + v_ * v_
            if 0.0 < s2 < 1.0:
                break
        f = np.sqrt(-2.0 * np.log(s2) / s2)
        spare[0] = v_ * f
        spare[1] = 1.0
        return u_ * f

    @njit(parallel=True, cache=True)
    def _fused1(e, ep, k1, b, st, sig, alpha, dt, prec, depondt, noisy,
                ptr, nbr, Jb, A, mu, B, Kc, C, K2, K3, u, has_c, has_u, pinned):
        n = e.shape[0]
        nc = st.size
        chunk = (n + nc - 1) // nc
        for c in prange(nc):
            spare = 0.0
            have = False
            for i in range(c * chunk, min(n, (c + 1) * chunk)):
                if pinned[i]:
                    b[i, 0] = b[i, 1] = b[i, 2] = 0.0
                    k1[i, 0] = k1[i, 1] = k1[i, 2] = 0.0
                    ep[i, 0], ep[i, 1], ep[i, 2] = e[i, 0], e[i, 1], e[i, 2]
                    continue
                if noisy:
                    b[i, 0] = sig[i] * _znorm(st, c, ZKN, ZWN, ZFN)
                    b[i, 1] = sig[i] * _znorm(st, c, ZKN, ZWN, ZFN)
                    b[i, 2] = sig[i] * _znorm(st, c, ZKN, ZWN, ZFN)
                bx, by, bz = _bloc(i, e, ptr, nbr, Jb, A, mu, B, Kc, C, K2, K3, u, has_c, has_u)
                bx += b[i, 0]
                by += b[i, 1]
                bz += b[i, 2]
                if depondt:
                    wx, wy, wz = _omega(e[i, 0], e[i, 1], e[i, 2], bx, by, bz, alpha[i], prec)
                    k1[i, 0], k1[i, 1], k1[i, 2] = wx, wy, wz
                    ep[i, 0], ep[i, 1], ep[i, 2] = _rot(e[i, 0], e[i, 1], e[i, 2], dt * wx, dt * wy, dt * wz)
                else:
                    fx, fy, fz = _rhs(e[i, 0], e[i, 1], e[i, 2], bx, by, bz, alpha[i], prec)
                    k1[i, 0], k1[i, 1], k1[i, 2] = fx, fy, fz
                    x, y, z = e[i, 0] + dt * fx, e[i, 1] + dt * fy, e[i, 2] + dt * fz
                    q = 1.0 / np.sqrt(x * x + y * y + z * z)
                    ep[i, 0], ep[i, 1], ep[i, 2] = x * q, y * q, z * q

    @njit(parallel=True, cache=True)
    def _fused2(e, ep, k1, b, alpha, dt, prec, depondt,
                ptr, nbr, Jb, A, mu, B, Kc, C, K2, K3, u, has_c, has_u, pinned):
        for i in prange(e.shape[0]):
            if pinned[i]:
                continue
            bx, by, bz = _bloc(i, ep, ptr, nbr, Jb, A, mu, B, Kc, C, K2, K3, u, has_c, has_u)
            bx += b[i, 0]
            by += b[i, 1]
            bz += b[i, 2]
            if depondt:
                wx, wy, wz = _omega(ep[i, 0], ep[i, 1], ep[i, 2], bx, by, bz, alpha[i], prec)
                x, y, z = _rot(e[i, 0], e[i, 1], e[i, 2], 0.5 * dt * (k1[i, 0] + wx), 0.5 * dt * (k1[i, 1] + wy),
                               0.5 * dt * (k1[i, 2] + wz))
            else:
                fx, fy, fz = _rhs(ep[i, 0], ep[i, 1], ep[i, 2], bx, by, bz, alpha[i], prec)
                x = e[i, 0] + 0.5 * dt * (k1[i, 0] + fx)
                y = e[i, 1] + 0.5 * dt * (k1[i, 1] + fy)
                z = e[i, 2] + 0.5 * dt * (k1[i, 2] + fz)
            q = 1.0 / np.sqrt(x * x + y * y + z * z)
            e[i, 0], e[i, 1], e[i, 2] = x * q, y * q, z * q

    # ------------------------------------------------------------------ Monte Carlo
    @njit(parallel=True, cache=True)
    def _mc_group(e, idx, st, ptr, nbr, Jb, A, mu, B, Kc, C, K2, K3, u, has_c, has_u, beta, sigma, move, overrelax,
                  has_A=True):
        """One pass over the mutually non-interacting sites idx.  move: 0 gaussian, 1 uniform, 2 heat bath
        (exact sampling of exp(beta H.e) for the linear field, Metropolis-corrected for the on-site terms);
        overrelax: reflection about H, accepted with the on-site energy change (H.e is conserved).
        RNG: xorshift64* streams over fixed chunks (thread-count independent).  Returns accepted moves."""
        n = len(idx)
        nc = st.size
        chunk = (n + nc - 1) // nc
        acc = np.zeros(nc, np.int64)
        for c in prange(nc):
            spare = 0.0
            have = False
            nacc = 0
            for t in range(c * chunk, min(n, (c + 1) * chunk)):
                i = idx[t]
                hx, hy, hz = _hfield(i, e, ptr, nbr, Jb)
                zf = MU_B * mu[i]
                Hx = 2 * hx + zf * B[0]
                Hy = 2 * hy + zf * B[1]
                Hz = 2 * hz + zf * B[2]
                ox, oy, oz = e[i, 0], e[i, 1], e[i, 2]
                bt = beta[i]
                lin = True
                if overrelax:
                    HH = Hx * Hx + Hy * Hy + Hz * Hz
                    if HH < 1e-28:
                        continue
                    p = 2 * (ox * Hx + oy * Hy + oz * Hz) / HH
                    nx, ny, nz = p * Hx - ox, p * Hy - oy, p * Hz - oz
                    q = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
                    nx, ny, nz = nx * q, ny * q, nz * q
                    lin = False
                elif move == 2:
                    Hn = np.sqrt(Hx * Hx + Hy * Hy + Hz * Hz)
                    xx = bt * Hn
                    r = _unif(st, c)
                    if xx < 1e-8 or not np.isfinite(xx):
                        cth = 2 * r - 1 if xx < 1e-8 else 1.0
                    else:
                        cth = 1 + np.log1p(-r * (-np.expm1(-2 * xx))) / xx
                    cth = min(1.0, max(-1.0, cth))
                    ph = 2 * np.pi * _unif(st, c)
                    if Hn > 0:
                        zx, zy, zz = Hx / Hn, Hy / Hn, Hz / Hn
                    else:
                        zx, zy, zz = 0.0, 0.0, 1.0
                    if abs(zx) < 0.9:
                        ax, ay, az = 1.0, 0.0, 0.0
                    else:
                        ax, ay, az = 0.0, 1.0, 0.0
                    t1x = zy * az - zz * ay
                    t1y = zz * ax - zx * az
                    t1z = zx * ay - zy * ax
                    n1 = np.sqrt(t1x * t1x + t1y * t1y + t1z * t1z)
                    t1x, t1y, t1z = t1x / n1, t1y / n1, t1z / n1
                    t2x = zy * t1z - zz * t1y
                    t2y = zz * t1x - zx * t1z
                    t2z = zx * t1y - zy * t1x
                    sn = np.sqrt(max(0.0, 1 - cth * cth))
                    nx = cth * zx + sn * (np.cos(ph) * t1x + np.sin(ph) * t2x)
                    ny = cth * zy + sn * (np.cos(ph) * t1y + np.sin(ph) * t2y)
                    nz = cth * zz + sn * (np.cos(ph) * t1z + np.sin(ph) * t2z)
                    lin = False
                else:
                    # Gaussian / uniform proposals: Marsaglia polar pairs, the spare is kept for the next site
                    gg0 = 0.0
                    gg1 = 0.0
                    gg2 = 0.0
                    for comp in range(3):
                        if have:
                            gv = spare
                            have = False
                        else:
                            while True:
                                u_ = 2.0 * _unif(st, c) - 1.0
                                v_ = 2.0 * _unif(st, c) - 1.0
                                s2 = u_ * u_ + v_ * v_
                                if 0.0 < s2 < 1.0:
                                    break
                            f_ = np.sqrt(-2.0 * np.log(s2) / s2)
                            gv = u_ * f_
                            spare = v_ * f_
                            have = True
                        if comp == 0:
                            gg0 = gv
                        elif comp == 1:
                            gg1 = gv
                        else:
                            gg2 = gv
                    if move == 0:
                        nx, ny, nz = ox + sigma * gg0, oy + sigma * gg1, oz + sigma * gg2
                    else:
                        nx, ny, nz = gg0, gg1, gg2
                    q = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
                    nx, ny, nz = nx * q, ny * q, nz * q
                dE = 0.0
                if has_A or has_c or has_u:
                    dE = _onsite_E(i, nx, ny, nz, A, Kc, C, K2, K3, u, has_c, has_u) - \
                        _onsite_E(i, ox, oy, oz, A, Kc, C, K2, K3, u, has_c, has_u)
                if lin:
                    dE -= Hx * (nx - ox) + Hy * (ny - oy) + Hz * (nz - oz)
                ok = dE <= 0
                if not ok:
                    ok = _unif(st, c) < np.exp(-min(dE * bt, 700.0))
                if ok:
                    e[i, 0], e[i, 1], e[i, 2] = nx, ny, nz
                    nacc += 1
            acc[c] = nacc
        return acc.sum()


if HAVE_NUMBA:
    @njit(parallel=True, cache=True)
    def _dip_update(Bd, j, dx, dy, dz, cells, sub, cj0, cj1, cj2, sj, W, fx, fy, fz):
        for k in prange(Bd.shape[0]):
            o0 = (cells[k, 0] - cj0) % fx
            o1 = (cells[k, 1] - cj1) % fy
            o2 = (cells[k, 2] - cj2) % fz
            sk = sub[k]
            Bd[k, 0] += W[sk, sj, o0, o1, o2, 0, 0] * dx + W[sk, sj, o0, o1, o2, 0, 1] * dy + W[sk, sj, o0, o1, o2, 0, 2] * dz
            Bd[k, 1] += W[sk, sj, o0, o1, o2, 1, 0] * dx + W[sk, sj, o0, o1, o2, 1, 1] * dy + W[sk, sj, o0, o1, o2, 1, 2] * dz
            Bd[k, 2] += W[sk, sj, o0, o1, o2, 2, 0] * dx + W[sk, sj, o0, o1, o2, 2, 1] * dy + W[sk, sj, o0, o1, o2, 2, 2] * dz

    @njit(cache=True)
    def _dip_update_serial(Bd, j, dx, dy, dz, cells, sub, cj0, cj1, cj2, sj, W, fx, fy, fz):
        for k in range(Bd.shape[0]):
            o0 = (cells[k, 0] - cj0) % fx
            o1 = (cells[k, 1] - cj1) % fy
            o2 = (cells[k, 2] - cj2) % fz
            sk = sub[k]
            Bd[k, 0] += W[sk, sj, o0, o1, o2, 0, 0] * dx + W[sk, sj, o0, o1, o2, 0, 1] * dy + W[sk, sj, o0, o1, o2, 0, 2] * dz
            Bd[k, 1] += W[sk, sj, o0, o1, o2, 1, 0] * dx + W[sk, sj, o0, o1, o2, 1, 1] * dy + W[sk, sj, o0, o1, o2, 1, 2] * dz
            Bd[k, 2] += W[sk, sj, o0, o1, o2, 2, 0] * dx + W[sk, sj, o0, o1, o2, 2, 1] * dy + W[sk, sj, o0, o1, o2, 2, 2] * dz

    @njit(cache=True)
    def _mc_dipolar_sweep(e, Bd, st, ptr, nbr, Jb, A, mu, B, Kc, C, K2, K3, u, has_c, has_u, pinned,
                          beta, sigma, move, cells, sub, W, par):
        """Exact single-spin Metropolis sweep with the full dipolar field kept up to date incrementally
        (O(N) per accepted move): detailed balance holds exactly for the long-range interaction."""
        N = e.shape[0]
        fx, fy, fz = W.shape[2], W.shape[3], W.shape[4]
        acc = 0
        for i in range(N):
            if pinned[i]:
                continue
            hx, hy, hz = _hfield(i, e, ptr, nbr, Jb)
            zf = MU_B * mu[i]
            si = sub[i]
            W00, W01, W02 = W[si, si, 0, 0, 0, 0, 0], W[si, si, 0, 0, 0, 0, 1], W[si, si, 0, 0, 0, 0, 2]
            W11, W12, W22 = W[si, si, 0, 0, 0, 1, 1], W[si, si, 0, 0, 0, 1, 2], W[si, si, 0, 0, 0, 2, 2]
            ox, oy, oz = e[i, 0], e[i, 1], e[i, 2]
            # field of all other spins (remove own periodic images)
            bx = Bd[i, 0] - mu[i] * (W00 * ox + W01 * oy + W02 * oz)
            by = Bd[i, 1] - mu[i] * (W01 * ox + W11 * oy + W12 * oz)
            bz = Bd[i, 2] - mu[i] * (W02 * ox + W12 * oy + W22 * oz)
            Hx = 2 * hx + zf * (B[0] + bx)
            Hy = 2 * hy + zf * (B[1] + by)
            Hz = 2 * hz + zf * (B[2] + bz)
            r1 = max(_unif(st, 0), 1e-300)
            r2 = _unif(st, 0)
            r3 = max(_unif(st, 0), 1e-300)
            r4 = _unif(st, 0)
            g0 = np.sqrt(-2 * np.log(r1)) * np.cos(2 * np.pi * r2)
            g1 = np.sqrt(-2 * np.log(r1)) * np.sin(2 * np.pi * r2)
            g2 = np.sqrt(-2 * np.log(r3)) * np.cos(2 * np.pi * r4)
            if move == 0:
                nx, ny, nz = ox + sigma * g0, oy + sigma * g1, oz + sigma * g2
            else:
                nx, ny, nz = g0, g1, g2
            q = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
            nx, ny, nz = nx * q, ny * q, nz * q
            dx, dy, dz = nx - ox, ny - oy, nz - oz
            self_n = nx * (W00 * nx + W01 * ny + W02 * nz) + ny * (W01 * nx + W11 * ny + W12 * nz) + \
                nz * (W02 * nx + W12 * ny + W22 * nz)
            self_o = ox * (W00 * ox + W01 * oy + W02 * oz) + oy * (W01 * ox + W11 * oy + W12 * oz) + \
                oz * (W02 * ox + W12 * oy + W22 * oz)
            dE = -(Hx * dx + Hy * dy + Hz * dz) - 0.5 * MU_B * mu[i] ** 2 * (self_n - self_o) + \
                _onsite_E(i, nx, ny, nz, A, Kc, C, K2, K3, u, has_c, has_u) - \
                _onsite_E(i, ox, oy, oz, A, Kc, C, K2, K3, u, has_c, has_u)
            if dE <= 0 or _unif(st, 0) < np.exp(-min(dE * beta[i], 700.0)):
                e[i, 0], e[i, 1], e[i, 2] = nx, ny, nz
                m_ = mu[i]
                if par:
                    _dip_update(Bd, i, m_ * dx, m_ * dy, m_ * dz, cells, sub, cells[i, 0], cells[i, 1], cells[i, 2],
                                si, W, fx, fy, fz)
                else:
                    _dip_update_serial(Bd, i, m_ * dx, m_ * dy, m_ * dz, cells, sub, cells[i, 0], cells[i, 1],
                                       cells[i, 2], si, W, fx, fy, fz)
                acc += 1
        return acc


def neighbour_blocks(Jsp, N):
    """(ptr, nbr, (typ, table)): per-site neighbour list; every 3x3 coupling block is stored once in
    ``table`` (row-major, 9 numbers) and referenced by an int32 type (translational invariance makes the
    table tiny, so the inner loop is cache-resident and memory traffic is 8 bytes per neighbour)."""
    J = Jsp.tocoo()
    i, j = J.row // 3, J.col // 3
    a, b = J.row % 3, J.col % 3
    key = i.astype(np.int64) * N + j
    uniq, inv = np.unique(key, return_inverse=True)
    Jb = np.zeros((len(uniq), 9))
    np.add.at(Jb, (inv, 3 * a + b), J.data)
    si = (uniq // N).astype(np.int64)
    nbr = (uniq % N).astype(np.int32)
    ptr = np.zeros(N + 1, np.int64)
    np.add.at(ptr, si + 1, 1)
    ptr = np.cumsum(ptr)
    if len(Jb) == 0:
        return ptr, nbr, (np.zeros(0, np.int32), np.zeros((1, 9)))
    scale = max(float(np.abs(Jb).max()), 1e-300)
    keyb = np.round(Jb / scale * 2**40).astype(np.int64)            # exact up to ~1e-12 relative
    table_keys, typ = np.unique(keyb, axis=0, return_inverse=True)
    table = np.zeros((len(table_keys), 9))
    np.add.at(table, typ.ravel(), Jb)
    cnt = np.bincount(typ.ravel(), minlength=len(table_keys))
    table /= cnt[:, None]
    if len(table) > 0.5 * len(Jb):                                  # no translational symmetry to exploit
        table = Jb
        typ = np.arange(len(Jb))
    # compact storage: isotropic (1 number) or isotropic + DMI (4 numbers) when exact
    M = table.reshape(-1, 3, 3)
    S = 0.5 * (M + M.transpose(0, 2, 1))
    iso = np.trace(M, axis1=1, axis2=2) / 3
    tol = 1e-14 * max(float(np.abs(M).max()), 1e-300)
    if np.all(np.abs(S - iso[:, None, None] * np.eye(3)) <= tol):
        Aq = 0.5 * (M - M.transpose(0, 2, 1))
        D = np.stack([Aq[:, 1, 2], Aq[:, 2, 0], Aq[:, 0, 1]], 1)       # J^A_ab = eps_abc D_c convention
        if np.all(np.abs(D) <= tol):
            table = iso[:, None].copy()
        else:
            table = np.column_stack([iso, D])
    return ptr, nbr, (typ.ravel().astype(np.int32), np.ascontiguousarray(table))


class FastModel:
    """Kernel-ready arrays of a SpinSystem (rebuild after editing couplings / terms)."""

    def __init__(self, system):
        from .terms import CubicAnisotropy, UniaxialHigher
        s = system
        self.sys = s
        self.ptr, self.nbr, self.Jb = neighbour_blocks(s.Jsp, s.N)
        self.A = np.ascontiguousarray(s.A, float)
        self.mu = np.ascontiguousarray(s.mu, float)
        self.Kc = np.zeros(s.N)
        self.C = np.eye(3)
        self.K2 = np.zeros(s.N)
        self.K3 = np.zeros(s.N)
        self.u = np.array([0, 0, 1.0])
        self.has_c = self.has_u = False
        self.other_terms = []
        for t in s.terms:
            if isinstance(t, CubicAnisotropy) and not self.has_c:
                self.Kc, self.C, self.has_c = np.asarray(t.Kc, float).copy(), np.asarray(t.C, float).copy(), True
            elif isinstance(t, UniaxialHigher) and not self.has_u:
                self.K2, self.K3 = np.asarray(t.K2, float).copy(), np.asarray(t.K3, float).copy()
                self.u, self.has_u = np.asarray(t.u, float).copy(), True
            else:
                self.other_terms.append(t)
        self.pinned = np.zeros(s.N, np.bool_) if s.pinned is None else np.asarray(s.pinned, np.bool_)
        self.rng_state = None
        self.nonlocal_ = bool(self.other_terms) or s.has_higher

    def extra_gradient(self, e):
        """dE/de of everything the kernels do not cover (meV)."""
        s = self.sys
        g = np.zeros_like(e)
        if s.has_higher:
            g += s.higher_gradient(e)
        for t in self.other_terms:
            g += t.gradient(e)
        return g

    def args(self):
        A = self.A if (self.sys.has_A or self.has_c or self.has_u) else np.zeros((0, 3, 3))
        return (self.ptr, self.nbr, self.Jb, A, self.mu, np.asarray(self.sys.B, float), self.Kc, self.C, self.K2,
                self.K3, self.u, self.has_c, self.has_u, self.pinned)

    def field(self, e, out=None, extra=None):
        out = np.empty_like(e) if out is None else out
        has_extra = extra is not None
        ex = extra if has_extra else np.zeros((1, 3))
        pick("_field", len(e))(e, out, self.ptr, self.nbr, self.Jb, self.args()[3], self.mu, np.asarray(self.sys.B, float), self.Kc,
               self.C, self.K2, self.K3, self.u, self.has_c, self.has_u, self.pinned, ex, has_extra)
        return out


def seed_streams(seed, n=256):
    """xorshift64* states for the chunked parallel Gaussian generator."""
    ss = np.random.SeedSequence(int(seed) if seed is not None else None)
    st = ss.generate_state(n, dtype=np.uint64)
    st[st == 0] = np.uint64(0x9E3779B97F4A7C15)
    return st


def available():
    return HAVE_NUMBA


def set_threads(n):
    if HAVE_NUMBA:
        numba.set_num_threads(int(n))


# serial twins of the parallel kernels (small systems: no thread start-up cost)
if HAVE_NUMBA:
    _mc_group_s = njit(cache=False)(_mc_group.py_func)
    _fused1_s = njit(cache=False)(_fused1.py_func)
    _fused2_s = njit(cache=False)(_fused2.py_func)
    _gauss3_s = njit(cache=False)(_gauss3.py_func)
    _field_s = njit(cache=False)(_field.py_func)

SERIAL_BELOW = 4096


def pick(name, n):
    """Parallel kernel for large n, its serial twin below SERIAL_BELOW sites."""
    return globals()[name + "_s"] if n < SERIAL_BELOW else globals()[name]

"""CUDA back end for the atomistic kernels (numba.cuda): one GPU thread per spin.

    from magex.cuda_backend import CudaLLG, CudaMC
    sim = CudaLLG(system, alpha=0.1, dt=1e-3, temperature=300, seed=1)
    e = sim.run(e0, steps=10000)                 # spins stay on the device between steps

The kernels are the device twins of magex/fast.py: symmetry-reduced coupling table (isotropic, isotropic +
DMI, or full 3x3 blocks), on-site tensors, cubic and higher uniaxial terms, Zeeman, pinning, site-resolved
temperature and damping; fused Heun / Depondt predictor and corrector; Metropolis / heat-bath colour sweeps.
Random numbers: one xoroshiro128+ state per thread (numba.cuda.random).

Status: correctness is tested in numba's CUDA simulator (NUMBA_ENABLE_CUDASIM=1, tests/test_cuda.py):
identical T = 0 trajectories to the CPU kernels and the correct thermal statistics.  It has not yet been
timed on a physical GPU, because none was available when it was written.
"""
from __future__ import annotations

import math

import numpy as np

from .spinmodel import GAMMA, K_B, MU_B

try:
    from numba import cuda
    from numba.cuda.random import (create_xoroshiro128p_states, xoroshiro128p_normal_float64,
                                   xoroshiro128p_uniform_float64)
    HAVE_CUDA_API = True
except Exception:  # pragma: no cover
    HAVE_CUDA_API = False


def available():
    return HAVE_CUDA_API and cuda.is_available()


if HAVE_CUDA_API:
    @cuda.jit(device=True)
    def _h(i, e, ptr, nbr, typ, T):
        hx = 0.0
        hy = 0.0
        hz = 0.0
        w = T.shape[1]
        for k in range(ptr[i], ptr[i + 1]):
            j = nbr[k]
            t = typ[k]
            ex = e[j, 0]
            ey = e[j, 1]
            ez = e[j, 2]
            if w == 1:
                Jv = T[t, 0]
                hx += Jv * ex
                hy += Jv * ey
                hz += Jv * ez
            elif w == 4:
                Jv = T[t, 0]
                hx += Jv * ex + (ey * T[t, 3] - ez * T[t, 2])
                hy += Jv * ey + (ez * T[t, 1] - ex * T[t, 3])
                hz += Jv * ez + (ex * T[t, 2] - ey * T[t, 1])
            else:
                hx += T[t, 0] * ex + T[t, 1] * ey + T[t, 2] * ez
                hy += T[t, 3] * ex + T[t, 4] * ey + T[t, 5] * ez
                hz += T[t, 6] * ex + T[t, 7] * ey + T[t, 8] * ez
        return hx, hy, hz

    @cuda.jit(device=True)
    def _onsite_grad(i, x, y, z, A, Kc, C, K2, K3, u, has_c, has_u):
        if A.shape[0] == 0:
            return 0.0, 0.0, 0.0
        gx = 2.0 * (A[i, 0, 0] * x + A[i, 0, 1] * y + A[i, 0, 2] * z)
        gy = 2.0 * (A[i, 1, 0] * x + A[i, 1, 1] * y + A[i, 1, 2] * z)
        gz = 2.0 * (A[i, 2, 0] * x + A[i, 2, 1] * y + A[i, 2, 2] * z)
        if has_c:
            a1 = C[0, 0] * x + C[0, 1] * y + C[0, 2] * z
            a2 = C[1, 0] * x + C[1, 1] * y + C[1, 2] * z
            a3 = C[2, 0] * x + C[2, 1] * y + C[2, 2] * z
            q1 = a1 * a1
            q2 = a2 * a2
            q3 = a3 * a3
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

    @cuda.jit(device=True)
    def _onsite_E(i, x, y, z, A, Kc, C, K2, K3, u, has_c, has_u):
        if A.shape[0] == 0:
            return 0.0
        E = (x * (A[i, 0, 0] * x + A[i, 0, 1] * y + A[i, 0, 2] * z) + y * (A[i, 1, 0] * x + A[i, 1, 1] * y + A[i, 1, 2] * z)
             + z * (A[i, 2, 0] * x + A[i, 2, 1] * y + A[i, 2, 2] * z))
        if has_c:
            a1 = C[0, 0] * x + C[0, 1] * y + C[0, 2] * z
            a2 = C[1, 0] * x + C[1, 1] * y + C[1, 2] * z
            a3 = C[2, 0] * x + C[2, 1] * y + C[2, 2] * z
            E += Kc[i] * (a1 * a1 * a2 * a2 + a2 * a2 * a3 * a3 + a3 * a3 * a1 * a1)
        if has_u:
            c = x * u[0] + y * u[1] + z * u[2]
            E += -K2[i] * c**4 - K3[i] * c**6
        return E

    @cuda.jit(device=True)
    def _field(i, e, ptr, nbr, typ, T, A, mu, B, Kc, C, K2, K3, u, has_c, has_u):
        hx, hy, hz = _h(i, e, ptr, nbr, typ, T)
        gx, gy, gz = _onsite_grad(i, e[i, 0], e[i, 1], e[i, 2], A, Kc, C, K2, K3, u, has_c, has_u)
        inv = 1.0 / (mu[i] * 5.7883818060e-2)
        return (2 * hx - gx) * inv + B[0], (2 * hy - gy) * inv + B[1], (2 * hz - gz) * inv + B[2]

    @cuda.jit(device=True)
    def _rhs(ex, ey, ez, bx, by, bz, a, prec):
        cx = ey * bz - ez * by
        cy = ez * bx - ex * bz
        cz = ex * by - ey * bx
        dx = ey * cz - ez * cy
        dy = ez * cx - ex * cz
        dz = ex * cy - ey * cx
        pre = -0.1760859630 / (1.0 + a * a)
        if prec:
            return pre * (cx + a * dx), pre * (cy + a * dy), pre * (cz + a * dz)
        return pre * a * dx, pre * a * dy, pre * a * dz

    @cuda.jit(device=True)
    def _rot(x, y, z, wx, wy, wz):
        th = math.sqrt(wx * wx + wy * wy + wz * wz)
        if th < 1e-14:
            return x, y, z
        kx = wx / th
        ky = wy / th
        kz = wz / th
        c = math.cos(th)
        s = math.sin(th)
        kd = kx * x + ky * y + kz * z
        return (x * c + (ky * z - kz * y) * s + kx * kd * (1 - c), y * c + (kz * x - kx * z) * s + ky * kd * (1 - c),
                z * c + (kx * y - ky * x) * s + kz * kd * (1 - c))

    @cuda.jit
    def _k_pred(e, ep, k1, b, states, sig, alpha, dt, prec, depondt, noisy,
                ptr, nbr, typ, T, A, mu, B, Kc, C, K2, K3, u, has_c, has_u, inactive):
        i = cuda.grid(1)
        if i >= e.shape[0]:
            return
        if inactive[i]:
            ep[i, 0] = e[i, 0]
            ep[i, 1] = e[i, 1]
            ep[i, 2] = e[i, 2]
            k1[i, 0] = 0.0
            k1[i, 1] = 0.0
            k1[i, 2] = 0.0
            b[i, 0] = 0.0
            b[i, 1] = 0.0
            b[i, 2] = 0.0
            return
        if noisy:
            b[i, 0] = sig[i] * xoroshiro128p_normal_float64(states, i)
            b[i, 1] = sig[i] * xoroshiro128p_normal_float64(states, i)
            b[i, 2] = sig[i] * xoroshiro128p_normal_float64(states, i)
        else:
            b[i, 0] = 0.0
            b[i, 1] = 0.0
            b[i, 2] = 0.0
        bx, by, bz = _field(i, e, ptr, nbr, typ, T, A, mu, B, Kc, C, K2, K3, u, has_c, has_u)
        bx += b[i, 0]
        by += b[i, 1]
        bz += b[i, 2]
        fx, fy, fz = _rhs(e[i, 0], e[i, 1], e[i, 2], bx, by, bz, alpha[i], prec)
        if depondt:
            wx = e[i, 1] * fz - e[i, 2] * fy
            wy = e[i, 2] * fx - e[i, 0] * fz
            wz = e[i, 0] * fy - e[i, 1] * fx
            k1[i, 0] = wx
            k1[i, 1] = wy
            k1[i, 2] = wz
            x, y, z = _rot(e[i, 0], e[i, 1], e[i, 2], dt * wx, dt * wy, dt * wz)
        else:
            k1[i, 0] = fx
            k1[i, 1] = fy
            k1[i, 2] = fz
            x = e[i, 0] + dt * fx
            y = e[i, 1] + dt * fy
            z = e[i, 2] + dt * fz
        q = 1.0 / math.sqrt(x * x + y * y + z * z)
        ep[i, 0] = x * q
        ep[i, 1] = y * q
        ep[i, 2] = z * q

    @cuda.jit
    def _k_corr(e, ep, k1, b, alpha, dt, prec, depondt,
                ptr, nbr, typ, T, A, mu, B, Kc, C, K2, K3, u, has_c, has_u, inactive):
        i = cuda.grid(1)
        if i >= e.shape[0] or inactive[i]:
            return
        bx, by, bz = _field(i, ep, ptr, nbr, typ, T, A, mu, B, Kc, C, K2, K3, u, has_c, has_u)
        bx += b[i, 0]
        by += b[i, 1]
        bz += b[i, 2]
        fx, fy, fz = _rhs(ep[i, 0], ep[i, 1], ep[i, 2], bx, by, bz, alpha[i], prec)
        if depondt:
            wx = ep[i, 1] * fz - ep[i, 2] * fy
            wy = ep[i, 2] * fx - ep[i, 0] * fz
            wz = ep[i, 0] * fy - ep[i, 1] * fx
            x, y, z = _rot(e[i, 0], e[i, 1], e[i, 2], 0.5 * dt * (k1[i, 0] + wx), 0.5 * dt * (k1[i, 1] + wy),
                           0.5 * dt * (k1[i, 2] + wz))
        else:
            x = e[i, 0] + 0.5 * dt * (k1[i, 0] + fx)
            y = e[i, 1] + 0.5 * dt * (k1[i, 1] + fy)
            z = e[i, 2] + 0.5 * dt * (k1[i, 2] + fz)
        q = 1.0 / math.sqrt(x * x + y * y + z * z)
        e[i, 0] = x * q
        e[i, 1] = y * q
        e[i, 2] = z * q

    @cuda.jit
    def _k_mc(e, idx, states, ptr, nbr, typ, T, A, mu, B, Kc, C, K2, K3, u, has_c, has_u, beta, sigma, move, acc):
        t = cuda.grid(1)
        if t >= idx.shape[0]:
            return
        i = idx[t]
        hx, hy, hz = _h(i, e, ptr, nbr, typ, T)
        zf = 5.7883818060e-2 * mu[i]
        Hx = 2 * hx + zf * B[0]
        Hy = 2 * hy + zf * B[1]
        Hz = 2 * hz + zf * B[2]
        ox = e[i, 0]
        oy = e[i, 1]
        oz = e[i, 2]
        bt = beta[i]
        lin = True
        if move == 2:
            Hn = math.sqrt(Hx * Hx + Hy * Hy + Hz * Hz)
            xx = bt * Hn
            r = xoroshiro128p_uniform_float64(states, t)
            if xx < 1e-8:
                cth = 2 * r - 1
            else:
                cth = 1 + math.log1p(-r * (-math.expm1(-2 * xx))) / xx
            cth = min(1.0, max(-1.0, cth))
            ph = 2 * math.pi * xoroshiro128p_uniform_float64(states, t)
            if Hn > 0:
                zx = Hx / Hn
                zy = Hy / Hn
                zz = Hz / Hn
            else:
                zx = 0.0
                zy = 0.0
                zz = 1.0
            if abs(zx) < 0.9:
                ax = 1.0
                ay = 0.0
                az = 0.0
            else:
                ax = 0.0
                ay = 1.0
                az = 0.0
            t1x = zy * az - zz * ay
            t1y = zz * ax - zx * az
            t1z = zx * ay - zy * ax
            n1 = math.sqrt(t1x * t1x + t1y * t1y + t1z * t1z)
            t1x /= n1
            t1y /= n1
            t1z /= n1
            t2x = zy * t1z - zz * t1y
            t2y = zz * t1x - zx * t1z
            t2z = zx * t1y - zy * t1x
            sn = math.sqrt(max(0.0, 1 - cth * cth))
            nx = cth * zx + sn * (math.cos(ph) * t1x + math.sin(ph) * t2x)
            ny = cth * zy + sn * (math.cos(ph) * t1y + math.sin(ph) * t2y)
            nz = cth * zz + sn * (math.cos(ph) * t1z + math.sin(ph) * t2z)
            lin = False
        else:
            g0 = xoroshiro128p_normal_float64(states, t)
            g1 = xoroshiro128p_normal_float64(states, t)
            g2 = xoroshiro128p_normal_float64(states, t)
            if move == 0:
                nx = ox + sigma * g0
                ny = oy + sigma * g1
                nz = oz + sigma * g2
            else:
                nx = g0
                ny = g1
                nz = g2
            q = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
            nx *= q
            ny *= q
            nz *= q
        dE = _onsite_E(i, nx, ny, nz, A, Kc, C, K2, K3, u, has_c, has_u) - \
            _onsite_E(i, ox, oy, oz, A, Kc, C, K2, K3, u, has_c, has_u)
        if lin:
            dE -= Hx * (nx - ox) + Hy * (ny - oy) + Hz * (nz - oz)
        ok = dE <= 0
        if not ok:
            ok = xoroshiro128p_uniform_float64(states, t) < math.exp(-min(dE * bt, 700.0))
        if ok:
            e[i, 0] = nx
            e[i, 1] = ny
            e[i, 2] = nz
            cuda.atomic.add(acc, 0, 1)


class _Device:
    """Model arrays on the GPU (from FastModel)."""

    def __init__(self, system):
        from . import fast
        if not HAVE_CUDA_API:
            raise RuntimeError("numba.cuda is not available")
        fm = fast.FastModel(system)
        if fm.nonlocal_:
            raise NotImplementedError("CUDA back end: dipolar / multi-site terms are not on the GPU yet")
        self.sys = system
        self.fm = fm
        a = fm.args()
        ptr, nbr, (typ, table), A, mu, B, Kc, C, K2, K3, u, has_c, has_u, pinned = a
        d = cuda.to_device
        self.arrs = [d(ptr), d(nbr), d(typ), d(table), d(A if A.shape[0] else np.zeros((0, 3, 3))), d(mu), None,
                     d(Kc), d(C), d(K2), d(K3), d(u)]
        self.has_c, self.has_u = bool(has_c), bool(has_u)
        self.pinned = d(np.ascontiguousarray(pinned, np.bool_))
        self.N = system.N
        from numba import config
        self.tpb = 16 if config.ENABLE_CUDASIM else 128     # the simulator runs one Python thread per GPU thread
        self.blocks = (self.N + self.tpb - 1) // self.tpb

    def args(self):
        B = cuda.to_device(np.asarray(self.sys.B, float))
        a = list(self.arrs)
        a[6] = B
        return (*a, self.has_c, self.has_u)


class CudaLLG:
    """Stochastic Heun / Depondt LLG on the GPU; same equations, units and noise as LLGSolver."""

    def __init__(self, system, alpha=0.1, dt=1e-3, temperature=0.0, seed=1, integrator="heun", precession=True):
        from .structure import site_temperature
        self.dev = _Device(system)
        self.sys = system
        N = system.N
        al = np.broadcast_to(np.asarray(alpha, float), (N,)).astype(float).copy()
        self.alpha = cuda.to_device(al)
        self.dt = float(dt)
        self.T = float(temperature)
        T = np.broadcast_to(site_temperature(system, self.T), (N,)).astype(float)
        self.noisy = bool(np.any(T > 0))
        sig = np.sqrt(2.0 * al * K_B * T / (GAMMA * system.mu * MU_B * self.dt))
        self.sig = cuda.to_device(np.ascontiguousarray(sig))
        self.depondt = integrator == "depondt"
        self.prec = bool(precession)
        self.states = create_xoroshiro128p_states(N, seed=int(seed))
        self.ep = cuda.device_array((N, 3))
        self.k1 = cuda.device_array((N, 3))
        self.b = cuda.to_device(np.zeros((N, 3)))
        self.time = 0.0

    def step_device(self, e_dev):
        d = self.dev
        args = d.args()
        _k_pred[d.blocks, d.tpb](e_dev, self.ep, self.k1, self.b, self.states, self.sig, self.alpha, self.dt, self.prec,
                                 self.depondt, self.noisy, *args, d.pinned)
        _k_corr[d.blocks, d.tpb](e_dev, self.ep, self.k1, self.b, self.alpha, self.dt, self.prec, self.depondt,
                                 *args, d.pinned)
        self.time += self.dt

    def run(self, e, steps, callback=None, every=0):
        e_dev = cuda.to_device(np.ascontiguousarray(e, float))
        for n in range(steps):
            self.step_device(e_dev)
            if callback and every and (n + 1) % every == 0:
                callback(n + 1, e_dev.copy_to_host())
        return e_dev.copy_to_host()


class CudaMC:
    """Metropolis / heat-bath colour sweeps on the GPU."""

    def __init__(self, system, temperature, seed=1, sigma=0.6, move="gaussian"):
        from .montecarlo import greedy_coloring
        from .structure import site_temperature
        self.dev = _Device(system)
        self.sys = system
        col = greedy_coloring(system.N, system.bonds)
        self.groups = []
        for c in range(col.max() + 1):
            g = np.where(col == c)[0]
            if system.pinned is not None:
                g = g[~system.pinned[g]]
            self.groups.append(cuda.to_device(np.ascontiguousarray(g, np.int64)))
        T = np.broadcast_to(site_temperature(system, float(temperature)), (system.N,)).astype(float)
        beta = np.where(T > 0, 1.0 / (K_B * np.maximum(T, 1e-300)), np.inf)
        self.beta = cuda.to_device(beta)
        self.sigma = float(sigma)
        self.move = {"gaussian": 0, "uniform": 1, "heatbath": 2}[move]
        if self.move == 2 and not np.all(np.isfinite(beta)):
            self.move = 0
        self.states = create_xoroshiro128p_states(max(len(g) for g in self.groups), seed=int(seed))
        self.acc = cuda.to_device(np.zeros(1, np.int64))
        self.tried = 0

    def sweep_device(self, e_dev):
        d = self.dev
        args = d.args()
        for g in self.groups:
            blocks = (g.shape[0] + d.tpb - 1) // d.tpb
            _k_mc[blocks, d.tpb](e_dev, g, self.states, *args, self.beta, self.sigma, self.move, self.acc)
            self.tried += g.shape[0]

    def acceptance(self):
        a = int(self.acc.copy_to_host()[0]) / max(self.tried, 1)
        self.acc = cuda.to_device(np.zeros(1, np.int64))
        self.tried = 0
        return a

"""User-defined field and energy terms from expressions (mumax3 custom fields / AddEdensTerm) and the
moving simulation frame (mumax3 ext_centerWall).

Expressions are evaluated with numpy on cell-centre coordinates x, y, z (m), time t (s), the reduced
magnetisation mx, my, mz, the material arrays Ms, A, Ku1 and any user constants:

  [[custom_field]]   B = ["0", "0", "1e-3*sin(2*pi*5e9*t)*exp(-(x/50e-9)**2)"]       (tesla)
  [[custom_energy]]  density = "-K4*(mx**4 + my**4 + mz**4)"   constants = {K4 = 1e4}  (J/m^3)
"""
from __future__ import annotations

import numpy as np

_NS = {k: getattr(np, k) for k in ("sin", "cos", "tan", "exp", "log", "sqrt", "abs", "tanh", "arctan2", "arctan",
                                   "sinh", "cosh", "where", "pi", "heaviside", "minimum", "maximum", "sign")}


def _compile(expr):
    code = compile(str(expr), "<magex expression>", "eval")
    for name in code.co_names:
        if name.startswith("__"):
            raise ValueError(f"forbidden name in expression: {name}")
    return code


def custom_field(spec, mesh):
    X, Y, Z = mesh.coords()
    codes = [_compile(e) for e in spec["B"]]
    consts = dict(spec.get("constants", {}))
    region = spec.get("region")

    def fn(m, t, sysm):
        ns = dict(_NS, x=X, y=Y, z=Z, t=t, mx=m[..., 0], my=m[..., 1], mz=m[..., 2], Ms=sysm.Ms, A=sysm.A,
                  Ku1=sysm.Ku1, **consts)
        B = np.stack([np.broadcast_to(np.asarray(eval(c, {"__builtins__": {}}, ns), float), mesh.n) for c in codes], -1)
        if region is not None:
            B = np.where((sysm.region == int(region))[..., None], B, 0.0)
        return B
    return fn


def custom_energy(spec, mesh):
    X, Y, Z = mesh.coords()
    code = _compile(spec["density"])
    consts = dict(spec.get("constants", {}))
    region = spec.get("region")

    def fn(m, sysm):
        ns = dict(_NS, x=X, y=Y, z=Z, t=sysm.t, mx=m[..., 0], my=m[..., 1], mz=m[..., 2], Ms=sysm.Ms, A=sysm.A,
                  Ku1=sysm.Ku1, **consts)
        w = np.broadcast_to(np.asarray(eval(code, {"__builtins__": {}}, ns), float), mesh.n)
        w = np.where(sysm.magnetic, w, 0.0)
        if region is not None:
            w = np.where(sysm.region == int(region), w, 0.0)
        return w
    return fn


class MovingFrame:
    """Keep a domain wall (or texture) centred along ``axis`` by shifting the magnetisation by whole cells.

    component: the magnetisation component that changes sign across the wall (mumax3 ext_centerWall(c)).
    The cumulative shift (m) is added to the measured wall position, so that the lab-frame position
    and velocity are available from ``positions``.
    """

    def __init__(self, sysm, component="z", axis="x"):
        self.sys = sysm
        self.c = {"x": 0, "y": 1, "z": 2}[component]
        self.ax = {"x": 0, "y": 1, "z": 2}[axis]
        self.shift = 0
        self.positions = []

    def __call__(self, t, m):
        s = self.sys
        mag = s.magnetic
        mc = float(m[..., self.c][mag].mean())
        L = s.mesh.n[self.ax] * s.mesh.d[self.ax]
        # sign of the left domain decides the direction of the offset
        left = float(np.take(m[..., self.c], 0, axis=self.ax)[np.take(mag, 0, axis=self.ax)].mean())
        off = mc * L / 2 * np.sign(left if left != 0 else 1)          # wall offset from the window centre
        n = int(np.round(off / s.mesh.d[self.ax]))
        self.positions.append((t, (self.shift * s.mesh.d[self.ax]) + off + L / 2))
        if n != 0:
            new = np.roll(m, -n, axis=self.ax)
            sl = [slice(None)] * 3
            if n > 0:
                sl[self.ax] = slice(-n, None)
                edge = np.take(m, [-1], axis=self.ax)
            else:
                sl[self.ax] = slice(0, -n)
                edge = np.take(m, [0], axis=self.ax)
            new[tuple(sl)] = edge
            m[...] = np.where(mag[..., None], new, 0.0)
            self.shift += n

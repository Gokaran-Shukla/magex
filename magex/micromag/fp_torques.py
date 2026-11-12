"""Torque efficiencies from first-principles transport, and spin pumping.

Spin-orbit torque efficiency from Hall conductivities (e.g. from Wannier-based spin/orbital Hall
codes), in the usual units sigma_SH, sigma_OH in (hbar/e) S/cm and sigma_xx in S/cm:

    theta_SH = 2 sigma_SH / sigma_xx,     theta_OH = 2 sigma_OH / sigma_xx
    theta_eff = T_int [ theta_SH + eta_LS theta_OH ] [1 - sech(d_NM / lambda_s)]

T_int is the interface spin transparency, eta_LS the orbital-to-spin conversion of the ferromagnet
(orbital torque, Go & Lee PRR 2, 013177 (2020)), d_NM and lambda_s the heavy-metal thickness and
spin diffusion length (finite-thickness factor; omitted when not given).

Spin pumping (Tserkovnyak, Brataas & Bauer, PRL 88, 117601 (2002)):

    j_s = (hbar / 4 pi) g_r  m x dm/dt                  (J/m^2)
    Delta alpha = g mu_B g_r / (4 pi Ms t_FM)
    inverse spin Hall field  E = theta_SH (2e/hbar) rho lambda_s tanh(d/2 lambda_s)/d  j_s,dc
"""
from __future__ import annotations

import json
import os

import numpy as np

from .constants import E_CHARGE, HBAR, MU_B


def theta_from_conductivities(sigma_xx, sigma_SH=0.0, sigma_OH=0.0, eta_LS=0.0, transparency=1.0,
                              nm_thickness_nm=None, lambda_s_nm=None):
    th = transparency * (2 * sigma_SH / sigma_xx + eta_LS * 2 * sigma_OH / sigma_xx)
    if nm_thickness_nm and lambda_s_nm:
        th *= 1 - 1 / np.cosh(nm_thickness_nm / lambda_s_nm)
    return float(th)


def resolve_theta(value, base="."):
    """theta as a number, or {sigma_xx, sigma_SH, sigma_OH, eta_LS, transparency, ...}, or {from = "file.json"}."""
    if not isinstance(value, dict):
        return float(value)
    v = dict(value)
    if "from" in v:
        p = v.pop("from")
        p = p if os.path.isabs(p) else os.path.join(base, p)
        with open(p) as f:
            d = json.load(f)
        for k in ("sigma_xx", "sigma_SH", "sigma_OH"):
            if k in d and k not in v:
                v[k] = d[k]
    return theta_from_conductivities(**v)


def spin_pumping_damping(g_mix, Ms, t_fm):
    """Delta alpha (dimensionless); g_mix in 1/m^2, Ms in A/m, t_fm in m."""
    return float(2.0 * MU_B * g_mix / (4 * np.pi * Ms * t_fm))


def pumped_spin_current(table, g_mix):
    """DC spin-current density (J/m^2, 3-vector) from a run table (uniform-precession approximation)."""
    _, idx = np.unique(table[:, 0], return_index=True)
    t = table[idx, 0]
    m = table[idx, 1:4]
    dm = np.gradient(m, t, axis=0)
    js = HBAR / (4 * np.pi) * g_mix * np.cross(m, dm)
    half = len(t) // 2
    return js[half:].mean(0), js


def ishe_voltage(js_dc, theta_SH, rho, lambda_s, d_nm, length):
    """ISHE voltage (V) along a strip of given length from the DC spin current j_s (J/m^2, scalar)."""
    jc = theta_SH * (2 * E_CHARGE / HBAR) * js_dc * lambda_s * np.tanh(d_nm / (2 * lambda_s)) / d_nm
    return float(rho * jc * length)

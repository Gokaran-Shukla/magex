"""Room-temperature-ish parameters of common thin-film materials (SI units).

Ms (A/m), A (J/m), Ku1 (J/m^3, + = easy axis along anisU), Kc1 (J/m^3), D (J/m^2), alpha.
Values are typical literature numbers for sputtered films; adjust to your sample.
"""
MATERIALS = {
    "Py": dict(Ms=8.0e5, A=1.3e-11, alpha=0.01, note="Ni80Fe20 permalloy (muMAG standard problem 4)"),
    "Co/Pt": dict(Ms=5.8e5, A=1.5e-11, Ku1=8.0e5, anisU=[0, 0, 1], D=3.0e-3, dmi="interfacial", alpha=0.3,
                  note="ultrathin Co on Pt with PMA and interfacial DMI (Sampaio et al. Nat. Nano. 2013)"),
    "Pt/Co/AlOx": dict(Ms=1.1e6, A=1.6e-11, Ku1=1.25e6, anisU=[0, 0, 1], D=2.0e-3, dmi="interfacial", alpha=0.5),
    "CoFeB": dict(Ms=1.1e6, A=1.5e-11, Ku1=8.3e5, anisU=[0, 0, 1], alpha=0.01,
                  note="CoFeB/MgO perpendicular free layer"),
    "Ta/CoFeB/MgO": dict(Ms=1.0e6, A=1.5e-11, Ku1=7.5e5, anisU=[0, 0, 1], D=0.3e-3, dmi="interfacial", alpha=0.02),
    "Fe": dict(Ms=1.7e6, A=2.1e-11, Kc1=4.8e4, alpha=0.002, note="bcc Fe, cubic anisotropy along x,y,z"),
    "Ni": dict(Ms=4.9e5, A=0.9e-11, Kc1=-5.7e3, alpha=0.045),
    "Co_hcp": dict(Ms=1.4e6, A=3.0e-11, Ku1=4.5e5, anisU=[0, 0, 1], alpha=0.005),
    "YIG": dict(Ms=1.4e5, A=3.7e-12, Kc1=-610.0, alpha=1e-4, note="Y3Fe5O12 thin film"),
    "FeGe": dict(Ms=3.84e5, A=8.78e-12, D=1.58e-3, dmi="bulk", alpha=0.3, note="B20 FeGe (helical period ~70 nm)"),
    "MnSi": dict(Ms=1.63e5, A=0.4e-12, D=0.115e-3, dmi="bulk", alpha=0.1, note="B20 MnSi (approx. 18 nm helix)"),
    "Fe3GeTe2": dict(Ms=3.76e5, A=1.0e-12, Ku1=1.46e6, anisU=[0, 0, 1], alpha=0.1, note="2D vdW ferromagnet"),
    "CrI3": dict(Ms=1.37e5, A=0.75e-12, Ku1=3.0e5, anisU=[0, 0, 1], alpha=0.1, note="monolayer-scale CrI3 (estimates)"),
}


def material(name, **override):
    """Copy of a library material with optional overrides."""
    if name not in MATERIALS:
        raise KeyError(f"unknown material '{name}'; available: {', '.join(MATERIALS)}")
    d = dict(MATERIALS[name])
    d.pop("note", None)
    d.update(override)
    return d

"""magex.micromag -- finite-difference micromagnetics for thin films and nanostructures.

OOMMF/mumax-style solver in pure NumPy/SciPy (FFT threads):

* Mesh + geometries (rectangle, disk, ellipse, ring, polygon, PNG mask) and multi-region,
  multilayer stacks with per-region materials
* Energy terms: exchange (harmonic-mean A at interfaces), demagnetisation (Newell tensor,
  FFT convolution, open or periodic boundaries), uniaxial (Ku1, Ku2) and cubic anisotropy,
  interfacial (C_nv) and bulk (B20) DMI with the chiral boundary conditions, Zeeman (static
  + time-dependent), spin-orbit torques (damping-like and field-like)
* Solvers: adaptive Dormand-Prince RK45 LLG, stochastic Heun at T > 0, Barzilai-Borwein energy
  minimiser, relax (damping-only), hysteresis loops
* I/O: OVF 2.0 (text and binary-8), table files, publication-quality figures
* Multiscale bridge: atomistic magex results.json -> A, D, K, Ms
"""
from .constants import MU0, GAMMA_LL, MU_B, K_B, HBAR, E_CHARGE
from .mesh import Mesh, geometry_mask
from .materials import MATERIALS, material
from .system import Micromagnet
from .solver import RK45, Heun, minimize, relax, hysteresis
from .io import write_ovf, read_ovf
from .bridge import micromagnetic_parameters

__all__ = ["MU0", "GAMMA_LL", "MU_B", "K_B", "HBAR", "E_CHARGE", "Mesh", "geometry_mask", "MATERIALS", "material",
           "Micromagnet", "RK45", "Heun", "minimize", "relax", "hysteresis", "write_ovf", "read_ovf", "micromagnetic_parameters"]

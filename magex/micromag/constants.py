"""SI constants used by the micromagnetic solver."""
import numpy as np

MU0 = 4e-7 * np.pi                 # T m / A   (pre-2019 exact value, used by OOMMF/mumax)
GAMMA_LL = 1.7595e11               # rad / (s T): mumax/OOMMF convention gamma = g mu_B / hbar with g = 2.0023
MU_B = 9.2740100783e-24            # J / T
K_B = 1.380649e-23                 # J / K
HBAR = 1.054571817e-34             # J s
E_CHARGE = 1.602176634e-19         # C
MEV = 1.602176634e-22              # J

import os
import numpy as np, sys
sys.path.insert(0,'..')
sys.path.insert(0,os.environ.get('SPIRIT_PYTHON', 'spirit/core/python'))
from magex.interop import export_spirit
from magex.spinmodel import SpinSystem, load_results, normalize
from spirit import state, system, geometry, configuration, hamiltonian, quantities
B=[0.3,-0.2,1.5]
export_spirit('tri.json','spi', cfg=dict(program=dict(field=B)))
s=SpinSystem(load_results('tri.json'),(30,30,1),(True,True,False),field=B)
rng=np.random.default_rng(4)
e=normalize(s.init_spins('skyrmion',radius=8.0)+0.3*rng.normal(size=(s.N,3)))
import os; os.chdir('spi')
with state.State('input.cfg', quiet=True) as p:
    pos=geometry.get_positions(p); nsp=system.get_spin_directions(p)
    print('Spirit N', len(pos), 'magex N', s.N)
    # map by positions (Spirit uses lattice_constant 1 with bravais in A)
    from scipy.spatial import cKDTree
    t=cKDTree(s.positions); d,idx=t.query(pos)
    print('max position mismatch', d.max())
    nsp[:]=e[idx]
    system.update_data(p); Es=system.get_energy(p)
    print('contributions', system.get_energy_contributions(p) if hasattr(system,'get_energy_contributions') else '')
    print('Spirit energy', Es, ' magex energy', s.energy(e), ' diff', Es-s.energy(e))

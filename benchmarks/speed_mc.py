import numpy as np, sys, time
sys.path.insert(0,'..')
from magex.spinmodel import SpinSystem, load_results
from magex.montecarlo import MetropolisMC
from magex import fast
s=SpinSystem(load_results('sc.json'),(32,32,32)); e=s.init_spins('ferro')
for th in [1,2]:
    fast.set_threads(th); mc=MetropolisMC(s,30.0,1,0.6,0); mc.sweep(e); ts=[]
    for _ in range(5):
        t=time.time(); mc.sweep(e); ts.append(time.time()-t)
    print(th,'threads', f'{min(ts)/s.N*1e9:.1f} ns/spin/sweep')

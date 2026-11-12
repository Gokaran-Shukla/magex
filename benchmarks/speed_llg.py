import numpy as np, sys, time
sys.path.insert(0,'..')
from magex.spinmodel import SpinSystem, load_results
from magex.llg import LLGSolver
from magex import fast
s=SpinSystem(load_results('sc.json'),(32,32,32)); e=s.init_spins('ferro')
for th in [1,2]:
    fast.set_threads(th); sv=LLGSolver(s,1.0,1e-4,10.0,1); e=sv.step(e); ts=[]
    for _ in range(3):
        t=time.time()
        for k in range(50): e=sv.step(e)
        ts.append((time.time()-t)/50)
    print(th, f'{min(ts)/s.N*1e9:.1f} ns/spin/step')

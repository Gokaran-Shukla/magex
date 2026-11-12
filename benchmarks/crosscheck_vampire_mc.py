import numpy as np, sys, time
sys.path.insert(0,'..')
from magex.spinmodel import SpinSystem, load_results
from magex.montecarlo import MetropolisMC
s=SpinSystem(load_results('sc.json'),(13,13,13)); e=s.init_spins('ferro'); t=time.time()
v=np.loadtxt('vamp/output', comments='#')
for T,mv in v:
    mc=MetropolisMC(s,T,1,0.6,0); ms=[]
    for n in range(9000):
        e=mc.sweep(e)
        if n%50==49 and n<3000: mc.sigma=float(np.clip(mc.sigma*0.5/max(1-mc.acceptance(),0.05),0.02,4))
        if n>=3000: ms.append(np.linalg.norm(e.mean(0)))
    print(f'{T:5.1f}  VAMPIRE {mv:.4f}  magex {np.mean(ms):.4f}')
print(f'magex time {time.time()-t:.1f}s, ns/spin/sweep {(time.time()-t)/(len(v)*9000*s.N)*1e9:.1f}')

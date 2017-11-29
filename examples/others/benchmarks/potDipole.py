# test code for pystokes
from __future__ import division
import numpy as np
import sys, os, time
import pystokes, pyforces

a  = 1
eta = 1.0/6

for i in range(6):
    Np = 20000*i
    uRbm = pystokes.unbounded.Rbm(a, Np, eta)   # instantiate the classes
    ff = pyforces.forceFields.Forces(Np)
    
    r = 2*np.linspace(-3*Np, 3*Np, 3*Np)
    p = np.ones(3*Np)
    v = np.zeros(3*Np)
    
    t1 = time.time()
    uRbm.potDipoleV(v, r, p)
    print 'Time take: ', time.time()-t1, "    Np ", Np

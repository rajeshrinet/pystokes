# Mobility as a function of volume fraction
from __future__ import division
import matplotlib.pyplot as plt 
import numpy as np
import pystokes, os, sys
import pyforces

#Parameters
a, eta, dim = 1.0, 1.0/6, 3
Np, Nb, Nm = 1, 4, 4
ta =(4*np.pi/3)**(1.0/3) 
L = ta/np.arange(0.01, 0.8, 0.01)

#Memory allocation
v = np.zeros(dim*Np)         
r = np.zeros(dim*Np)        
p = np.zeros(dim*Np)  
vv  = np.zeros(np.size(L))
phi = np.zeros(np.size(L) )

mu=1.0/(6*np.pi*eta*a)
print '\phi', '   ', '\mu' 
for i in range(np.size(L)):
    v = v*0
    r[0], r[1], r[2] = 0.0, 0.0, 0.0
    p[2]=-1
    
    pRbm = pystokes.periodic.Rbm(a, Np, eta, L[i])   
    pRbm.potDipoleV(v, r, 0.4*p, Nb, Nm)           # and StokesletV module of pystokes
    
    phi[i] = (4*np.pi*a**3)/(3*L[i]**3)
    mu00 = p[2]
    vv[i] = v[2]/mu00     
    print phi[i], '   ', vv[i]


slope, intercept = np.polyfit(phi, vv, 1)
print slope, intercept

plt.plot(phi, vv, '-o', color="#A60628")
plt.xlabel(r'$\phi$', fontsize=20)
#plt.xlim(0.01, np.max(phi**(1.0/3)))
plt.ylabel(r'$\mu/\mu_0$', fontsize=20)
plt.show()

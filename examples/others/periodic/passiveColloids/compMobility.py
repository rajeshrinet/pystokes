# Mobility as a function of volume fraction
import matplotlib.pyplot as plt 
import numpy as np
import pystokes, os, sys
import pyforces

#Parameters
a, eta, dim = 1.0, 1.0/6, 3
Np, Nb, Nm = 1, 1, 8
ta =(4*np.pi/3)**(1.0/3) 
L = ta/np.arange(0.01, 0.4, 0.01)

#Memory allocation
v = np.zeros(dim*Np)         
r = np.zeros(dim*Np)        
F = np.zeros(dim*Np)  
vv  = np.zeros(np.size(L))
phi = np.zeros(np.size(L) )

mu=1.0/(6*np.pi*eta*a)
print ('\phi', '   ', '\mu' )
for i in range(np.size(L)):
    v = v*0
    F = F*0

    r[0], r[1], r[2] = 0.0, 0.0, 0.0

    ff = pyforces.forceFields.Forces(Np)
    ff.sedimentation(F, g=-1)                          
    
    pRbm = pystokes.periodic.Rbm(a, Np, eta, L[i])   
    pRbm.mobilityTT(v, r, F, Nb, Nm)                  
    
    phi[i] = (4*np.pi*a**3)/(3*L[i]**3)
    mu00 = mu*F[2]
    vv[i] = v[2]/mu00     
    print (phi[i], '   ', vv[i])


slope, intercept = np.polyfit(phi**(1.0/3), vv, 1)
print (slope, intercept)

plt.plot(phi**(1.0/3), vv, '-o', color="#A60628")
plt.xlabel(r'$\phi^{1/3}$', fontsize=20)
plt.xlim(0.01, np.max(phi**(1.0/3)))
plt.ylabel(r'$\mu/\mu_0$', fontsize=20)
plt.show()

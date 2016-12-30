#Import stuff
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pystokes
import pyforces
a, eta, dim = 1, 1.0/6, 3
L, Ng = 128, 64

Np, Nt = 51, Ng*Ng
r = np.zeros(3*Np)
p = np.zeros(3*Np)
S = np.zeros(5*Np)
rt = np.zeros(dim*Nt)                   # Memory Allocation for field points

xx = np.linspace(-L, L, Ng)
yy = np.linspace(-L, L, Ng)
X, Y = np.meshgrid(xx, yy)
rt[0:2*Nt] = np.concatenate((X.reshape(Ng*Ng), Y.reshape(Ng*Ng)))


S0 = 1
##linear
r[:Np] = -2*(Np-1) + 4*np.arange(Np)
r[Np:2*Np] = 0

##even
r[:Np] = -1.5*(Np-1) + 3*np.arange(Np)
r[Np:2*Np] = 60*np.cos(np.linspace(-np.pi/2, np.pi/2, Np))


## sine
r[:Np] = -2*(Np-1) + 4*np.arange(Np)
r[Np:2*Np] = 40*np.sin(np.linspace(0, 2*np.pi, Np))


for i in range(Np):
    if i == Np-1:
        p[i]       = r[i]  - r[i-1]  ;
        p[i+Np]  = r[i+Np] - r[i+Np-1] ;
        p[i]   = p[i]   /modp;
        p[i+Np]= p[i+Np]/modp;
    else:
        p[i]       = r[i+1] - r[i];
        p[i+Np]  = r[i+Np+1] -r[i+Np];
        modp = np.sqrt( p[i]*p[i] + p[i+Np]*p[i+Np]);
        p[i]   = p[i]   /modp;
        p[i+Np]= p[i+Np]/modp;

for i in range(Np):
    S[i]       = S0*(p[i]*p[i] -(1.0/3))
    S[i+ Np]   = S0*(p[i + Np]*p[i + Np] -(1.0/3))
    S[i+ 2*Np] = S0*(p[i]*p[i + Np])
    S[i+ 3*Np] = S0*(p[i]*p[i + 2*Np])
    S[i+ 4*Np] = S0*(p[i + Np]*p[i + 2*Np])

####Instantiate the Flow class
uFlow = pystokes.unbounded.Flow(a, eta, Np, Nt)

vv = np.zeros(dim*Nt)                   # Memory Allocation for field Velocities
uFlow.stressletV(vv, rt, r, S)
vx, vy = vv[0:Nt].reshape(Ng, Ng), vv[Nt:2*Nt].reshape(Ng, Ng)


##Plotting
plt.figure()
plt.plot(r[:Np], r[Np:2*Np], marker='o', markerfacecolor='#348ABD', markersize=8 )   # plot the particle at r


plt.streamplot(X, Y, vx, vy, density=1, color="#A60628", arrowstyle='->', arrowsize =1.5)
plt.xlim([-L, L]); plt.ylim([-L, L])
plt.xlabel(r'$x/a$', fontsize=20); plt.ylabel(r'$y/a$', fontsize=20);
plt.show()

#Import stuff
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pystokes
import pyforces
a, eta, dim = 1, 1.0/6, 3
L, Ng = 10, 128

Np, Nt = 1, Ng*Ng
r = np.zeros(3*Np)
F = np.zeros(3*Np)
rt = np.zeros(dim*Nt)                   # Memory Allocation for field points

xx = np.linspace(-L, L, Ng)
yy = np.linspace(-L, L, Ng)
X, Y = np.meshgrid(xx, yy)
rt[Nt:3*Nt] = np.concatenate((X.reshape(Ng*Ng), Y.reshape(Ng*Ng)))


S0 = 1
F[2]=1
#

####Instantiate the Flow class
uFlow = pystokes.unbounded.Flow(a, eta, Np, Nt)

vv = np.zeros(dim*Nt)                   # Memory Allocation for field Velocities
uFlow.stokesletV(vv, rt, r, F)
vx, vy, vz = vv[0:Nt].reshape(Ng, Ng), vv[Nt:2*Nt].reshape(Ng, Ng), vv[2*Nt:3*Nt].reshape(Ng, Ng)


##Plotting
plt.figure()
plt.plot(r[1], r[2], marker='o', markerfacecolor='#348ABD', markersize=32 )   # plot the particle at r
plt.streamplot(X, Y, vy, vz, density=2, color="#A60628", arrowstyle='->', arrowsize =1.5)
plt.xlim([-L, L]); plt.ylim([-L, L])
plt.xlabel(r'$x/a$', fontsize=20); plt.ylabel(r'$y/a$', fontsize=20);
plt.show()

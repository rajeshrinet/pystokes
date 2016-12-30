# Crowley instability
from __future__ import division
import pystokes
import pyforces
import matplotlib.pyplot as plt 
import numpy as np
from mpl_toolkits.mplot3d import Axes3D



a, Np1d, Np = 1, 10, 100            # radius and number of particles
L, dim = 128,  3                    # size and dimensionality and the box
latticeShape = 'square'             # shape of the lattice
v = np.zeros(dim*Np)                # Memory allocation for velocity
r = np.zeros(dim*Np)                # Position vector of the particles
F = np.zeros(dim*Np)                # Forces on the particles
Nb, Nm = 1, 4

rm = pystokes.periodic.Rbm(a, Np, 1.0/6, L)   # instantiate the classes
ff = pyforces.forceFields.Forces(Np)

def initialise(r, Np1d, shape):
    ''' this is the module to initialise the particles on the lattice'''
    if shape=='cube':
        for i in range(Np1d):
            for j in range(Np1d):
                for k in range(Np1d):
                    ii = i*Np1d**2 + j*Np1d + k
                    r[ii]      = L/2 + 3*(-Np1d + 2*i)                  
                    r[ii+Np]   = L/2 + 3*(-Np1d + 2+ 2*j)               
                    r[ii+2*Np] = L/2 + 3*(-Np1d + 2+ 2*k)               
    elif shape=='square':             
        for i in range(Np1d):
            for j in range(Np1d):
                ii = i*Np1d + j
                r[ii]      = L/2 + 4*(-Np1d + 2*i)                  
                r[ii+Np]   = L/2 + 4*(-Np1d + 2+ 2*j)               
        for i in range(Np):
                r[i+2*Np] = L/2
        
    elif shape=='rectangle':             
        for i in range(Np1d):
            for j in range(Np1d):
                ii = i*Np1d + j
                r[ii]      = L/2 + 4*(-Np1d + 2*i)                  
                r[ii+Np]   = L/2 + 4*(-Np1d + 2+ 2*j)               
        for i in range(Np):
                r[i+2*Np] = L/2
    else:
            pass


initialise(r, Np1d, latticeShape)
dt = 0.01
fig = plt.figure()
for tt in range(20):
    ff.sedimentation(F, g=-10)       # call the Sedimentation module of ForceFields 
    rm.stokesletV(v, r, F, Nb, Nm)           # and StokesletV module of pystokes
    print v
    r = (r + v*dt)%L
    x = r[0:Np]
    y = r[Np:2*Np]
    z = r[2*Np:3*Np]
    cc = x*x + y*y + z*z 
    ax3D = fig.add_subplot(111, projection='3d')
    scatCollection = ax3D.scatter(x, y, z, s=30,  c=cc, cmap=plt.cm.spectral )
    ax3D.set_xlim([0, L])
    ax3D.set_ylim([0, L])
    ax3D.set_zlim([0, L])
    #plt.savefig('Time= %04d.png'%(tt))   # if u want to save the plots instead
    print tt
    plt.pause(0.001)
plt.show()

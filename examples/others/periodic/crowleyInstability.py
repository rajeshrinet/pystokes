# Crowley instability
import pystokes 
import matplotlib.pyplot as plt 
import numpy as np
from mpl_toolkits.mplot3d import Axes3D



a, N1d, N = 1, 10, 100            # radius and number of particles
L, dim = 128,  3                    # size and dimensionality and the box
latticeShape = 'square'             # shape of the lattice
v = np.zeros(dim*N)                # Memory allocation for velocity
r = np.zeros(dim*N)                # Position vector of the particles
F = np.zeros(dim*N)                # Forces on the particles
Nb, Nm = 1, 4

rm = pystokes.periodic.Rbm(a, N, 1.0/6, L)   # instantiate the classes
ff = pystokes.forceFields.Forces(N)

def initialise(r, N1d, shape):
    ''' this is the module to initialise the particles on the lattice'''
    if shape=='cube':
        for i in range(N1d):
            for j in range(N1d):
                for k in range(N1d):
                    ii = i*N1d**2 + j*N1d + k
                    r[ii]      = L/2 + 3*(-N1d + 2*i)                  
                    r[ii+N]   = L/2 + 3*(-N1d + 2+ 2*j)               
                    r[ii+2*N] = L/2 + 3*(-N1d + 2+ 2*k)               
    elif shape=='square':             
        for i in range(N1d):
            for j in range(N1d):
                ii = i*N1d + j
                r[ii]      = L/2 + 4*(-N1d + 2*i)                  
                r[ii+N]   = L/2 + 4*(-N1d + 2+ 2*j)               
        for i in range(N):
                r[i+2*N] = L/2
        
    elif shape=='rectangle':             
        for i in range(N1d):
            for j in range(N1d):
                ii = i*N1d + j
                r[ii]      = L/2 + 4*(-N1d + 2*i)                  
                r[ii+N]   = L/2 + 4*(-N1d + 2+ 2*j)               
        for i in range(N):
                r[i+2*N] = L/2
    else:
            pass


initialise(r, N1d, latticeShape)
dt = 0.01
fig = plt.figure()
for tt in range(32):
    ff.sedimentation(F, g=-10)       # call the Sedimentation module of ForceFields 
    v=v*0                            # setting v=0 in each time step
    rm.mobilityTT(v, r, F, Nb, Nm)   # and StokesletV module of pystokes
    r = (r + v*dt)%L
    x = r[0:N]
    y = r[N:2*N]
    z = r[2*N:3*N]
    cc = x*x + y*y + z*z 
    ax3D = fig.add_subplot(111, projection='3d')
    scatCollection = ax3D.scatter(x, y, z, s=30,  c=cc, cmap=plt.cm.RdBu )
    ax3D.set_xlim([0, L])
    ax3D.set_ylim([0, L])
    ax3D.set_zlim([0, L])
    #plt.savefig('Time= %04d.png'%(tt))   # if u want to save the plots instead
    print (tt)
    plt.pause(0.0000001)
plt.show()
